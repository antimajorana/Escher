use super::*;

struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        loop {
            let id = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!("papertilling-aliases-{}-{id}", std::process::id()));
            match std::fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(e) => panic!("create test directory: {e}"),
            }
        }
    }

    fn file(&self) -> PathBuf { self.0.join("nested").join("lattice_names.json") }
}

impl Drop for TempDir {
    fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
}

fn keys() -> (LatticeAliasKey, LatticeAliasKey) {
    (
        LatticeAliasKey::new("p4g", "bonds", 1, "[|D|=2] (3,4:0,1,1) (4,4:1,0,0)"),
        LatticeAliasKey::new("p6m", "named", 13, "[|D|=2] (3,4:0,0,1) (6,4:1,1,0)"),
    )
}

#[test]
fn alias_save_load_reset_roundtrip_and_readonly_graceful_fallback() {
    let dir = TempDir::new();
    let file = dir.file();
    let (k1, k2) = keys();
    let mut store = AliasStore::load_from_path(file.clone());
    assert_eq!(store.get(&k1), None);
    store.set(k1.clone(), "  My Rotating Squares  ", "p4g #1 (1,2)").unwrap();
    store.set(k2.clone(), "  Δοκιμή 雪 🧪  ", "Kagomé").unwrap();

    let mut reloaded = AliasStore::load_from_path(file.clone());
    assert_eq!(reloaded.get(&k1), Some("My Rotating Squares"));
    assert_eq!(reloaded.get(&k2), Some("Δοκιμή 雪 🧪"));
    reloaded.set(k2.clone(), "Kagomé", "Kagomé").unwrap();
    reloaded.reset(&k1).unwrap();
    let reset = AliasStore::load_from_path(file.clone());
    assert_eq!(reset.get(&k1), None);
    assert_eq!(reset.get(&k2), None);

    let blocked = file.join("cannot_create_under_file").join("lattice_names.json");
    let mut ro_store = AliasStore::load_from_path(blocked);
    assert!(ro_store.set(k1.clone(), "Session Only Alias", "default").is_err());
    assert_eq!(ro_store.get(&k1), Some("Session Only Alias"));
}

#[test]
fn independent_stale_stores_merge_only_the_changed_alias_including_resets() {
    let dir = TempDir::new();
    let file = dir.file();
    let (k1, k2) = keys();
    let mut a = AliasStore::load_from_path(file.clone());
    let mut b = AliasStore::load_from_path(file.clone());
    a.set(k1.clone(), "First", "default").unwrap();
    b.set(k2.clone(), "Second", "default").unwrap();
    let both = AliasStore::load_from_path(file.clone());
    assert_eq!(both.get(&k1), Some("First"));
    assert_eq!(both.get(&k2), Some("Second"));

    a.set(k1.clone(), "New First", "default").unwrap();
    b.reset(&k2).unwrap(); // Must not restore b's stale snapshot of k1.
    let after = AliasStore::load_from_path(file.clone());
    assert_eq!(after.get(&k1), Some("New First"));
    assert_eq!(after.get(&k2), None);
    a.reset(&k1).unwrap(); // Must not resurrect the independently deleted k2.
    assert!(read_aliases(&file).unwrap().is_empty());
}

#[test]
fn concurrent_rule_keys_share_one_alias_without_removing_unrelated_entries() {
    let dir = TempDir::new();
    let file = dir.file();
    let (k1, k2) = keys();
    let mut other_rule = k1.clone();
    other_rule.rule = String::from("sheet");
    other_rule.dsymbol_key_or_index = format!("#9:{}", k1.dsymbol_suffix().unwrap());
    let mut a = AliasStore::load_from_path(file.clone());
    let mut b = AliasStore::load_from_path(file.clone());
    a.set(k1.clone(), "Old", "default").unwrap();
    a.set(k2.clone(), "Unrelated", "default").unwrap();
    b.set(other_rule.clone(), "Updated", "default").unwrap();
    let after = AliasStore::load_from_path(file.clone());
    assert_eq!(after.get(&k1), Some("Updated"));
    assert_eq!(after.get(&k2), Some("Unrelated"));
    assert_eq!(read_aliases(&file).unwrap().len(), 2);
    a.reset(&k1).unwrap();
    let reset = AliasStore::load_from_path(file);
    assert_eq!(reset.get(&other_rule), None);
    assert_eq!(reset.get(&k2), Some("Unrelated"));
}

#[test]
fn busy_lock_is_nonblocking_and_a_failed_session_edit_is_not_saved_by_an_unrelated_edit() {
    let dir = TempDir::new();
    let file = dir.file();
    let (k1, k2) = keys();
    let mut a = AliasStore::load_from_path(file.clone());
    a.set(k1.clone(), "On disk", "default").unwrap();
    let before = std::fs::read(&file).unwrap();
    let lock = OpenOptions::new().read(true).write(true).open(with_suffix(&file, ".lock")).unwrap();
    lock.try_lock().unwrap();

    let start = std::time::Instant::now();
    assert!(a.set(k1.clone(), "Session only", "default").is_err());
    assert!(start.elapsed() < std::time::Duration::from_secs(2), "must not wait for the lock");
    assert_eq!(a.get(&k1), Some("Session only"));
    assert_eq!(std::fs::read(&file).unwrap(), before);
    drop(lock);
    a.set(k2.clone(), "Second", "default").unwrap();
    let disk = AliasStore::load_from_path(file);
    assert_eq!(disk.get(&k1), Some("On disk"));
    assert_eq!(disk.get(&k2), Some("Second"));
    assert_eq!(a.get(&k1), Some("Session only"));
}

#[test]
fn failed_staged_write_or_rename_preserves_the_old_file_and_cleans_up() {
    let dir = TempDir::new();
    let file = dir.file();
    let (k1, _) = keys();
    AliasStore::load_from_path(file.clone()).set(k1, "Original", "default").unwrap();
    let before = std::fs::read(&file).unwrap();
    let failed = replace_atomically(&file, |out| {
        out.write_all(b"{partial")?;
        Err(io::Error::other("injected write failure"))
    });
    assert!(failed.is_err());
    assert_eq!(std::fs::read(&file).unwrap(), before);
    assert!(!std::fs::read_dir(file.parent().unwrap()).unwrap()
        .any(|e| e.unwrap().path().extension().is_some_and(|ext| ext == "tmp")));

    let dest_dir = file.parent().unwrap().join("not_a_file");
    std::fs::create_dir(&dest_dir).unwrap();
    assert!(replace_atomically(&dest_dir, |out| out.write_all(b"new")).is_err());
    assert!(dest_dir.is_dir());
    assert!(!std::fs::read_dir(file.parent().unwrap()).unwrap()
        .any(|e| e.unwrap().path().extension().is_some_and(|ext| ext == "tmp")));
}

#[test]
fn invalid_or_future_version_files_are_not_overwritten() {
    let dir = TempDir::new();
    let file = dir.file();
    std::fs::create_dir_all(file.parent().unwrap()).unwrap();
    let (k1, _) = keys();
    for raw in ["{broken json", r#"{"version":999,"entries":[]}"#] {
        std::fs::write(&file, raw).unwrap();
        let mut store = AliasStore::load_from_path(file.clone());
        assert_eq!(store.get(&k1), None);
        assert!(store.set(k1.clone(), "Session", "default").is_err());
        assert_eq!(store.get(&k1), Some("Session"));
        assert_eq!(std::fs::read_to_string(&file).unwrap(), raw);
    }
}

#[test]
fn read_only_file_is_not_replaced_even_when_the_directory_is_writable() {
    let dir = TempDir::new();
    let file = dir.file();
    let (k1, _) = keys();
    let mut store = AliasStore::load_from_path(file.clone());
    store.set(k1.clone(), "Original", "default").unwrap();
    let before = std::fs::read(&file).unwrap();
    let original_permissions = std::fs::metadata(&file).unwrap().permissions();
    let mut ro = original_permissions.clone();
    ro.set_readonly(true);
    std::fs::set_permissions(&file, ro).unwrap();
    let result = store.set(k1.clone(), "Session", "default");
    // Restore before assertions so a failing test still leaves its owned temporary tree removable.
    std::fs::set_permissions(&file, original_permissions).unwrap();
    assert!(result.is_err());
    assert_eq!(store.get(&k1), Some("Session"));
    assert_eq!(std::fs::read(&file).unwrap(), before);
}
