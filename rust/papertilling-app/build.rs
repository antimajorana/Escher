//! Embeds `assets/icon.ico` as the native Win32 executable icon (`RT_GROUP_ICON`)
//! when building `papertilling-app` on Windows MSVC, with zero extra crate dependencies.

use std::env;
use std::fs;
use std::path::PathBuf;

fn push_res_record(out: &mut Vec<u8>, rtype: u16, rname: u16, mem_flags: u16, lang: u16, data: &[u8]) {
    let data_size = u32::try_from(data.len()).expect("icon resource too large");
    let header_size: u32 = 32;
    out.extend_from_slice(&data_size.to_le_bytes());
    out.extend_from_slice(&header_size.to_le_bytes());
    out.extend_from_slice(&0xFFFFu16.to_le_bytes());
    out.extend_from_slice(&rtype.to_le_bytes());
    out.extend_from_slice(&0xFFFFu16.to_le_bytes());
    out.extend_from_slice(&rname.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // DataVersion
    out.extend_from_slice(&mem_flags.to_le_bytes());
    out.extend_from_slice(&lang.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // Version
    out.extend_from_slice(&0u32.to_le_bytes()); // Characteristics
    out.extend_from_slice(data);
    while !out.len().is_multiple_of(4) {
        out.push(0);
    }
}

fn ico_to_win32_res(ico: &[u8]) -> Vec<u8> {
    assert!(ico.len() >= 6, "invalid .ico header");
    let reserved = u16::from_le_bytes([ico[0], ico[1]]);
    let ico_type = u16::from_le_bytes([ico[2], ico[3]]);
    let count = usize::from(u16::from_le_bytes([ico[4], ico[5]]));
    assert!(reserved == 0 && ico_type == 1 && count > 0, "unexpected .ico header");
    assert!(ico.len() >= 6 + count * 16, "truncated .ico directory");

    let mut res = Vec::with_capacity(ico.len() + 256);
    // Mandatory 32-byte null prelude record in Win32 .res files.
    push_res_record(&mut res, 0, 0, 0, 0, &[]);

    let mut grp_dir = Vec::with_capacity(6 + count * 14);
    grp_dir.extend_from_slice(&ico[0..6]);

    for i in 0..count {
        let ent = &ico[6 + i * 16..6 + (i + 1) * 16];
        let bytes_in_res = u32::from_le_bytes([ent[8], ent[9], ent[10], ent[11]]) as usize;
        let img_offset = u32::from_le_bytes([ent[12], ent[13], ent[14], ent[15]]) as usize;
        let img_data = &ico[img_offset..img_offset + bytes_in_res];
        let icon_id = u16::try_from(i + 1).expect("too many icon entries");

        // RT_ICON = 3
        push_res_record(&mut res, 3, icon_id, 0x1010, 0x0409, img_data);

        // GRPICONDIRENTRY: first 12 bytes of ICONDIRENTRY + 2-byte resource ID.
        grp_dir.extend_from_slice(&ent[0..12]);
        grp_dir.extend_from_slice(&icon_id.to_le_bytes());
    }

    // RT_GROUP_ICON = 14 (resource ID 1 = main executable icon)
    push_res_record(&mut res, 14, 1, 0x1030, 0x0409, &grp_dir);
    res
}

fn main() {
    println!("cargo:rerun-if-changed=assets/icon.ico");
    println!("cargo:rerun-if-changed=assets/icon.png");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    if target_os == "windows" && target_env == "msvc" {
        let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
        let ico_bytes = fs::read(manifest_dir.join("assets/icon.ico")).expect("read assets/icon.ico");
        let res_bytes = ico_to_win32_res(&ico_bytes);
        let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
        let res_path = out_dir.join("papertilling_icon.res");
        fs::write(&res_path, res_bytes).expect("write papertilling_icon.res");
        println!("cargo:rustc-link-arg-bins={}", res_path.display());
    }
}
