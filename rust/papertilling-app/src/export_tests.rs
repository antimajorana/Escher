//! SVG / PNG export of the scenes of the four tabs: files written to a temporary folder and read back
//! (SVG: well-formed XML, size and viewBox, no non-finite number; PNG: decoded with the `png` crate).

use std::path::{Path, PathBuf};
use std::time::Instant;

use papertilling_core::defect::surgery::DefectSpec;
use papertilling_core::types::{BuiltCrystal, Cell2D};

use super::*;
use crate::backend::{AppBackend, DefectSelectionUi, SurfaceBuildParamsUi, SurfaceTargetUi};
use crate::canvas2d::{CanvasMarker, MarkerKind};
use crate::defects_tab::tests::empty_crystal;
use crate::live_combine::{CombineBuildOptions, CombineDrawMode};
use crate::scene2d::site_palette;

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!("papertilling-export-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Minimal XML well-formedness: balanced tags, quoted attributes, escaped text; returns the root's attributes.
fn check_xml(s: &str) -> Result<Vec<(String, String)>, String> {
    let (mut stack, mut root, mut rest) = (Vec::<String>::new(), None, s);
    let check_text = |t: &str| -> Result<(), String> {
        let mut parts = t.split('&');
        parts.next();
        for p in parts {
            let ent = p.split(';').next().unwrap_or("");
            if !["amp", "lt", "gt", "quot", "apos"].contains(&ent) {
                return Err(format!("bad entity &{ent}"));
            }
        }
        if t.contains('>') { Err(format!("raw '>' in text {t:?}")) } else { Ok(()) }
    };
    while let Some(lt) = rest.find('<') {
        check_text(&rest[..lt])?;
        let gt = rest[lt..].find('>').ok_or("unclosed tag")? + lt;
        let tag = &rest[lt + 1..gt];
        rest = &rest[gt + 1..];
        if let Some(pi) = tag.strip_prefix('?') {
            if !pi.ends_with('?') { return Err(format!("bad declaration {tag}")); }
            continue;
        }
        if let Some(name) = tag.strip_prefix('/') {
            if stack.pop().as_deref() != Some(name.trim()) { return Err(format!("unbalanced </{name}>")); }
            continue;
        }
        let self_closing = tag.ends_with('/');
        let body = tag.trim_end_matches('/');
        let name = body.split_whitespace().next().ok_or("empty tag")?.to_string();
        let mut attrs = Vec::new();
        let mut a = body[name.len()..].trim_start();
        while !a.is_empty() {
            let eq = a.find("=\"").ok_or_else(|| format!("unquoted attribute in <{name}>: {a}"))?;
            let key = a[..eq].trim().to_string();
            let end = a[eq + 2..].find('"').ok_or("open quote")? + eq + 2;
            let val = a[eq + 2..end].to_string();
            if val.contains('<') || key.is_empty() || key.contains(char::is_whitespace) { return Err(format!("bad attribute {key}={val}")); }
            attrs.push((key, val));
            a = a[end + 1..].trim_start();
        }
        if stack.is_empty() {
            if root.is_some() { return Err("two root elements".into()); }
            root = Some(attrs.clone());
        }
        if !self_closing {
            stack.push(name);
        }
    }
    check_text(rest)?;
    if !stack.is_empty() || !rest.trim().is_empty() { return Err(format!("unclosed {stack:?}")); }
    root.ok_or_else(|| "no root".into())
}

fn check_svg_file(path: &Path, w: u32, h: u32) -> String {
    let s = std::fs::read_to_string(path).unwrap();
    let root = check_xml(&s).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let get = |k: &str| root.iter().find(|(a, _)| a == k).map(|(_, v)| v.clone());
    assert_eq!(get("xmlns").as_deref(), Some("http://www.w3.org/2000/svg"));
    assert_eq!(get("width"), Some(w.to_string()));
    assert_eq!(get("height"), Some(h.to_string()));
    assert_eq!(get("viewBox"), Some(format!("0 0 {w} {h}")));
    let non_finite = s.split(|c: char| c == '"' || c == ',' || c.is_whitespace()).any(|t| ["NaN", "inf", "-inf"].contains(&t));
    assert!(!non_finite, "{}: non-finite number", path.display());
    s
}

fn decode_png(path: &Path) -> (u32, u32, Vec<u8>) {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
    let dec = png::Decoder::new(std::io::Cursor::new(bytes));
    let mut reader = dec.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).unwrap();
    assert_eq!((info.color_type, info.bit_depth), (png::ColorType::Rgba, png::BitDepth::Eight));
    buf.truncate(info.buffer_size());
    (info.width, info.height, buf)
}

fn px(img: &[u8], w: u32, x: u32, y: u32) -> [u8; 4] {
    let i = ((y * w + x) * 4) as usize;
    [img[i], img[i + 1], img[i + 2], img[i + 3]]
}

/// Every extension case, read back. Returns the PNG pixels.
fn export_all_ways(scene: &Scene2D, markers: &[CanvasMarker], dir: &Path, stem: &str) -> Vec<u8> {
    for name in [format!("{stem}.svg"), format!("{stem}.SVG")] {
        let msg = export_scene_with_markers(scene, markers, &dir.join(&name)).unwrap();
        assert!(msg.contains("SVG") && msg.contains(&name), "{msg}");
        check_svg_file(&dir.join(&name), SVG_SIZE, SVG_SIZE);
    }
    let msg = export_scene_with_markers(scene, markers, &dir.join(stem)).unwrap();
    assert!(msg.contains(&format!("{stem}.svg")), "no extension: {msg}");
    assert!(!dir.join(stem).exists());
    check_svg_file(&dir.join(format!("{stem}.svg")), SVG_SIZE, SVG_SIZE);
    let err = export_scene_with_markers(scene, markers, &dir.join(format!("{stem}.jpg"))).unwrap_err();
    assert!(err.contains(".jpg") && !dir.join(format!("{stem}.jpg")).exists(), "{err}");
    let mut last = Vec::new();
    for name in [format!("{stem}.png"), format!("{stem}.PNG")] {
        let msg = export_scene_with_markers(scene, markers, &dir.join(&name)).unwrap();
        assert!(msg.contains("PNG"), "{msg}");
        let (w, h, img) = decode_png(&dir.join(&name));
        assert_eq!((w, h), (PNG_SIZE, PNG_SIZE));
        assert_eq!(px(&img, w, 1, h - 2), BACKGROUND, "corner is background");
        last = img;
    }
    last
}

fn count_px(img: &[u8], pred: impl Fn([u8; 4]) -> bool) -> usize {
    img.as_chunks::<4>().0.iter().filter(|c| pred(**c)).count()
}

#[test]
fn every_tab_scene_exports_valid_svg_and_png() {
    let backend = AppBackend::new();
    let dir = TempDir::new("tabs");
    let kagome = backend.find_named("kagome").unwrap();
    let square = backend.find_named("square").unwrap();

    let (cell, _) = backend.build_cell(kagome, &[]);
    let gallery = Scene2D::PeriodicCell { cell, repeat: 3, show_unit_cell: true };
    let img = export_all_ways(&gallery, &[], &dir.0, "gallery");
    assert!(count_px(&img, |c| c[..3] == site_palette(0)[..3]) > 50, "site colour as on screen");

    for target in [SurfaceTargetUi::Torus, SurfaceTargetUi::Genus2] {
        let (mesh, _) = backend.build_surface(kagome, target, &SurfaceBuildParamsUi::default()).unwrap();
        let net = Scene2D::SurfaceNet { mesh };
        let img = export_all_ways(&net, &[], &dir.0, &format!("net_{}", target.label().replace(' ', "_")));
        assert!(count_px(&img, |c| c[..3] == TITLE[..3]) > 30, "title text drawn");
    }

    let cat = backend.defect_catalogue(square);
    let wc = cat.walls.iter().position(|w| w.id == "Wc1").unwrap();
    for (stem, sel) in [("wall_wc1", DefectSelectionUi::Wall(wc)), ("disclination", DefectSelectionUi::Disclination(0))] {
        let (crystal, r) = backend.build_defect(square, &sel, 3.8, true);
        let scene = Scene2D::CrystalPatch { crystal, title: r.title, note: String::from("rigid picture: 2 crossings & <1> fold") };
        let svg_path = dir.0.join(format!("{stem}_n.svg"));
        export_scene_to_path(&scene, &svg_path).unwrap();
        let svg = check_svg_file(&svg_path, SVG_SIZE, SVG_SIZE);
        assert!(svg.contains("&amp; &lt;1&gt; fold") && svg.contains("—"), "escaped note, UTF-8 title");
        let img = export_all_ways(&scene, &[], &dir.0, stem);
        assert!(count_px(&img, |c| c[..3] == NOTE_TEXT[..3]) > 30, "note text drawn in the banner");
    }

    let info = backend.combine_lattice_info(square);
    let specs = [DefectSpec::disclination(90.0, [0.0, 0.0]), DefectSpec::disclination(-90.0, [0.5, 0.5])];
    let opts = CombineBuildOptions { draw_mode: CombineDrawMode::FastRelaxed, full_planar_repair: false, auto_radius: true, manual_radius: info.r_auto_base, core_radius: 2 };
    let out = crate::live_combine::build_combined_crystal(square, &info, &specs, &opts, false);
    let markers = [
        CanvasMarker { index: 0, pos: [0.0, 0.0], kind: MarkerKind::DisclinationPlus, label: String::from("+90°") },
        CanvasMarker { index: 1, pos: [0.5, 0.5], kind: MarkerKind::DisclinationMinus, label: String::from("-90°") },
    ];
    let scene = Scene2D::CrystalPatch { crystal: out.crystal, title: String::from("square — combined"), note: out.note };
    let img = export_all_ways(&scene, &markers, &dir.0, "combine");
    for m in &markers {
        let c = marker_colors(m.kind);
        assert!(count_px(&img, |p| p[..3] == c) > 100, "{:?} pin drawn in its on-screen colour", m.kind);
    }
    let svg = render_scene_svg_with_markers(&scene, &markers, 400, 300);
    check_xml(&svg).unwrap();
    assert_eq!(svg.matches("r=\"9.5\"").count(), 2);
}

fn crystal_with(pos: Vec<[f64; 2]>, edges: Vec<(usize, usize)>) -> BuiltCrystal {
    let n = pos.len();
    let mut c = empty_crystal();
    c.ref_pos = pos.clone();
    c.pos = pos;
    c.basis_idx = vec![0; n];
    c.side = vec![None; n];
    c.adj = vec![Vec::new(); n];
    c.edges = edges;
    c
}

#[test]
fn empty_and_non_finite_scenes_export_cleanly() {
    let dir = TempDir::new("edge");
    let empty_cell = Cell2D { a1: [1.0, 0.0], a2: [0.0, 1.0], basis: Vec::new(), bonds: Vec::new() };
    let scenes = [
        Scene2D::PeriodicCell { cell: empty_cell, repeat: 3, show_unit_cell: false },
        Scene2D::CrystalPatch { crystal: empty_crystal(), title: String::new(), note: String::new() },
        Scene2D::CrystalPatch {
            crystal: crystal_with(vec![[0.0, 0.0], [f64::NAN, 1.0], [1.0, f64::INFINITY], [1.0, 0.0]], vec![(0, 1), (0, 3), (2, 3)]),
            title: String::from("t"),
            note: String::new(),
        },
    ];
    for (k, scene) in scenes.iter().enumerate() {
        let img = export_all_ways(scene, &[], &dir.0, &format!("edge{k}"));
        if k < 2 {
            assert_eq!(count_px(&img, |c| c != BACKGROUND), 0, "empty scene {k}: background only");
        } else {
            let svg = std::fs::read_to_string(dir.0.join("edge2.svg")).unwrap();
            assert_eq!(svg.matches("<line ").count(), 1, "only the finite bond 0-3 is drawn");
        }
    }
}

#[test]
fn png_sizes_out_of_range_are_errors_and_large_scenes_are_fast() {
    let backend = AppBackend::new();
    let square = backend.find_named("square").unwrap();
    let (crystal, r) = backend.build_defect(square, &DefectSelectionUi::Dislocation(0), 3.8, false);
    let scene = Scene2D::CrystalPatch { crystal, title: r.title, note: r.note };
    for (w, h) in [(0, 10), (10, 0), (100_000, 100_000), (8193, 1), (8000, 8000)] {
        assert!(encode_scene_png(&scene, w, h).is_err(), "{w}x{h} must be refused");
    }
    let wide = encode_scene_png(&scene, 8192, 2).unwrap();
    assert_eq!(&wide[..8], b"\x89PNG\r\n\x1a\n");
    assert!(encode_scene_png(&scene, 1, 1).is_ok() && render_scene_svg(&scene, 1, 1).ends_with("</svg>\n"));

    let info = papertilling_core::defect::core_graph::LatticeCoreInfo::new(&backend.build_cell(square, &[]).0);
    let big = crate::live_ops::build_pristine_disk(&info, 45.0);
    assert!(big.pos.len() > 6000, "{} sites", big.pos.len());
    let scene = Scene2D::CrystalPatch { crystal: big, title: String::from("big"), note: String::new() };
    let t = Instant::now();
    let png = encode_scene_png(&scene, PNG_SIZE, PNG_SIZE).unwrap();
    let svg = render_scene_svg(&scene, SVG_SIZE, SVG_SIZE);
    println!("# time {} sites: PNG {:.2} s ({} bytes), SVG {} bytes", scene_sites(&scene), t.elapsed().as_secs_f64(), png.len(), svg.len());
    assert!(t.elapsed().as_secs_f64() < 20.0);
    check_xml(&svg).unwrap();
}

fn scene_sites(scene: &Scene2D) -> usize {
    compile_scene(scene).site_count
}

#[test]
fn translucent_polygon_is_blended_once_per_pixel() {
    // (0, 0, 0, 128) over the background 248: 248 x (1 - 128/255) = 123.5 -> 124 (hand computed), the same at
    // every inside pixel, including those on a diagonal of the square (a triangle-fan fill blended them twice).
    let mut img = Raster::new(40, 40, BACKGROUND).unwrap();
    img.fill_polygon(&[[5.0, 5.0], [35.0, 5.0], [35.0, 35.0], [5.0, 35.0]], [0, 0, 0, 128]);
    for (x, y) in [(10, 10), (20, 20), (30, 30), (20, 10), (34, 34), (5, 5)] {
        assert_eq!(img.rgba[(y * 40 + x) * 4], 124, "pixel ({x}, {y})");
    }
    assert_eq!(img.rgba[(2 * 40 + 2) * 4], BACKGROUND[0]);
}
