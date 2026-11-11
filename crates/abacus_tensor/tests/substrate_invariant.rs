//! Substrate-invariant enforcement (Golden Source:
//! `docs/golden_source_operational_substrate_invariant.md`).
//!
//! Mechanical pin of §§2-4: scans non-test crate sources and fails on
//! (a) CSV-read calls outside the closed allowed list, (b) CSV writes
//! outside examples/tests reporting paths, (c) third-party
//! dataframe/graph/table dependencies in Cargo.toml files.

use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn rs_files(dir: &std::path::Path, out: &mut Vec<PathBuf>) {
    let rd = std::fs::read_dir(dir).unwrap();
    for e in rd {
        let p = e.unwrap().path();
        if p.is_dir() {
            let name = p.file_name().unwrap().to_string_lossy().to_string();
            if name == "target" || name == "vendor" {
                continue;
            }
            rs_files(&p, out);
        } else if p.extension().map(|e| e == "rs").unwrap_or(false) {
            out.push(p);
        }
    }
}

fn is_test_or_example(path: &std::path::Path) -> bool {
    path.components().any(|c| {
        let s = c.as_os_str().to_string_lossy().to_string();
        s == "tests" || s == "examples" || s == "fixtures"
    })
}

fn rel(root: &std::path::Path, p: &std::path::Path) -> String {
    p.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/")
}

/// Allowed CSV-read sites (§2 closed list): row-14/24 builders, CLI
/// ingestion bins, API ingestion routes, build.rs-adjacent vendor patching
/// excluded (no csv there), plus any tests/examples/fixtures.
fn csv_read_allowed(r: &str) -> bool {
    const ALLOW: [&str; 9] = [
        "crates/abacus_tensor/src/coo/builder.rs",
        "crates/abacus_tensor/src/sidecar/builder.rs",
        "crates/abacus_tensor/src/samld/stage.rs",
        "crates/abacus_tensor/src/samld/convert.rs",
        "crates/abacus_tensor/src/bin/build_samld.rs",
        "crates/abacus_tensor/src/bin/build_sidecar.rs",
        "crates/abacus_tensor/src/bin/build_samld_bin.rs",
        "crates/abacus_tensor/src/bin/ingestion_benchmark.rs",
        "crates/abacus_netapi/src/routes.rs",
    ];
    ALLOW.iter().any(|a| r == *a || r.ends_with(a))
}

/// A CSV read = actually opening a `.csv` file, embedding one, or
/// invoking a CSV-ingesting builder call. Bare re-exports (`pub use`)
/// and prose/comments do not count.
fn has_csv_read(src: &str, path: &str) -> bool {
    if path.ends_with("coo/mod.rs")
        || path.ends_with("sidecar/mod.rs")
        || path.ends_with("samld/mod.rs")
        || path.ends_with("samld/convert.rs")
    {
        return false; // pure re-export shims of the row-14/24 builders
    }
    for line in src.lines() {
        let t = line.trim();
        if t.starts_with("//") || t.starts_with("//!") || t.starts_with('*') {
            continue;
        }
        if t.contains("build_from_csv(")
            || t.contains("build_sidecar_from_csv(")
            || t.contains("build_from_reader(")
            || t.contains("build_sidecar(std::io::Cursor")
            || t.contains("build_sidecar(CSV")
            || (t.contains(".csv") && (t.contains("open") || t.contains("read") || t.contains("include_str")))
        {
            return true;
        }
    }
    false
}

#[test]
fn csv_reads_confined_to_ingestion_boundary() {
    let root = workspace_root();
    let mut files = Vec::new();
    rs_files(&root.join("crates"), &mut files);
    let mut violations = Vec::new();
    for f in &files {
        if is_test_or_example(f) {
            continue;
        }
        let r = rel(&root, f);
        if r.contains("support/") {
            continue;
        }
        let src = std::fs::read_to_string(f).unwrap();
        if has_csv_read(&src, &r) && !csv_read_allowed(&r) {
            violations.push(r);
        }
    }
    assert!(
        violations.is_empty(),
        "CSV reads outside the ingestion boundary (§2): {violations:?}"
    );
}

#[test]
fn csv_writes_confined_to_reporting_paths() {
    let root = workspace_root();
    let mut files = Vec::new();
    rs_files(&root.join("crates"), &mut files);
    let mut violations = Vec::new();
    for f in &files {
        let r = rel(&root, f);
        let src = std::fs::read_to_string(f).unwrap();
        // CSV writes allowed in: examples/tests reporting paths, and the
        // closed ingestion-CLI list (build_samld/build_sidecar write
        // tensor.csf/sidecar/stats artifacts; the benchmark writes its
        // report/capped-CSV scratch) — all ingestion-boundary outputs.
        let is_ingestion_bin = r.starts_with("crates/abacus_tensor/src/bin/");
        let writes_csv = (src.contains("fs::write") || src.contains("File::create"))
            && (src.contains(".csv") || src.contains("k{}.csv"));
        if writes_csv && !is_test_or_example(f) && !is_ingestion_bin {
            violations.push(r);
        }
    }
    assert!(
        violations.is_empty(),
        "CSV writes outside examples/tests reporting paths (§2): {violations:?}"
    );
}

#[test]
fn no_third_party_operational_substrate_dependency() {
    let root = workspace_root();
    let banned = [
        "polars",
        "ndarray",
        "dataframe",
        "petgraph",
        "csv",
        "arrow",
        "parquet",
    ];
    let mut violations = Vec::new();
    let mut manifests = Vec::new();
    rs_files(&root.join("crates"), &mut manifests);
    for dir in [
        root.join("crates/abacus_tensor"),
        root.join("crates/abacus_execution"),
        root.join("crates/abacus_netapi"),
        root.join("crates/abacus_ontology"),
        root.join("crates/abacus_orchestration"),
        root.join("crates/abacus_temporal_gnn"),
    ] {
        let toml = dir.join("Cargo.toml");
        if !toml.exists() {
            continue;
        }
        let src = std::fs::read_to_string(&toml).unwrap();
        let lower = src.to_lowercase();
        for b in banned {
            // Match a dependency declaration line, not prose in comments.
            for line in lower.lines() {
                let t = line.trim();
                if t.starts_with('#') {
                    continue;
                }
                if t.starts_with(&format!("{b} "))
                    || t.starts_with(&format!("{b}="))
                    || t.starts_with(&format!("{b}-"))
                    || t.starts_with(&format!("{b}_"))
                {
                    violations.push(format!("{}: {t}", toml.display()));
                }
            }
        }
    }
    let _ = manifests;
    assert!(
        violations.is_empty(),
        "third-party operational substrate dependency (§§3-4): {violations:?}"
    );
}
