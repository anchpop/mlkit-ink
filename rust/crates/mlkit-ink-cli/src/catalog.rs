//! The pack catalog: resolve a name to a URL, download it, verify it, unzip it.
//!
//! `manifest.json` ships inside the `digital-ink-recognition` AAR at
//! `assets/manifest.json`. Every pack is a plain, unauthenticated
//! `https://dl.google.com/handwriting/models/` URL with a published sha1, which
//! is why this project needs no credentials and no SDK.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use sha1::{Digest, Sha1};

#[derive(Debug, Deserialize)]
struct Manifest {
    packs: Vec<Pack>,
}

#[derive(Debug, Deserialize)]
pub struct Pack {
    pub name: String,
    pub download_urls: Vec<String>,
    pub sha1_checksum: String,
}

pub struct Catalog {
    packs: Vec<Pack>,
    models_dir: PathBuf,
}

impl Catalog {
    pub fn load(root: &Path) -> Result<Self> {
        let path = root.join("manifest.json");
        let text =
            fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        let manifest: Manifest =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        Ok(Catalog {
            packs: manifest.packs,
            models_dir: root.join("models"),
        })
    }

    pub fn get(&self, name: &str) -> Result<&Pack> {
        self.packs
            .iter()
            .find(|p| p.name == name)
            .with_context(|| format!("pack {name:?} is not in the catalog"))
    }

    /// Download if needed, then extract. Returns the extracted member paths.
    ///
    /// Extraction is idempotent *and* atomic, and both halves matter: the 22 MB
    /// language model is read through an mmap, so rewriting a file in place
    /// would truncate a live mapping out from under another reader and kill it
    /// with SIGBUS. Members that are already correct are left alone, and
    /// anything we do write lands in a temp file that is then renamed, so an
    /// existing mapping keeps pointing at the old inode.
    pub fn ensure(&self, name: &str, quiet: bool) -> Result<Vec<PathBuf>> {
        let pack = self.get(name)?;
        // Two language packs can ship members with the same basename, so each
        // pack gets its own directory.
        let dest = self.models_dir.join(name);
        // The archive lives beside its extracted members, matching the layout the
        // Python tooling already populated so both share one cache.
        let archive_path = dest.join(format!("{name}.zip"));
        self.download(pack, &archive_path, quiet)?;

        let file = fs::File::open(&archive_path)
            .with_context(|| format!("opening {}", archive_path.display()))?;
        let mut archive = zip::ZipArchive::new(file)
            .with_context(|| format!("reading {}", archive_path.display()))?;

        let mut extracted = Vec::new();
        for i in 0..archive.len() {
            let mut member = archive.by_index(i)?;
            if member.is_dir() {
                continue;
            }
            let name = member
                .enclosed_name()
                .context("zip member escapes the destination directory")?;
            let target = dest.join(name);
            if matches!(fs::metadata(&target), Ok(meta) if meta.len() == member.size()) {
                extracted.push(target);
                continue;
            }
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent)?;
            }
            let temp = temp_path(&target, "x");
            let mut out = fs::File::create(&temp)?;
            std::io::copy(&mut member, &mut out)?;
            out.sync_all()?;
            drop(out);
            fs::rename(&temp, &target)?;
            if !quiet {
                let size = fs::metadata(&target)?.len();
                println!("  -> {}  ({:.2} MB)", target.display(), size as f64 / 1e6);
            }
            extracted.push(target);
        }
        Ok(extracted)
    }

    fn download(&self, pack: &Pack, archive_path: &Path, quiet: bool) -> Result<()> {
        if let Some(parent) = archive_path.parent() {
            fs::create_dir_all(parent)?;
        }
        if archive_path.exists() && sha1_hex(&fs::read(archive_path)?) == pack.sha1_checksum {
            return Ok(());
        }
        let url = pack
            .download_urls
            .first()
            .context("pack has no download URL")?;
        if !quiet {
            println!("GET {url}");
        }
        let mut body = Vec::new();
        ureq::get(url)
            .call()
            .with_context(|| format!("fetching {url}"))?
            .into_body()
            .into_reader()
            .read_to_end(&mut body)?;

        let got = sha1_hex(&body);
        if got != pack.sha1_checksum {
            bail!("sha1 mismatch for {url}: {got} != {}", pack.sha1_checksum);
        }
        // Rename into place so a partial download can never be mistaken for a
        // verified one on the next run.
        let temp = temp_path(archive_path, "dl");
        let mut file = fs::File::create(&temp)?;
        file.write_all(&body)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temp, archive_path)?;
        if !quiet {
            println!("  ok sha1={got}  {:.2} MB", body.len() as f64 / 1e6);
        }
        Ok(())
    }
}

/// A temporary name no other writer will pick.
///
/// A fixed `.partial` suffix is not enough: two processes fetching overlapping
/// packs would share it, and one could truncate the other's half-written file
/// or rename a file its peer still holds open — destroying the very atomicity
/// this dance exists to provide. The Python tooling uses `mkstemp` for the same
/// reason.
fn temp_path(target: &Path, tag: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
    let name = target
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("pack");
    target.with_file_name(format!(".{tag}-{}-{unique}-{name}", std::process::id()))
}

fn sha1_hex(bytes: &[u8]) -> String {
    let digest = Sha1::digest(bytes);
    digest.iter().map(|b| format!("{b:02x}")).collect()
}
