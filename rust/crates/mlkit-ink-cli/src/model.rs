//! Loading packs off disk for the host.
//!
//! The core [`Recognizer`] borrows its model bytes, which keeps the 22 MB
//! language model out of its own allocation but means something has to own
//! those bytes. That is this type: [`Loaded`] holds them, and
//! [`Loaded::recognizer`] hands out a recognizer borrowing from it.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use memmap2::Mmap;
use mlkit_ink::Recognizer;
use mlkit_ink::packs::PackMapping;

use crate::catalog::Catalog;

pub struct Loaded {
    pub recospec: Vec<u8>,
    pub tflite: Vec<u8>,
    /// Mapped rather than read: it is 22 MB for English and the decoder touches
    /// only a small, scattered fraction of it per recognition.
    pub fst: Option<Mmap>,
    pub paths: Vec<PathBuf>,
}

impl Loaded {
    pub fn recognizer(&self) -> Result<Recognizer<'_>> {
        Recognizer::load(&self.recospec, &self.tflite, self.fst.as_deref())
            .map_err(|e| anyhow!("{e}"))
    }
}

/// Resolve a BCP-47 tag, fetching any packs that are not cached yet.
pub fn load(root: &Path, tag: &str, use_lm: bool, quiet: bool) -> Result<Loaded> {
    let mapping_bytes = fs::read(root.join("packmapping.pb"))
        .with_context(|| format!("reading {}", root.join("packmapping.pb").display()))?;
    let mapping = PackMapping::parse(&mapping_bytes).map_err(|e| anyhow!("{e}"))?;
    let names = mapping.resolve(tag).map_err(|e| anyhow!("{e}"))?;
    let catalog = Catalog::load(root)?;

    let mut paths = Vec::new();
    let recospec = fs::read(pick(
        &catalog.ensure(&names.recospec, quiet)?,
        ".recospec.local",
        &mut paths,
    )?)?;
    let tflite = fs::read(pick(
        &catalog.ensure(&names.tflite, quiet)?,
        ".tflite",
        &mut paths,
    )?)?;
    let fst = match names.fst.as_deref().filter(|_| use_lm) {
        Some(name) => {
            let path = pick(
                &catalog.ensure(name, quiet)?,
                ".compact.fst.local",
                &mut paths,
            )?;
            let file = fs::File::open(&path)?;
            // SAFETY-adjacent note: `Mmap::map` is unsafe because another process
            // could truncate the file underneath us. The catalog's extraction is
            // atomic precisely so that cannot happen here.
            Some(unsafe { Mmap::map(&file)? })
        }
        None => None,
    };

    Ok(Loaded {
        recospec,
        tflite,
        fst,
        paths,
    })
}

/// Pick the one member of a pack with the expected suffix.
///
/// Some FST pack generations use a different filename, so a suffix miss falls
/// back to "the only file in the pack" rather than failing.
fn pick(members: &[PathBuf], suffix: &str, seen: &mut Vec<PathBuf>) -> Result<PathBuf> {
    let matching: Vec<_> = members.iter().filter(|p| ends_with(p, suffix)).collect();
    let candidates = if matching.is_empty() {
        members.iter().collect()
    } else {
        matching
    };
    match candidates.as_slice() {
        [one] => {
            seen.push((*one).clone());
            Ok((*one).clone())
        }
        other => Err(anyhow!(
            "expected one {suffix} artifact in the pack, found {}",
            other.len()
        )),
    }
}

fn ends_with(path: &Path, suffix: &str) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.ends_with(suffix))
}
