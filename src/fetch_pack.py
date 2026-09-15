#!/usr/bin/env python3
"""Resolve an ML Kit Digital Ink pack name to its dl.google.com URL, download, verify, unzip.

The catalog ships inside the digital-ink-recognition AAR at assets/manifest.json;
every pack is a plain unauthenticated https://dl.google.com/handwriting/models/ URL.

Extraction is **atomic and idempotent**. That matters because `src/fst.py` mmaps the 22 MB
language model: a plain `extractall` rewrites the file in place, so re-resolving a language
while another process holds an mmap truncates that mapping out from under it and the reader
dies with SIGBUS. Here, already-correct members are left untouched, and anything we do write
goes to a temp file that is then atomically renamed -- an existing mmap keeps pointing at the
old inode and is never invalidated.
"""
import hashlib, json, os, pathlib, sys, tempfile, urllib.request, zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKS = {p["name"]: p for p in json.loads((ROOT / "manifest.json").read_text())["packs"]}


def _download(url, zpath, sha1):
    """Download to a temp file and rename, so a partial download is never mistaken for a good one."""
    zpath.parent.mkdir(parents=True, exist_ok=True)
    if zpath.exists() and hashlib.sha1(zpath.read_bytes()).hexdigest() == sha1:
        return
    print(f"GET {url}")
    fd, tmp = tempfile.mkstemp(dir=zpath.parent, prefix=".dl-")
    os.close(fd)
    tmp = pathlib.Path(tmp)
    try:
        urllib.request.urlretrieve(url, tmp)
        got = hashlib.sha1(tmp.read_bytes()).hexdigest()
        if got != sha1:
            raise ValueError(f"sha1 mismatch for {url}: {got} != {sha1}")
        os.replace(tmp, zpath)
        print(f"  ok sha1={got}  {zpath.stat().st_size / 1e6:.2f} MB")
    finally:
        tmp.unlink(missing_ok=True)


def fetch(name, dest=ROOT / "models"):
    """Download (if needed) and extract a pack. Returns the extracted member paths."""
    pack = PACKS[name]
    dest = pathlib.Path(dest)
    zpath = dest / (name + ".zip")
    _download(pack["download_urls"][0], zpath, pack["sha1_checksum"])

    with zipfile.ZipFile(zpath) as z:
        members = [i for i in z.infolist() if not i.is_dir()]
        out = []
        for info in members:
            target = dest / info.filename
            # Already extracted and the right size: leave it alone. Rewriting it would break
            # any live mmap for no benefit.
            if target.exists() and target.stat().st_size == info.file_size:
                out.append(target)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".x-")
            tmp = pathlib.Path(tmp)
            try:
                with z.open(info) as src, os.fdopen(fd, "wb") as dst:
                    while chunk := src.read(1 << 20):
                        dst.write(chunk)
                os.replace(tmp, target)  # atomic: readers keep the old inode
                print(f"  -> {info.filename}  ({target.stat().st_size / 1e6:.2f} MB)")
            finally:
                tmp.unlink(missing_ok=True)
            out.append(target)
    return out


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        if arg == "--list":
            for n in sorted(PACKS):
                print(n)
        else:
            fetch(arg)
