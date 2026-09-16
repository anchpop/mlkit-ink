#!/usr/bin/env python3
"""Serve the browser demo, and nothing else.

The obvious way to do this -- `python3 -m http.server` from the repo root --
is dangerous here, and was: it binds every interface and serves every file,
which on this repo includes `.git/git-crypt/keys/default`. Anyone on the same
network could fetch the symmetric key that the whole encryption scheme rests
on. So this binds loopback only and refuses to serve anything outside an
explicit allowlist: the demo page, its wasm bundle, and the two model files the
page fetches.
"""
from __future__ import annotations

import argparse
import http.server
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Exactly what the page loads. Anything else is a 404, including `.git`.
ALLOWED_DIRS = ("web",)
ALLOWED_FILES = (
    "models/qrnn_en_us_reco_20200318_fst_20191208_recospec_zip/"
    "qrnn.en_us.reco_20200318.fst_20191208.recospec.local",
    "models/indy_lstm_latin_6x216_tflite_20191208_zip/"
    "latin_indy_lstm_6x216_20191208.tflite",
    "models/en_us_20191208_compact_fst_zip/en_us.compact.fst.local",
)


def permitted(relative: str) -> bool:
    """Allowlist, not denylist: a new secret in the repo must not become
    reachable just because nobody remembered to exclude it."""
    if relative in ALLOWED_FILES:
        return True
    head = relative.split("/", 1)[0]
    return head in ALLOWED_DIRS


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def send_head(self):
        """Refuse anything off the allowlist with an honest 404.

        Done here rather than in `translate_path`, where redirecting to
        /dev/null happens to serve empty bodies but still answers 200 — which
        looks like "allowed but empty" and depends on a readable /dev/null for
        its safety. A rejection should be a rejection.
        """
        resolved = Path(super().translate_path(self.path)).resolve()
        try:
            relative = resolved.relative_to(ROOT).as_posix()
        except ValueError:
            relative = None  # escaped the tree entirely
        if relative is None or not permitted(relative):
            self.send_error(404, "Not Found")
            return None
        return super().send_head()

    def log_message(self, fmt, *args):
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8347)
    # Opt-in, loudly, because the models are Google's and the page is not ours
    # to publish. See the Legal section of README.md.
    ap.add_argument("--host", default="127.0.0.1",
                    help="default loopback; pass 0.0.0.0 to expose on your network")
    args = ap.parse_args()

    server = http.server.ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"\n  http://{args.host}:{args.port}/web/\n")
    if args.host != "127.0.0.1":
        print("  WARNING: reachable from your network, and it serves Google's model files.\n")
    server.serve_forever()


if __name__ == "__main__":
    main()
