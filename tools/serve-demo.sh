#!/usr/bin/env bash
# Build the wasm bundle and serve the browser demo on loopback.
#
# The serving is done by tools/serve_demo.py rather than `python3 -m
# http.server`, which binds every interface and would happily hand out
# .git/git-crypt/keys/default to anyone on your network.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v wasm-pack >/dev/null; then
  echo "wasm-pack is required: cargo install wasm-pack" >&2
  exit 1
fi

echo "building wasm…"
(cd "$root/rust/crates/mlkit-ink-wasm" && wasm-pack build --release --target web --out-dir "$root/web/pkg")

# Always fetch: it is idempotent and sha1-verified, so a cached pack costs a
# hash. Testing for the net instead would be wrong -- the Latin net is shared by
# some sixty languages, so fetching French leaves it present without the en-US
# recospec the page also needs, and the demo would 404 on a file the check said
# was there.
echo "checking the en-US packs…"
(cd "$root/rust" && cargo run --release -q -p mlkit-ink-cli -- fetch en-US >/dev/null)

exec python3 "$root/tools/serve_demo.py" --port "${1:-8347}"
