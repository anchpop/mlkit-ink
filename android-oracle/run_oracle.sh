#!/usr/bin/env bash
# stdout is only the oracle JSON; build/adb progress and diagnostics go to stderr.
# Prerequisites: brew install openjdk gradle; Python 3.11+; Android SDK platform 35.
# Gradle installs build-tools 36.0.0 if absent and its SDK license is accepted.
# Overrides: ADB, ANDROID_SERIAL, JAVA_HOME, ANDROID_HOME, ORACLE_TIMEOUT (seconds).
# The first/changed APK install can require a device Play Protect response; this
# script never disables verification. Identical installed APKs are not reinstalled.
# SDK scores are preserved as-is, including null for models without score support.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 INPUT.json [OUTPUT.json]" >&2
    exit 2
fi
INPUT="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1")"
OUTPUT="${2:-$ROOT/ink_output.json}"
[[ -f "$INPUT" ]] || { echo "Input not found: $INPUT" >&2; exit 2; }
python3 - "$INPUT" "$OUTPUT" <<'PY_CHECK'
import os, sys
source, destination = sys.argv[1:]
if (os.path.realpath(source) == os.path.realpath(destination)
        or (os.path.exists(destination) and os.path.samefile(source, destination))):
    sys.exit('Input and output must be different files')
PY_CHECK
# Fixed device filenames mean runs must be serialized.
mkdir "$ROOT/.oracle-lock" 2>/dev/null || { echo "Another oracle run is active (or remove stale $ROOT/.oracle-lock)." >&2; exit 2; }
trap 'rmdir "$ROOT/.oracle-lock"' EXIT
ADB="${ADB:-/opt/homebrew/bin/adb}"
SERIAL="${ANDROID_SERIAL:-45241FDAS000WM}"
TIMEOUT="${ORACLE_TIMEOUT:-300}"
export JAVA_HOME="${JAVA_HOME:-/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home}"
export ANDROID_HOME="${ANDROID_HOME:-$HOME/Library/Android/sdk}"
export PATH="$JAVA_HOME/bin:$PATH"
[[ -x "$JAVA_HOME/bin/java" ]] || { echo 'Install JDK: brew install openjdk gradle' >&2; exit 2; }
"$ROOT/gradlew" -p "$ROOT" --console=plain :app:assembleDebug >&2
"$ADB" -s "$SERIAL" get-state >&2
# Avoid reinstalling an identical APK for every input; this also avoids involving
# Android's package verifier when all we changed was an ink JSON file.
python3 - "$ADB" "$SERIAL" "$ROOT/app/build/outputs/apk/debug/app-debug.apk" <<'PY_INSTALL'
import hashlib, pathlib, shlex, subprocess, sys
adb, serial, apk = sys.argv[1:]
base = [adb, '-s', serial]
installed = subprocess.run(base + ['shell', 'pm', 'path', 'dev.inkoracle'], capture_output=True, text=True).stdout
paths = [line.removeprefix('package:') for line in installed.splitlines() if line.startswith('package:')]
local_hash = hashlib.sha256(pathlib.Path(apk).read_bytes()).hexdigest()
remote_hash = ''
if len(paths) == 1:
    result = subprocess.run(base + ['shell', 'sha256sum ' + shlex.quote(paths[0])],
                            capture_output=True, text=True)
    if result.returncode == 0:
        remote_hash = result.stdout.split()[0]
if local_hash == remote_hash:
    print('Installed APK matches build; skipping reinstall.', file=sys.stderr)
else:
    # Wake the display for the platform installer; never dismiss the keyguard or
    # disable package verification. The Activity keeps itself awake afterward.
    subprocess.run(base + ['shell', 'input', 'keyevent', 'KEYCODE_WAKEUP'], check=True)
    try:
        subprocess.run(base + ['install', '-r', apk], check=True, stdout=sys.stderr, timeout=180)
    except subprocess.TimeoutExpired:
        sys.exit('APK installation timed out; unlock the device and respond to any Play Protect prompt, then retry. The oracle does not bypass package verification.')
PY_INSTALL
"$ADB" -s "$SERIAL" shell am force-stop dev.inkoracle
"$ADB" -s "$SERIAL" shell appops set dev.inkoracle MANAGE_EXTERNAL_STORAGE allow
"$ADB" -s "$SERIAL" shell rm -f /sdcard/ink_output.json /sdcard/ink_output.json.tmp
"$ADB" -s "$SERIAL" push "$INPUT" /sdcard/ink_input.json >&2
"$ADB" -s "$SERIAL" shell am start -W -n dev.inkoracle/.MainActivity >&2
python3 - "$ADB" "$SERIAL" "$TIMEOUT" <<'PY'
import subprocess, sys, time
adb, serial, timeout = sys.argv[1:]
end = time.monotonic() + float(timeout)
while time.monotonic() < end:
    result = subprocess.run([adb, '-s', serial, 'shell', 'test', '-f', '/sdcard/ink_output.json'], timeout=15)
    if result.returncode == 0:
        break
    time.sleep(0.5)
else:
    subprocess.run([adb, '-s', serial, 'logcat', '-d', '-s', 'InkOracle:*', 'DIRecoDownload:*', 'MddModelManager:*', 'zzaab:*', 'AndroidRuntime:E'], stdout=sys.stderr)
    sys.exit('Oracle timed out before publishing output')
PY
mkdir -p "$(dirname "$OUTPUT")"
"$ADB" -s "$SERIAL" pull /sdcard/ink_output.json "$OUTPUT" >&2
cat "$OUTPUT"
printf '\n'
python3 - "$INPUT" "$OUTPUT" <<'PY'
import hashlib, json, sys
with open(sys.argv[2]) as f:
    result = json.load(f)
with open(sys.argv[1], 'rb') as f:
    expected = hashlib.sha256(f.read()).hexdigest()
if result.get('inputSha256') != expected:
    sys.exit('Output inputSha256 does not match supplied input')
if result.get('status') != 'ok':
    sys.exit('Oracle reported an error; see result JSON')
PY
