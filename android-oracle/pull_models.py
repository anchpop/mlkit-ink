#!/usr/bin/env python3
"""Pull the SDK's private model files and compare them with public zip members.

Run after run_oracle.sh. Requires root adb shell; never writes to device storage.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import zipfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent.parent / 'oracle-models')
args = parser.parse_args()
adb = [os.environ.get('ADB', '/opt/homebrew/bin/adb'), '-s', os.environ.get('ANDROID_SERIAL', '45241FDAS000WM')]

def shell(command):
    return subprocess.check_output(adb + ['shell', command], text=True).strip()

def sha1(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha1').hexdigest()

if shell('id -u') != '0':
    sys.exit('Model extraction requires root adb shell')
root = '/data/user/0/dev.inkoracle'
# SDK 19's observed model directory; the datadownloadfile_* children are dynamic.
paths = shell(f'find {root}/files/mlkit_digital_ink_recognition -type f').splitlines()
if not paths:
    sys.exit('No model files found. Run the oracle first; inspect app files if SDK layout changes.')
args.output.mkdir(parents=True, exist_ok=True)
records = []
for remote in sorted(paths):
    relative = Path(remote).relative_to(root)
    local = args.output / 'device' / relative
    local.parent.mkdir(parents=True, exist_ok=True)
    size = int(shell('stat -c %s ' + shlex.quote(remote)))
    digest = shell('sha1sum ' + shlex.quote(remote)).split()[0]
    subprocess.run(adb + ['pull', remote, str(local)], check=True, stdout=sys.stderr)
    if local.stat().st_size != size or sha1(local) != digest:
        sys.exit(f'Pulled file verification failed: {remote}')
    records.append({'devicePath': remote, 'dataDataAlias': remote.replace('/data/user/0/', '/data/data/', 1),
                    'localPath': str(local.resolve()), 'size': size, 'sha1': digest})

comparisons = []
for archive in sorted((args.output / 'reference').glob('*.zip')):
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if name.endswith('/'):
                continue
            data = z.read(name)
            digest = hashlib.sha1(data).hexdigest()
            matches = [r['devicePath'] for r in records if r['size'] == len(data) and r['sha1'] == digest
                       and Path(r['localPath']).read_bytes() == data]
            comparisons.append({'archive': str(archive.resolve()), 'archiveSha1': sha1(archive),
                                'member': name, 'size': len(data), 'sha1': digest,
                                'byteIdenticalDevicePaths': matches})
report = {'deviceSerial': adb[2], 'buildFingerprint': shell('getprop ro.build.fingerprint'),
          'files': records, 'referenceComparisons': comparisons}
result = json.dumps(report, indent=2) + '\n'
(args.output / 'manifest.json').write_text(result)
print(result, end='')
if not comparisons or any(not c['byteIdenticalDevicePaths'] for c in comparisons):
    sys.exit('One or more reference packs were absent or did not match; see manifest.json')
