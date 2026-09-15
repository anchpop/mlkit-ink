#!/usr/bin/env python3
"""End-to-end regression tests on the connected device (not a mock SDK)."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'test-results'
RESULTS.mkdir(exist_ok=True)


def run(name, source, expected_status='ok'):
    output = RESULTS / f'{name}.json'
    command = [str(ROOT / 'run_oracle.sh'), str(source), str(output)]
    print('+', ' '.join(command), flush=True)
    with (RESULTS / f'{name}.run.log').open('w') as log:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=log, text=True)
    (RESULTS / f'{name}.stdout.log').write_text(result.stdout)
    assert result.returncode == (0 if expected_status == 'ok' else 1), (name, result.returncode)
    data = json.loads(result.stdout)
    assert data == json.loads(output.read_text())
    assert data['inputSha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert data['status'] == expected_status
    if expected_status == 'ok':
        assert data['candidates'] and data['candidates'][0]['text'].strip()
        assert all('text' in c and 'score' in c for c in data['candidates'])
    else:
        assert data['error']['stage'] == 'validate_input'
    print(name, json.dumps(data, ensure_ascii=False), flush=True)
    return data


hi = run('hi', ROOT / 'test-inputs' / 'hi.json')
cat = run('cat', ROOT / 'test-inputs' / 'cat.json')
assert hi['candidates'][0]['text'] == 'hi'
assert cat['candidates'][0]['text'] == 'cat'
assert run('hi-repeat', ROOT / 'test-inputs' / 'hi.json') == hi

with tempfile.TemporaryDirectory(prefix='ink-oracle-tests-') as directory:
    source = Path(directory) / 'invalid.json'
    for name, stroke in [
        ('invalid-lengths', {'x': [1, 2], 'y': [3]}),
        ('invalid-timestamp', {'x': [1], 'y': [3], 't': [1.5]}),
    ]:
        source.write_text(json.dumps({'language': 'en-US', 'strokes': [stroke]}) + '\n')
        run(name, source, 'error')
print('PASS: timed, untimed, repeatability, array validation, timestamp validation')
