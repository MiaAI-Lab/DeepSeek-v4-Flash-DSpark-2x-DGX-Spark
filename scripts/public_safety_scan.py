#!/usr/bin/env python3
import re, sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else '.')
patterns = [
    ('home-path', re.compile(r'/home/[A-Za-z0-9_.-]+')),
    ('github-token', re.compile(r'(?:ghp|github_pat)_[A-Za-z0-9_\-]{20,}')),
    ('hf-token', re.compile(r'\bhf_[A-Za-z0-9]{24,}\b')),
    ('agentmail-token', re.compile(r'\bam_[A-Za-z0-9]{24,}\b')),
    ('email-address', re.compile(r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}')),
    ('lan-ip', re.compile(r'\b(?:10|192\.168|172\.(?:1[6-9]|2[0-9]|3[01]))\.\d+\.\d+\b')),
    ('private-hostname', re.compile(r'\b(?:r0b0t-dgx|r0b0tdgx1?|gn100-2eea)\b', re.IGNORECASE)),
    ('secret-assignment', re.compile(r'(GITHUB_TOKEN|HF_TOKEN|AGENTMAIL_API_KEY|OPENAI_API_KEY|XAI_API_KEY)\s*=\s*[^\s\"\']+')),
]
allow_files = {
    Path('scripts/public_safety_scan.py'),
    # This gate contains literal forbidden-pattern fixtures by design.
    Path('scripts/ci-verify.sh'),
}
allow_pattern_names = {
    Path('scripts/verify_release.py'): {'private-hostname'},
    Path('tests/test_release_contract.py'): {'private-hostname'},
}
skip_parts = {'wheels', '.git', '.hermes', '__pycache__'}
skip_suffixes = {'.gz', '.whl', '.so', '.pyc', '.diff', '.patch', '.bin', '.safetensors'}
allowed_exact = {'@mr-r0b0t'}
failures = []
for p in ROOT.rglob('*'):
    if not p.is_file() or any(part in skip_parts for part in p.parts) or p.suffix in skip_suffixes:
        continue
    rel = p.relative_to(ROOT)
    if rel in allow_files:
        continue
    try:
        raw = p.read_bytes()
        if b'\x00' in raw:
            continue
        text = raw.decode('utf-8', errors='ignore')
    except Exception:
        continue
    for name, rx in patterns:
        if name in allow_pattern_names.get(rel, set()):
            continue
        for m in rx.finditer(text):
            val = m.group(0)
            if val in allowed_exact:
                continue
            failures.append(f'{rel}: {name}: {val[:120]}')
if failures:
    print('PUBLIC SAFETY SCAN FAILED')
    for f in failures[:200]: print(f)
    if len(failures) > 200: print(f'... {len(failures)-200} more')
    sys.exit(1)
print(f'PUBLIC SAFETY SCAN PASSED: {ROOT}')
