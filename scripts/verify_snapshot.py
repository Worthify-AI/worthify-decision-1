"""Verify curated inventory, source attribution and relative Markdown links."""
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def verify(root=ROOT):
    manifest = json.loads((root/'SOURCE_PROVENANCE.json').read_text())
    if not re.fullmatch('[0-9a-f]{40}', manifest['source_revision']):
        raise ValueError('Missing immutable source revision')
    inventory = manifest['snapshot_files_sha256']
    ignored = {'.git', '.venv', '__pycache__', '.pytest_cache', '.cache', 'build', 'dist'}
    actual = {str(path.relative_to(root)) for path in root.rglob('*')
              if path.is_file() and not any(part in ignored or part.endswith('.egg-info')
                  for part in path.relative_to(root).parts)}
    expected = set(inventory) | {'SOURCE_PROVENANCE.json'}
    if actual != expected:
        raise ValueError(f'Snapshot inventory differs: extra={sorted(actual-expected)}, missing={sorted(expected-actual)}')
    for name, digest in inventory.items():
        path = root/name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Snapshot hash mismatch: {name}')
    for name, record in manifest['files'].items():
        if name not in inventory or not re.fullmatch('[0-9a-f]{40}',record['source_git_blob']):
            raise ValueError(f'Invalid source lineage: {name}')
        if record['relationship'] == 'unchanged' and record['source_sha256'] != inventory[name]:
            raise ValueError(f'Unchanged source differs: {name}')
    broken=[]
    for name in inventory:
        if not name.endswith('.md'): continue
        path=root/name
        for target in re.findall(r'!?\[[^\]]*\]\(([^\s)]+)(?:\s+"[^"\n]*")?\)',path.read_text()):
            parts=urlsplit(target.strip('<>'))
            if parts.scheme or parts.netloc or not parts.path: continue
            linked=(path.parent/unquote(parts.path)).resolve()
            if not linked.is_relative_to(root.resolve()) or not linked.exists(): broken.append((name,target))
    if broken: raise ValueError(f'Broken local links: {broken}')
    print(json.dumps({'status':'ok','snapshot_files':len(inventory),'source_revision':manifest['source_revision']}))


if __name__=='__main__':
    verify()
