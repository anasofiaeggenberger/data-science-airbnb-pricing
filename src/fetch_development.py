"""Descarga los bytes registrados y verifica SHA-256 sin cambiar manifiestos."""
import argparse
import hashlib
import json
from urllib.request import urlopen

from src.preparation import REPO_ROOT


def fetch_snapshots(include_test=False):
    manifest = json.loads((REPO_ROOT / 'docs/data_manifest.json').read_text())
    for entry in manifest:
        if entry['snapshot_date'] != '2026-06-15' and not include_test:
            continue
        path = REPO_ROOT / entry['path']
        content = path.read_bytes() if path.exists() else urlopen(entry['url'], timeout=120).read()
        if hashlib.sha256(content).hexdigest() != entry['sha256']:
            raise ValueError(f'El archivo no coincide con el manifiesto: {path.name}')
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(content)
        print(f'Hash verificado: {path.name} ({len(content)} bytes)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-test', action='store_true', help='Solo descarga/verifica bytes; no analiza el test.')
    fetch_snapshots(parser.parse_args().include_test)
