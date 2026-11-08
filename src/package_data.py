"""Empaqueta los snapshots originales verificados, sin leer sus filas."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_STORED
from src.preparation import REPO_ROOT


def main():
    manifest=REPO_ROOT/'docs/data_manifest.json'
    entries=json.loads(manifest.read_text())
    paths=[]
    for item in entries:
        path=REPO_ROOT/item['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError(f'Hash incorrecto: {path.name}')
        paths.append(path)
    destination=REPO_ROOT/'output/airbnb_datos_entrega.zip'
    destination.parent.mkdir(exist_ok=True)
    with ZipFile(destination,'w',compression=ZIP_STORED) as archive:
        for path in paths+[manifest,REPO_ROOT/'docs/split_manifest.json']:
            archive.write(path,path.relative_to(REPO_ROOT))
    print(destination)


if __name__=='__main__': main()
