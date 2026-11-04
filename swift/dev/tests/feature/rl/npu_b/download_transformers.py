#!/usr/bin/env python3
"""Download Transformers wheels for offline NPU tests (standard library only)."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen


VERSIONS = ('5.8.0', '5.9.0', '5.12.0', '5.16.0', '5.22.1')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download(version, output_dir):
    with urlopen(f'https://pypi.org/pypi/transformers/{version}/json', timeout=60) as response:
        metadata = json.load(response)
    wheel = next(item for item in metadata['urls'] if item['filename'].endswith('-py3-none-any.whl'))
    destination = output_dir / wheel['filename']
    expected = wheel['digests']['sha256']
    if destination.exists() and sha256(destination) == expected:
        print(f'Already verified: {destination}', flush=True)
    else:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=output_dir, suffix='.part', delete=False) as stream:
                temporary = Path(stream.name)
                with urlopen(wheel['url'], timeout=120) as response:
                    for chunk in iter(lambda: response.read(1024 * 1024), b''):
                        stream.write(chunk)
            if sha256(temporary) != expected:
                raise RuntimeError(f'SHA-256 mismatch: {wheel["filename"]}')
            os.replace(temporary, destination)
            print(f'Downloaded and verified: {destination}', flush=True)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return {
        'version': version,
        'filename': destination.name,
        'sha256': expected,
        'url': wheel['url'],
        'requires_python': metadata['info']['requires_python'],
        'requires_dist': metadata['info']['requires_dist'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('offline_packages/transformers'))
    parser.add_argument('--versions', nargs='+', default=list(VERSIONS))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = [download(version, args.output_dir) for version in dict.fromkeys(args.versions)]
    (args.output_dir / 'manifest.json').write_text(json.dumps(records, indent=2) + '\n', encoding='utf-8')
    (args.output_dir / 'SHA256SUMS').write_text(
        ''.join(f'{item["sha256"]}  {item["filename"]}\n' for item in records), encoding='utf-8')
    print('Done. Transformers wheels only; dependencies are not downloaded.')


if __name__ == '__main__':
    main()
