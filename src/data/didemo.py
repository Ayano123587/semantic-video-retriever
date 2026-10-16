"""Pinned official DiDeMo metadata; natural videos and human temporal labels."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

REVISION = 'b6a555c8134581305d0ed4716fbc192860e0b88c'
FILES = ['train_data.json', 'val_data.json', 'test_data.json',
         'video_licenses.txt', 'yfcc100m_hash.txt']
ROOT = f'https://raw.githubusercontent.com/LisaAnne/LocalizingMoments/{REVISION}/data/'


def fetch_metadata(directory='data/external/didemo'):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    def fetch(name):
        path = directory/name
        if not path.exists():
            request = Request(ROOT+name, headers={'User-Agent': 'semantic-video-research/0.1'})
            with urlopen(request, timeout=60) as response:
                content = response.read(15_000_001)
            if len(content) > 15_000_000:
                raise ValueError('Unexpected metadata size')
            path.write_bytes(content)
        return name, hashlib.sha256(path.read_bytes()).hexdigest()
    with ThreadPoolExecutor(max_workers=3) as pool:
        hashes = dict(pool.map(fetch, FILES))
    return {'repository': 'https://github.com/LisaAnne/LocalizingMoments',
            'revision': REVISION, 'sha256': hashes}


if __name__ == '__main__':
    print(json.dumps(fetch_metadata(), indent=2))
