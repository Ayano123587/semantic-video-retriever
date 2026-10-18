"""Pinned official DiDeMo metadata; natural videos and human temporal labels."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import os
import shutil
from tempfile import TemporaryDirectory
import time
from urllib.request import Request, urlopen

REVISION = 'b6a555c8134581305d0ed4716fbc192860e0b88c'
FILES = ['train_data.json', 'val_data.json', 'test_data.json',
         'video_licenses.txt', 'yfcc100m_hash.txt']
ROOT = f'https://raw.githubusercontent.com/LisaAnne/LocalizingMoments/{REVISION}/data/'
HASHES = {
    'train_data.json': '11130a5a05e00f141f16b0ea38e67fa58e04f7168526e0d87e9dd306b2ddb78a',
    'val_data.json': 'b0364cc256553332feb19d46bcc4cd2b09774949fe6c0b25e7ed0ff3c6aefebb',
    'test_data.json': '1891c04ec48b3d364c739594b2b6413806b74bd9027c092d896e7ebb930ff1cd',
    'video_licenses.txt': 'a6b2efc4b071e6c5474f98bd1f23ebcfcd880530ba35a12dcac8070f96a6e2d4',
    'yfcc100m_hash.txt': '481d9aaf020624d5915200bcf4752fb46d3e1931167e8b46715a5f342577cc4d'}


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
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != HASHES[name]:
            raise ValueError(f'Pinned DiDeMo metadata checksum mismatch: {name}')
        return name, digest
    with ThreadPoolExecutor(max_workers=3) as pool:
        hashes = dict(pool.map(fetch, FILES))
    return {'repository': 'https://github.com/LisaAnne/LocalizingMoments',
            'revision': REVISION, 'sha256': hashes}


def selection(directory='data/external/didemo', output='data/interim/didemo-full-selection.json', counts=None, *, noncommercial_research=False):
    """Read video IDs for staging only; no caption text or target score is used."""
    provenance = fetch_metadata(directory)
    directory = Path(directory)
    licenses = {}
    for line in (directory/'video_licenses.txt').read_text(encoding='utf-8').splitlines()[1:]:
        fields = line.split()
        if len(fields) == 3:
            licenses[fields[0]] = {'original_url': fields[1], 'license': fields[2]}
    hashes = dict(line.split() for line in (directory/'yfcc100m_hash.txt').read_text().splitlines())
    allowed = {'http://creativecommons.org/licenses/by/2.0/',
               'http://creativecommons.org/licenses/by-sa/2.0/'}
    if noncommercial_research:
        allowed.update({'http://creativecommons.org/licenses/by-nc/2.0/',
                        'http://creativecommons.org/licenses/by-nc-sa/2.0/'})
    selected, official_sets = {}, {}
    if counts is not None and (len(counts)!=3 or any(type(n) is not int or n<1 for n in counts)):
        raise ValueError('Positive integer source-selection counts required')
    coverage = {}
    for split, count in zip(('train','val','test'),counts or (None,None,None)):
        rows = json.loads((directory/f'{split}_data.json').read_text(encoding='utf-8'))
        official_sets[split] = {row['video'] for row in rows}
        candidates = [name for name in official_sets[split]
                      if licenses.get(name.split('_')[1],{}).get('license') in allowed]
        candidates.sort(key=lambda name: hashlib.sha256(name.encode('utf-8')).hexdigest())
        coverage[split] = {'official_videos':len(official_sets[split]),
                           'official_queries':len(rows),'license_eligible_videos':len(candidates),
                           'selected_videos':len(candidates if count is None else candidates[:count])}
        selected[split] = []
        for name in candidates if count is None else candidates[:count]:
            identifier = name.split('_')[1]
            checksum = hashes[identifier]
            selected[split].append({'video':name, 'id':identifier, **licenses[identifier],
                'aws_url': f'https://multimedia-commons.s3-us-west-2.amazonaws.com/data/videos/mp4/{checksum[:3]}/{checksum[3:6]}/{checksum}.mp4'})
    if any(official_sets[a] & official_sets[b] for a,b in [('train','val'),('train','test'),('val','test')]):
        raise ValueError('Official video split overlap')
    result = {'source': provenance, 'selection':'SHA256 name order; CC BY2.0/BY-SA2.0 only; no score/caption filtering',
              'splits': selected, 'test_gold_used_for_selection': False}
    if noncommercial_research:
        result['selection'] = 'Every adaptation-permitting CC2.0 ID for noncommercial research; no score/caption filtering'
        result['use_terms'] = {'noncommercial_research_only':True,
            'preserve_per_source_attribution':True,'redistribute_media_or_weights':False,
            'sharealike_terms_retained':True,'no_derivatives_license_categories_excluded':True}
    if counts is None:
        result['coverage'] = coverage
        result['sampling_cap'] = None
        result['scope'] = 'Every license-eligible official video; not every official license category'
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if json.loads(destination.read_text(encoding='utf-8')) != result:
            raise ValueError('Refuse to revise frozen video selection')
    else:
        destination.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


def media_filename(video):
    """Keep official IDs separate from filenames that Windows can represent."""
    if not isinstance(video,str) or not re.fullmatch(r'[A-Za-z0-9@_-]+\.[A-Za-z0-9]{0,10}',video):
        raise ValueError('Invalid source video filename')
    if video.endswith('.'):
        return 'source-'+hashlib.sha256(video.encode('utf-8')).hexdigest()+'.mp4'
    return video


@contextmanager
def decoder_source(path):
    """AWS bytes are MP4 even when the official upload suffix differs."""
    path=Path(path)
    if path.suffix.lower()=='.mp4':
        yield path
        return
    with TemporaryDirectory(prefix='decoder-source-',dir=path.parent) as temporary:
        compatible=Path(temporary)/'source.mp4'
        try:os.link(path,compatible)
        except OSError:shutil.copyfile(path,compatible)
        yield compatible


def download_one(record, directory='data/raw/didemo'):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # The official names preserve original upload extensions, while AWS serves
    # transcoded MP4 bytes. Validate a safe basename, not one particular suffix.
    path = directory/media_filename(record['video'])
    if not path.exists():
        temporary = path.with_suffix('.part')
        request = Request(record['aws_url'], headers={'User-Agent':'semantic-video-research/0.1'})
        try:
            with urlopen(request,timeout=60) as response, temporary.open('wb') as handle:
                size = 0
                while block := response.read(1024*1024):
                    size += len(block)
                    if size > 128*1024*1024:
                        raise ValueError('Source exceeds per-video128MiB ingestion limit')
                    handle.write(block)
            temporary.replace(path)
        except Exception:
            # Preserve partial download as ignored evidence, never decode it.
            raise
    return {'video': record['video'], 'bytes':path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'license':record['license'], 'original_url':record['original_url']}


def download_all(plan, output='data/interim/didemo-downloads', directory='data/raw/didemo'):
    """Fixed cohort; failed availability never triggers score-guided replacement."""
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    prior = {}
    journal = output/'receipts.jsonl'
    if journal.exists():
        for line in journal.read_text(encoding='utf-8').splitlines():
            row=json.loads(line)
            if row.get('status')=='complete':prior[row['video']]=row['sha256']
    tasks = [(split,record) for split,records in plan['splits'].items() for record in records]
    start=time.monotonic()
    def fetch(task):
        split,record=task
        try:
            receipt=download_one(record,directory)
            if record['video'] in prior and receipt['sha256']!=prior[record['video']]:
                raise ValueError('Previously downloaded video changed')
            return {**receipt,'split':split,'status':'complete'}
        except Exception as error:
            return {'video':record['video'],'split':split,'status':'failed',
                    'error':type(error).__name__,'reason':str(error)}
    results=[]
    with ThreadPoolExecutor(max_workers=4) as pool, journal.open('a',encoding='utf-8') as handle:
        for number,result in enumerate(pool.map(fetch,tasks),1):
            results.append(result)
            handle.write(json.dumps(result,allow_nan=False)+'\n');handle.flush()
            if number%50==0:print(json.dumps({'processed':number,'total':len(tasks)}),flush=True)
    counts={split:{status:sum(r['split']==split and r['status']==status for r in results)
                   for status in ('complete','failed')} for split in plan['splits']}
    seen={}
    duplicates=[]
    for row in results:
        if row['status']!='complete':continue
        if row['sha256'] in seen and seen[row['sha256']]['split']!=row['split']:
            duplicates.append({'sha256':row['sha256'],'splits':[seen[row['sha256']]['split'],row['split']]})
        seen[row['sha256']]=row
    report={'counts':counts,'seconds':time.monotonic()-start,'cross_split_exact_media_duplicates':duplicates,
            'receipt_sha256':hashlib.sha256(journal.read_bytes()).hexdigest(),
            'selection_sha256':hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest(),
            'media_bytes':sum(r.get('bytes',0) for r in results),'synthetic_media':False,
            'selection_uses_caption_text':False,'replacement_policy':'none'}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    if duplicates:raise ValueError('Exact media content crosses official video splits; resolve before training')
    return report


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe',action='store_true')
    parser.add_argument('--download',action='store_true')
    parser.add_argument('--expanded',action='store_true',help='Prospectively expanded availability cohort; no label/score filtering')
    parser.add_argument('--sample',action='store_true',help='Explicit historical small diagnostic cohort; not a full benchmark')
    parser.add_argument('--noncommercial-research',action='store_true',help='Include adaptation-permitting BY-NC/BY-NC-SA sources for noncommercial research; retain all source terms')
    args=parser.parse_args()
    if args.expanded and args.sample:parser.error('Choose one historical sampling mode')
    if args.noncommercial_research and (args.expanded or args.sample):parser.error('Research coverage has no historical sampling mode')
    result = (selection(output='data/interim/didemo-research-selection.json',noncommercial_research=True) if args.noncommercial_research else
              selection(output='data/interim/didemo-expanded-selection.json',counts=(3000,500,500)) if args.expanded else
              selection(output='data/interim/didemo-selection.json',counts=(500,50,100)) if args.sample else selection())
    if args.download:
        download_all(result,output='data/interim/didemo-research-downloads' if args.noncommercial_research else
                     'data/interim/didemo-expanded-downloads' if args.expanded else
                     'data/interim/didemo-downloads' if args.sample else 'data/interim/didemo-full-downloads')
    elif args.probe:
        print(json.dumps(download_one(result['splits']['train'][0]),indent=2))
    else:
        print(json.dumps({'metadata': result['source'], 'selected_videos':
                          {key:len(value) for key,value in result['splits'].items()}}, indent=2))
