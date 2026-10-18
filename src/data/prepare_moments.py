"""Freeze licensed, downloaded DiDeMo train/validation sources without test gold."""
import argparse,hashlib,json,math
from pathlib import Path
from src.data.didemo import HASHES,REVISION
from src.models.moments import interval,moments


def stage(output, directory='data/external/didemo', receipts='data/interim/didemo-expanded-downloads/receipts.jsonl', selection='data/interim/didemo-expanded-selection.json', decoder_preflight=None):
    output=Path(output)
    if output.exists():raise ValueError('Refuse to overwrite frozen preparation')
    directory=Path(directory);receipt_path=Path(receipts);selection_path=Path(selection)
    plan=json.loads(selection_path.read_text(encoding='utf-8'))
    permitted={(split,r['video']):r for split,rows in plan['splits'].items() for r in rows}
    available={}
    for line in receipt_path.read_text(encoding='utf-8').splitlines():
        row=json.loads(line)
        if row['status']!='complete':continue
        key=(row['split'],row['video'])
        if key not in permitted:raise ValueError('Receipt outside frozen selection')
        if key in available and available[key]['sha256']!=row['sha256']:raise ValueError('Conflicting media receipts')
        available[key]=row
    excluded=set();decoder_hash=None
    if decoder_preflight is not None:
        decoder_path=Path(decoder_preflight);decoder_hash=hashlib.sha256(decoder_path.read_bytes()).hexdigest()
        decoded=json.loads(decoder_path.read_text(encoding='utf-8'));seen=set()
        if 'receipt_sha256' in decoded and decoded['receipt_sha256']!=hashlib.sha256(receipt_path.read_bytes()).hexdigest():
            raise ValueError('Decoder receipt cohort mismatch')
        official_geometry={}
        for split in ('train','val','test'):
            path=directory/f'{split}_data.json';content=path.read_bytes()
            if hashlib.sha256(content).hexdigest()!=HASHES[path.name]:raise ValueError('Changed official metadata')
            for row in json.loads(content):
                key=(split,row['video']);n=row['num_segments']
                if key in official_geometry and official_geometry[key]!=n:raise ValueError('Inconsistent video geometry')
                official_geometry[key]=n
        hash_aliases={}
        for key,receipt in available.items():
            hash_aliases.setdefault(receipt['sha256'],[]).append({
                'split':key[0],'video':key[1],'num_segments':official_geometry[key]})
        for aliases in hash_aliases.values():aliases.sort(key=lambda r:(r['split'],r['video']))
        for row in decoded['rows']:
            key=(row['split'],row['video'])
            if key not in available:raise ValueError('Decoder source outside frozen availability cohort')
            expected=available[key]['sha256']
            if row.get('sha256',expected)!=expected:raise ValueError('Decoder source hash mismatch')
            aliases=hash_aliases[expected]
            conflict=len({alias['num_segments'] for alias in aliases})>1
            if conflict or row.get('reason')=='annotation_geometry_conflict':
                if not conflict or row.get('reason')!='annotation_geometry_conflict' or row.get('status')!='invalid' or row.get('aliases')!=aliases or row.get('last_frame_decoded') is not False or row.get('geometry_inspected') is not False:
                    raise ValueError('Decoder annotation conflict provenance mismatch')
            if expected in seen:raise ValueError('Duplicate decoder inspection')
            seen.add(expected)
            duration=row.get('duration',0);fps=row.get('fps',0);segments=row.get('num_segments',0)
            if segments!=official_geometry[key]:raise ValueError('Decoder annotation geometry mismatch')
            valid=type(segments) is int and 1<=segments<=6 and all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in (duration,fps)) and duration>(segments-1)*5
            if 'status' in row:valid=valid and row['status']=='valid' and row.get('last_frame_decoded') is True
            if not valid:excluded.add(expected)
        if seen!={r['sha256'] for r in available.values()}:raise ValueError('Incomplete decoder coverage')
        available={key:r for key,r in available.items() if r['sha256'] not in excluded}
    counts={split:sum(key[0]==split for key in available) for split in ('train','val','test')}
    if any(counts[s]<n for s,n in [('train',100),('val',20),('test',40)]):raise ValueError('Insufficient available videos')
    content_groups={}
    for key,row in available.items():content_groups.setdefault(row['sha256'],[]).append(key)
    if any(len({key[0] for key in group})>1 for group in content_groups.values()):
        raise ValueError('Exact media content crosses source splits')
    aliases={key:min(group)[1] for group in content_groups.values() for key in group}
    alias_copies=sum(len(group)-1 for group in content_groups.values())
    staged={};media=[];diagnostics={}
    for split in ('train','val','test'):
        path=directory/f'{split}_data.json'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=HASHES[path.name]:raise ValueError('Changed official metadata')
        rows=json.loads(path.read_text(encoding='utf-8'))
        queries=[];seen_videos={};seen_ids=set()
        for source_index,row in enumerate(rows):
            key=(split,row['video'])
            if key not in available:continue
            n=row['num_segments'];moments(n)
            if row['video'] in seen_videos and seen_videos[row['video']]!=n:raise ValueError('Inconsistent video geometry')
            seen_videos[row['video']]=n
            if split=='test':continue  # Do not copy descriptions, target intervals or annotation IDs.
            identity=f'{split}:{source_index}:{row["annotation_id"]}'
            if identity in seen_ids:raise ValueError('Duplicate stable row identity')
            seen_ids.add(identity)
            times=[list(interval(t)) for t in row['times']]
            if not times or any(t[1]>=n for t in times):raise ValueError('Invalid human annotation geometry')
            text=row['description']
            if not isinstance(text,str) or not text.strip():raise ValueError('Empty human description')
            queries.append({'id':identity,'video':aliases[key],'source_video':row['video'],'text':text,'times':times})
        if len(seen_videos)!=counts[split]:raise ValueError('Receipt not represented by official split')
        canonical_geometry={}
        for name,n in seen_videos.items():
            canonical=aliases[(split,name)]
            if canonical in canonical_geometry and canonical_geometry[canonical]!=n:
                raise ValueError('Identical media with inconsistent annotated geometry')
            canonical_geometry[canonical]=n
        for name,n in sorted(canonical_geometry.items()):
            receipt=available[(split,name)];source=permitted[(split,name)]
            media.append({'split':split,'video':name,'num_segments':n,'sha256':receipt['sha256'],
                'bytes':receipt['bytes'],'aws_url':source['aws_url'],'license':source['license'],
                'original_url':source['original_url']})
        if split!='test':staged[f'{split}.json']=queries
        diagnostics[split]={'source_video_ids':len(seen_videos),'unique_media':len(canonical_geometry),'queries':len(queries) if split!='test' else None,
                            'test_targets_staged':False}
    staged['media.json']=media
    output.mkdir(parents=True,exist_ok=False)
    hashes={}
    for name,value in staged.items():
        content=(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
        (output/name).write_bytes(content);hashes[name]=hashlib.sha256(content).hexdigest()
    manifest={'source_revision':REVISION,'official_metadata_sha256':HASHES,
        'receipt_sha256':hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        'selection_file_sha256':hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        'files':hashes,'counts':diagnostics,'raw_media_redistributed':False,
        'decoder_preflight_sha256':decoder_hash,'excluded_decoder_media_sha256':sorted(excluded),
        'within_split_alias_copies':alias_copies,'alias_policy':'same-SHA video IDs coalesced; all human queries retained',
        'test_target_policy':'test JSON read only for video geometry; captions/intervals never staged',
        'availability_bias':'Fixed available CC subset; not full official benchmark; no score-based replacements'}
    (output/'manifest.json').write_bytes((json.dumps(manifest,indent=2)+'\n').encode())
    print(json.dumps(manifest,indent=2));return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--decoder-preflight')
    p.add_argument('--directory',default='data/external/didemo')
    p.add_argument('--receipts',default='data/interim/didemo-expanded-downloads/receipts.jsonl')
    p.add_argument('--selection',default='data/interim/didemo-expanded-selection.json')
    a=p.parse_args();stage(a.output,directory=a.directory,receipts=a.receipts,selection=a.selection,decoder_preflight=a.decoder_preflight)
