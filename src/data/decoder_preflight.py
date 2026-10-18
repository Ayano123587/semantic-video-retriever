"""Inspect every retained source before fitting; preserve per-source evidence."""
import argparse
import hashlib
import json
from pathlib import Path
from src.data.didemo import HASHES,media_filename,decoder_source
from src.models.moment_encoder import frame_schedule


def inspect(record,path,reader_factory=None):
    path=Path(path)
    with path.open('rb') as handle:actual=hashlib.file_digest(handle,'sha256').hexdigest()
    if actual!=record['sha256']:raise ValueError('Raw media hash mismatch')
    if reader_factory is None:
        import imageio.v2 as imageio
        reader_factory=lambda p:imageio.get_reader(p,format='ffmpeg')
    result={k:record[k] for k in ('split','video','sha256','num_segments')}
    result.update(duration=0.,fps=0.,last_frame_decoded=False)
    reader=None
    try:
        with decoder_source(path) as compatible:
            reader=reader_factory(compatible)
            try:
                meta=reader.get_meta_data()
                duration=float(meta['duration']);fps=float(meta['fps'])
                schedule=frame_schedule(record['num_segments'],duration,fps)
                frame=reader.get_data(schedule[-1]['frame'])
            finally:reader.close();reader=None
        if getattr(frame,'ndim',0)!=3 or min(frame.shape[:2])<=0:
            raise ValueError('Invalid decoded frame')
        result.update(duration=duration,fps=fps,last_frame_decoded=True,status='valid',
                      last_frame=schedule[-1]['frame'])
    except (ValueError,OSError,RuntimeError,IndexError,KeyError) as error:
        result.update(status='invalid',error=type(error).__name__+': '+str(error))
    finally:
        if reader is not None:reader.close()
    return result


def preflight(receipts,directory,media,output):
    receipt_path=Path(receipts);directory=Path(directory);output=Path(output)
    rows=[json.loads(line) for line in receipt_path.read_text(encoding='utf-8').splitlines()]
    geometry={}
    for split in ('train','val','test'):
        path=directory/f'{split}_data.json';content=path.read_bytes()
        if hashlib.sha256(content).hexdigest()!=HASHES[path.name]:raise ValueError('Official source changed')
        for row in json.loads(content):
            key=(split,row['video']);n=row['num_segments']
            if key in geometry and geometry[key]!=n:raise ValueError('Inconsistent source geometry')
            geometry[key]=n
    retained={};aliases={}
    for row in rows:
        if row['status']!='complete':continue
        record={**row,'num_segments':geometry[(row['split'],row['video'])]}
        retained.setdefault(row['sha256'],record)
        aliases.setdefault(row['sha256'],{})[(row['split'],row['video'])]={
            'split':row['split'],'video':row['video'],'num_segments':record['num_segments']}
    if output.exists():raise ValueError('Fresh decoder output required')
    output.mkdir(parents=True);results=[]
    with (output/'rows.jsonl').open('x',encoding='utf-8') as handle:
        for record in retained.values():
            provenance=sorted(aliases[record['sha256']].values(),key=lambda r:(r['split'],r['video']))
            if len({r['num_segments'] for r in provenance})>1:
                # No geometry wins: exclude every alias of these identical bytes.
                result={k:record[k] for k in ('split','video','sha256','num_segments')}
                result.update(status='invalid',reason='annotation_geometry_conflict',
                              aliases=provenance,duration=0.,fps=0.,last_frame_decoded=False,
                              geometry_inspected=False)
            else:
                result=inspect(record,Path(media)/media_filename(record['video']))
            results.append(result)
            handle.write(json.dumps(result)+'\n');handle.flush()
            print(json.dumps({'inspected':len(results),'total':len(retained),'status':result['status']}),flush=True)
    report={'rows':results,'receipt_sha256':hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'valid':sum(r['status']=='valid' for r in results),'total':len(results),
            'targets_used_for_filtering':False}
    (output/'preflight.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipts',required=True);p.add_argument('--directory',default='data/external/didemo')
    p.add_argument('--media',default='data/raw/didemo');p.add_argument('--output',required=True)
    a=p.parse_args();preflight(a.receipts,a.directory,a.media,a.output)


if __name__=='__main__':main()
