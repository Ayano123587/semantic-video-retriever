"""Validation-only configuration selection and one movie-held-out test run."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from src.models.encoder import CLIPEncoder
from src.models.index import VectorIndex

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def relevant(result,query):return result['video']==query['video'] and any(a<=result['timestamp_seconds']<=b for a,b in query['relevant_intervals_seconds'])
def metrics(ranked,queries):
    ranks=[next((i+1 for i,r in enumerate(rs) if relevant(r,q)),None) for rs,q in zip(ranked,queries)]
    return {'hit_at_1':sum(r==1 for r in ranks)/len(ranks),'hit_at_3':sum(r is not None for r in ranks)/len(ranks),'mrr_at_3':sum(1/r if r else 0 for r in ranks)/len(ranks)}

def rank(index,queries,encoder,prefix):return [index.query(encoder.text(prefix+q['query']),3,4.) for q in queries]
def select():
    queries=json.loads(Path('docs/validation-queries.json').read_text());encoder=CLIPEncoder();runs=[]
    for interval in [2,1]:
        index=VectorIndex.load(f'data/processed/validation-{interval}')
        for prefix in ['','a video frame of ']:
            ranked=rank(index,queries,encoder,prefix)
            runs.append({'interval':interval,'prefix':prefix,'metrics':metrics(ranked,queries),'rankings':ranked,'index_manifest_sha256':digest(f'data/processed/validation-{interval}/manifest.json')})
    winner=max(runs,key=lambda r:(r['metrics']['hit_at_1'],r['metrics']['hit_at_3'],r['interval'],r['prefix']==''))
    report={'selection_protocol':'validation Hit@1 then Hit@3; ties favor interval2 and raw query','validation_annotations_sha256':digest('docs/validation-queries.json'),'test_annotations_sha256':digest('docs/test-queries.json'),'winner':{'interval':winner['interval'],'prefix':winner['prefix']},'runs':runs,'test_evaluated':False}
    Path('models/selection.json').write_text(json.dumps(report,indent=2));print(json.dumps(report['winner']))

def diverse(records,order):
    out=[]
    for i in order:
        r=records[int(i)]
        if any(r['video_sha256']==x['video_sha256'] and abs(r['timestamp_seconds']-x['timestamp_seconds'])<4 for x in out):continue
        out.append(r)
        if len(out)==3:break
    return out

def test_once():
    output=Path('models/heldout-evaluation.json')
    if output.exists():raise ValueError('Frozen test has already been evaluated; preserve result and use a new independent test set for later studies')
    selection=json.loads(Path('models/selection.json').read_text());queries=json.loads(Path('docs/test-queries.json').read_text())
    if digest('docs/test-queries.json')!=selection['test_annotations_sha256']:raise ValueError('Frozen annotations changed')
    index=VectorIndex.load('data/processed/heldout')
    if index.metadata['sample_interval_seconds']!=selection['winner']['interval']:raise ValueError('Held-out index interval differs from selected interval')
    encoder=CLIPEncoder();ranked=rank(index,queries,encoder,selection['winner']['prefix'])
    rng=np.random.default_rng(42);draws=[]
    for _ in range(1000):draws.append(metrics([diverse(index.records,rng.permutation(len(index.records))) for q in queries],queries))
    random={k:float(np.mean([d[k] for d in draws])) for k in draws[0]}
    chronological=diverse(index.records,sorted(range(len(index.records)),key=lambda i:index.records[i]['timestamp_seconds']))
    report={'test_movie':'sintel-trailer.mp4','queries':len(queries),'selection_sha256':digest('models/selection.json'),'annotations_sha256':digest('docs/test-queries.json'),'index_metadata':index.metadata,'selected_config':selection['winner'],'metrics':metrics(ranked,queries),'random_baseline_1000_draws':random,'chronological_baseline':metrics([chronological]*len(queries),queries),'results':[dict(q,ranking=r) for q,r in zip(queries,ranked)],'limitation':'One movie-held-out trailer, eight manually annotated queries; unknown pretrained CLIP exposure. No claim of broad benchmark performance.'}
    output.write_text(json.dumps(report,indent=2));print(json.dumps(report['metrics']))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['select','test']);a=p.parse_args();select() if a.action=='select' else test_once()
if __name__=='__main__':main()
