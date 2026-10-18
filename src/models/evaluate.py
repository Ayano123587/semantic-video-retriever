"""Small manual interval-annotation demo; not a broad video retrieval benchmark."""
import argparse,json
from pathlib import Path
from src.models.encoder import CLIPEncoder
from src.models.index import VectorIndex
from src.models.predict_model import search

def evaluate(index_path,annotations,output):
    output=Path(output)
    if output.exists():raise ValueError('Fresh demo report path required')
    index=VectorIndex.load(index_path);queries=json.loads(Path(annotations).read_text(encoding='utf-8'))
    if not isinstance(queries,list) or not queries:raise ValueError('Nonempty manual query list required')
    videos={r['video_sha256'] for r in index.records};targets=[]
    for q in queries:
        target=q.get('video_sha256',next(iter(videos)) if len(videos)==1 else None)
        if target not in videos:raise ValueError('Manual annotations must identify the target video in a multi-video index')
        targets.append(target)
    encoder=CLIPEncoder();results=[]
    for q,target in zip(queries,targets):
        ranked=search(index,q['query'],3,4.,encoder)
        relevant=lambda r:r['video_sha256']==target and any(start<=r['timestamp_seconds']<=end for start,end in q['relevant_intervals_seconds'])
        hits=[relevant(r) for r in ranked]
        results.append(dict(q,target_video_sha256=target,results=ranked,hit_at_1=bool(hits and hits[0]),hit_at_3=any(hits)))
    report={'queries':len(results),'hit_at_1':sum(r['hit_at_1'] for r in results)/len(results),'hit_at_3':sum(r['hit_at_3'] for r in results)/len(results),'scope':'User-supplied manual interval annotations; descriptive demo, not a research benchmark','model':index.metadata,'results':results}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as handle:json.dump(report,handle,indent=2)
    print(json.dumps(report,indent=2))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--index',default='data/processed/index');p.add_argument('--annotations',default='docs/demo-queries.json');p.add_argument('--output',required=True);a=p.parse_args();evaluate(a.index,a.annotations,a.output)
if __name__=='__main__':main()
