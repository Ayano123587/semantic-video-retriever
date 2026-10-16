"""Small manual interval-annotation demo; not a broad video retrieval benchmark."""
import argparse,json
from pathlib import Path
from src.models.encoder import CLIPEncoder
from src.models.index import VectorIndex
from src.models.predict_model import search

def evaluate(index_path,annotations,output):
    index=VectorIndex.load(index_path);queries=json.loads(Path(annotations).read_text());encoder=CLIPEncoder();results=[]
    for q in queries:
        ranked=search(index,q['query'],3,4.,encoder)
        relevant=lambda r:any(start<=r['timestamp_seconds']<=end for start,end in q['relevant_intervals_seconds'])
        hits=[relevant(r) for r in ranked]
        results.append(dict(q,results=ranked,hit_at_1=bool(hits and hits[0]),hit_at_3=any(hits)))
    report={'queries':len(results),'hit_at_1':sum(r['hit_at_1'] for r in results)/len(results),'hit_at_3':sum(r['hit_at_3'] for r in results)/len(results),'scope':'4 manually annotated queries on one 30-second CC-BY excerpt; descriptive demo, no claim of benchmark generalization','model':index.metadata,'results':results}
    Path(output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
def main():
    p=argparse.ArgumentParser();p.add_argument('--index',default='data/processed/index');p.add_argument('--annotations',default='docs/demo-queries.json');p.add_argument('--output',default='models/demo-evaluation.json');a=p.parse_args();evaluate(a.index,a.annotations,a.output)
if __name__=='__main__':main()
