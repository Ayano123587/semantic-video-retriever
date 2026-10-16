import argparse,json
from src.config import MODEL_ID,MODEL_REVISION
from src.models.encoder import CLIPEncoder
from src.models.index import VectorIndex

def search(index,query,k=5,suppress_seconds=4.,encoder=None):
    if not query.strip():raise ValueError('Query must not be empty')
    if index.metadata.get('model_id')!=MODEL_ID or index.metadata.get('model_revision')!=MODEL_REVISION:raise ValueError('Index encoder revision mismatch')
    encoder=encoder or CLIPEncoder();return index.query(encoder.text(query),k,suppress_seconds)

def main():
    p=argparse.ArgumentParser();p.add_argument('query');p.add_argument('--index',default='data/processed/index');p.add_argument('--top-k',type=int,default=5);p.add_argument('--suppress-seconds',type=float,default=4.);a=p.parse_args();print(json.dumps(search(VectorIndex.load(a.index),a.query,a.top_k,a.suppress_seconds),indent=2))
if __name__=='__main__':main()
