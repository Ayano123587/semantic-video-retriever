"""Inspectable exact cosine index; no unsafe pickle persistence."""
import hashlib,json
from pathlib import Path
import numpy as np
from src.config import MODEL_ID,MODEL_REVISION

def normalize(v):
    v=np.asarray(v,dtype=np.float32)
    if v.ndim!=2 or not len(v) or not np.isfinite(v).all():raise ValueError('Expected nonempty finite 2D matrix')
    norms=np.linalg.norm(v,axis=1,keepdims=True)
    if (norms<=0).any():raise ValueError('Zero vector')
    return v/norms

class VectorIndex:
    def __init__(self,vectors,records,metadata=None):
        self.vectors=normalize(vectors);self.records=records;self.metadata=metadata or {}
        if len(vectors)!=len(records):raise ValueError('Vector/record count mismatch')
        for r in records:
            if not np.isfinite(r['timestamp_seconds']) or r['timestamp_seconds']<0:raise ValueError('Invalid timestamp')
    def save(self,path):
        p=Path(path);p.mkdir(parents=True,exist_ok=True);np.save(p/'vectors.npy',self.vectors,allow_pickle=False)
        metadata=dict(self.metadata);metadata['vectors_sha256']=hashlib.sha256((p/'vectors.npy').read_bytes()).hexdigest()
        (p/'manifest.json').write_text(json.dumps({'metadata':metadata,'records':self.records},indent=2),encoding='utf-8')
    @classmethod
    def load(cls,path):
        p=Path(path);m=json.loads((p/'manifest.json').read_text(encoding='utf-8'))
        if hashlib.sha256((p/'vectors.npy').read_bytes()).hexdigest()!=m['metadata']['vectors_sha256']:raise ValueError('Index integrity mismatch')
        return cls(np.load(p/'vectors.npy',allow_pickle=False),m['records'],m['metadata'])
    def query(self,vector,k=5,suppress_seconds=4.):
        if k<1 or suppress_seconds<0:raise ValueError('Invalid retrieval parameters')
        q=normalize(np.asarray(vector).reshape(1,-1))[0]
        if len(q)!=self.vectors.shape[1]:raise ValueError('Query dimension mismatch')
        scores=self.vectors@q;chosen=[]
        for i in np.argsort(-scores,kind='stable'):
            record=self.records[int(i)]
            if any(record['video_sha256']==r['video_sha256'] and abs(record['timestamp_seconds']-r['timestamp_seconds'])<suppress_seconds for r in chosen):continue
            chosen.append(dict(record,cosine_similarity=float(scores[i])))
            if len(chosen)==k:break
        return chosen

def deduplicate(vectors,records,threshold=.985):
    if not 0<threshold<=1:raise ValueError('Invalid duplicate threshold')
    vectors=normalize(vectors);keep=[]
    for i,r in enumerate(records):
        # Only collapse near-identical embeddings within the same sampled scene.
        candidates=[j for j in keep if records[j]['video_sha256']==r['video_sha256'] and records[j]['scene_id']==r['scene_id']]
        if not any(float(vectors[i]@vectors[j])>=threshold for j in candidates):keep.append(i)
    return vectors[keep],[records[i] for i in keep]
