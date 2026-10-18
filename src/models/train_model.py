"""Build an index with pretrained CLIP. No task-specific training is claimed."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from src.config import MODEL_ID,MODEL_REVISION
from src.features.build_features import sample
from src.models.encoder import CLIPEncoder
from src.models.index import VectorIndex,deduplicate

def build(videos,output,interval=2.,device='cpu',batch_size=8):
    if batch_size<1:raise ValueError('batch_size must be positive')
    encoder=CLIPEncoder(device);embeddings=[];records=[];sources=[]
    for video in videos:
        video=Path(video)
        with video.open('rb') as source:digest=hashlib.file_digest(source,'sha256').hexdigest()
        batch=[];pending=[]
        sources.append({'video':video.name,'sha256':digest,'bytes':video.stat().st_size})
        for image,record in sample(video,interval):
            batch.append(image);pending.append(dict(record,video=video.name,video_sha256=digest))
            if len(batch)==batch_size:embeddings.append(encoder.images(batch));records.extend(pending);batch=[];pending=[]
        if batch:embeddings.append(encoder.images(batch));records.extend(pending)
    if not embeddings:raise ValueError('No video frames')
    before=len(records);vectors,records=deduplicate(np.concatenate(embeddings),records)
    metadata={'model_id':MODEL_ID,'weights_sha256':'a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f','model_revision':MODEL_REVISION,'embedding_dim':vectors.shape[1],'sample_interval_seconds':interval,'sampled_frames':before,'indexed_frames':len(records),'dedup_threshold':.985,'sources':sources,'method':'pretrained CLIP exact cosine; sampled RGB-change shot heuristic; no task-specific training'}
    index=VectorIndex(vectors,records,metadata);index.save(output)
    Path('models').mkdir(exist_ok=True);Path('models/index-summary.json').write_text(json.dumps(metadata,indent=2));return metadata

def main():
    p=argparse.ArgumentParser();p.add_argument('videos',nargs='+',type=Path);p.add_argument('--output',default='data/processed/index');p.add_argument('--interval',type=float,default=2.);p.add_argument('--device',default='cpu');a=p.parse_args();print(json.dumps(build(a.videos,a.output,a.interval,a.device),indent=2))
if __name__=='__main__':main()
