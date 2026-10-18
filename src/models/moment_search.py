"""Search cached video moments with the exact validation-selected policy."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from src.config import MODEL_ID,MODEL_REVISION
from src.models.encoder import CLIPEncoder,WEIGHTS_SHA256
from src.models.moment_training import Adapter,digest
from src.models.localization_followup import score_policy
from src.models.moments import interval


class MomentSearch:
    def __init__(self,cache,checkpoint,checkpoint_sha256,split='val'):
        if split not in ('train','val','test'):raise ValueError('Unknown media split')
        cache=Path(cache);checkpoint=Path(checkpoint)
        if digest(checkpoint)!=checkpoint_sha256:raise ValueError('Selected checkpoint hash mismatch')
        self.checkpoint_sha256=checkpoint_sha256
        self.manifest_sha256=digest(cache/'manifest.json')
        manifest=json.loads((cache/'manifest.json').read_text(encoding='utf-8'))
        self.bundle=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if self.bundle.get('cache_manifest_sha256')!=self.manifest_sha256:
            raise ValueError('Checkpoint feature cache mismatch')
        self.encoder_receipt=manifest['encoder']
        if self.bundle.get('encoder')!=self.encoder_receipt:raise ValueError('Checkpoint encoder mismatch')
        if (self.encoder_receipt.get('model_id'),self.encoder_receipt.get('revision'),self.encoder_receipt.get('weights_sha256'))!=(MODEL_ID,MODEL_REVISION,WEIGHTS_SHA256):
            raise ValueError('Unrecognized pretrained encoder')
        if any(self.encoder_receipt.get('loading_info',{}).get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
            raise ValueError('Incomplete pretrained loading')
        if self.bundle.get('schema')!=1 or not isinstance(self.bundle.get('states'),list) or len(self.bundle['states']) not in (0,3):
            raise ValueError('Invalid selected policy schema')
        for state in self.bundle['states']:
            model=Adapter();model.load_state_dict(state,strict=True)
            if any(t.dtype!=torch.float32 or not torch.isfinite(t).all() for t in state.values()):
                raise ValueError('Invalid adapter state')
        names=(f'{split}-moments.npy',f'{split}-candidates.json','frames.json')
        for name in names:
            if name not in manifest['files'] or digest(cache/name)!=manifest['files'][name]:
                raise ValueError('Changed moment index file: '+name)
        values=np.load(cache/names[0],allow_pickle=False)
        self.candidates=json.loads((cache/names[1]).read_text(encoding='utf-8'))
        if values.shape!=(len(self.candidates),512) or not len(values) or values.dtype!=np.float32:
            raise ValueError('Invalid moment feature geometry')
        if not np.isfinite(values).all() or not np.allclose(np.linalg.norm(values,axis=1),1,atol=1e-4):
            raise ValueError('Invalid moment features')
        self.features=torch.from_numpy(values)
        frames=json.loads((cache/'frames.json').read_text(encoding='utf-8'))
        self.duration={r['video']:float(r['duration']) for r in frames if r['split']==split}
        keys=set()
        for row in self.candidates:
            key=(row['video'],interval(row['moment']))
            if key in keys:raise ValueError('Duplicate moment candidate')
            keys.add(key)
            duration=self.duration.get(row['video'],float('nan'))
            if not np.isfinite(duration) or duration<=key[1][0]*5:
                raise ValueError('Invalid source duration')

    def search_vector(self,vector,top_k=5):
        if type(top_k) is not int or top_k<1:raise ValueError('Positive integer top-k required')
        vector=np.asarray(vector,dtype=np.float32)
        if vector.shape!=(512,) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector),1,atol=1e-4):
            raise ValueError('Unit query vector required')
        scores=score_policy(self.bundle,torch.from_numpy(vector[None]),self.features)[0]
        order=np.argsort(-scores,kind='stable')[:top_k]
        return [{'video':self.candidates[i]['video'],'start_seconds':self.candidates[i]['moment'][0]*5,
                 'end_seconds':min((self.candidates[i]['moment'][1]+1)*5,self.duration[self.candidates[i]['video']]),
                 'score':float(scores[i])} for i in order]

    def search(self,query,top_k=5,encoder=None):
        if not isinstance(query,str) or not query.strip():raise ValueError('Nonempty query required')
        encoder=encoder or CLIPEncoder()
        if encoder.receipt!=self.encoder_receipt:raise ValueError('Query encoder identity mismatch')
        return self.search_vector(encoder.text(query),top_k)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('query')
    p.add_argument('--cache',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--checkpoint-sha256',required=True);p.add_argument('--split',default='val',choices=['train','val','test'])
    p.add_argument('--top-k',type=int,default=5);a=p.parse_args()
    index=MomentSearch(a.cache,a.checkpoint,a.checkpoint_sha256,a.split)
    print(json.dumps({'checkpoint_sha256':index.checkpoint_sha256,'cache_manifest_sha256':index.manifest_sha256,
                      'matches':index.search(a.query,a.top_k)},indent=2))


if __name__=='__main__':main()
