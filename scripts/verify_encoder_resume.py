"""Compare actual pretrained encoding uninterrupted and across a checkpoint."""
import argparse
import json
from pathlib import Path
import numpy as np
from src.models.moment_encoder import encode


def verify(source,media,output):
    output=Path(output)
    if output.exists():raise ValueError('Fresh verification output required')
    output.mkdir(parents=True)
    full=encode(source,media,output/'uninterrupted',smoke=True,stream_download=True)
    partial=encode(source,media,output/'interrupted',smoke=True,stream_download=True,max_seconds=1e-12)
    if partial['status']!='partial' or partial['encoded_videos']!=1:
        raise ValueError('Expected an actual one-video checkpoint')
    resumed=encode(source,media,output/'resumed',smoke=True,stream_download=True,
                   resume_from=output/'interrupted'/'encoding-progress')
    if resumed['status']!='complete' or resumed['resumed_videos']!=1:
        raise ValueError('Expected completed encoding with one reused video')
    if full['encoder']!=resumed['encoder']:raise ValueError('Encoder identity changed')
    if full['encoding_identity']!=resumed['encoding_identity']:
        raise ValueError('Encoding algorithm, source order or execution identity changed')
    comparisons={}
    for name in full['files']:
        a=output/'uninterrupted'/name;b=output/'resumed'/name
        if name.endswith('.npy'):
            x=np.load(a,allow_pickle=False);y=np.load(b,allow_pickle=False)
            if x.shape!=y.shape:raise ValueError('Shape mismatch')
            error=float(np.max(np.abs(x-y)))
            if error>1e-6:raise ValueError('Interrupted encoding differs from uninterrupted encoding')
            comparisons[name]={'maximum_absolute_difference':error}
        else:
            if a.read_bytes()!=b.read_bytes():raise ValueError('Metadata replay mismatch: '+name)
            comparisons[name]={'identical_bytes':True}
    receipt={'actual_pretrained_encoding':True,'full':full,'interrupted':partial,
             'resumed':resumed,'comparisons':comparisons,'benchmark_performance_claim':False}
    (output/'resume-verification.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--media',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();verify(a.source,a.media,a.output)
