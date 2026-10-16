"""Stage a private, readable, hash-bound one-shot DiDeMo test release.

This prepares files only. It never uploads, launches, parses test gold, or
recreates a release marker. Do not rerun a launched release to evade its marker.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from src.models.moment_evaluation import POLICY, MARKER, verified_inputs
from src.models.moment_training import digest


def stage(output, source, cache, checkpoint, official_test, dataset_id):
    root=Path(__file__).resolve().parents[1]
    output=Path(output).resolve();source=Path(source);cache=Path(cache)
    if output.exists():raise ValueError('Fresh private staging directory required')
    if (source/MARKER).exists() or (source/'test-dispatch-receipt.json').exists():
        raise ValueError('Release dispatched or attempted; preserve evidence instead of restaging')
    identities,_,_,_,_=verified_inputs(source,cache,checkpoint)
    policy=json.loads((source/POLICY).read_text(encoding='utf-8'))
    if any(policy.get(k)!=v for k,v in identities.items()) or policy.get('test_targets_used') is not False:
        raise ValueError('Frozen policy mismatch')
    if digest(official_test)!=identities['official_test_sha256']:raise ValueError('Official test bytes changed')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    paths=['src/__init__.py','src/config.py','src/data/__init__.py','src/data/didemo.py',
           'src/models/__init__.py','src/models/encoder.py','src/models/moments.py',
           'src/models/moment_training.py','src/models/moment_evaluation.py','scripts/kaggle_test_release.py']
    # Require the exact source being staged to exist unchanged in this commit.
    for name in paths:
        committed=subprocess.check_output(['git','show',commit+':'+name],cwd=root)
        if committed.replace(b'\r\n',b'\n')!=(root/name).read_bytes().replace(b'\r\n',b'\n'):
            raise ValueError('Commit reviewed source before staging: '+name)
    output.mkdir(parents=True);entries=[]
    def copy(original,target):
        content=Path(original).read_bytes();flat=target.replace('/','__')
        (output/flat).write_bytes(content)
        entries.append({'path':target,'flat_name':flat,'sha256':hashlib.sha256(content).hexdigest()})
    for name in paths:copy(root/name,name)
    prepared=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    encoded=json.loads((cache/'manifest.json').read_text(encoding='utf-8'))
    for name in ['manifest.json',*prepared['files'],POLICY]:copy(source/name,'prepared/'+name)
    for name in ['manifest.json',*encoded['files']]:copy(cache/name,'cache/'+name)
    copy(checkpoint,'selected.pt');copy(official_test,'official-test.json')
    manifest={'schema':1,'source_commit':commit,'files':entries,'identities':identities,
              'frozen_policy_sha256':digest(source/POLICY),'test_gold_parsed':False,
              'scope':'one-shot private CPU release; no raw videos or training'}
    m=output/'video-test-release-manifest.json'
    m.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
    sha=digest(m)
    # Plain readable launcher binds both runner bytes and the entire private stage.
    runner=next(r for r in entries if r['path']=='scripts/kaggle_test_release.py')
    launcher="""import hashlib, os, runpy
from pathlib import Path
files=list(Path('/kaggle/input').rglob('video-test-release-manifest.json'))
if len(files)!=1:raise ValueError('Exactly one private release dataset required')
manifest=files[0]
if hashlib.sha256(manifest.read_bytes()).hexdigest()!=%r:raise ValueError('Wrong release manifest')
runner=manifest.parent/'scripts__kaggle_test_release.py'
if hashlib.sha256(runner.read_bytes()).hexdigest()!=%r:raise ValueError('Wrong runner source')
os.environ['VIDEO_RELEASE_MANIFEST_SHA256']=%r
runpy.run_path(str(runner),run_name='__main__')
"""%(sha,runner['sha256'],sha)
    (output/'kernel.py').write_text(launcher,encoding='utf-8',newline='\n')
    (output/'dataset-metadata.json').write_text(json.dumps({'title':'Private frozen DiDeMo test release',
        'id':dataset_id,'licenses':[{'name':'other'}]},indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'files':len(entries),'source_commit':commit,'manifest_sha256':sha,'output':str(output),'uploaded':False}))
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['output','source','cache','checkpoint','official-test','dataset-id']:p.add_argument('--'+name,required=True)
    a=p.parse_args();stage(a.output,a.source,a.cache,a.checkpoint,a.official_test,a.dataset_id)
