"""Private one-shot CPU test runner. Launch only after review and slot allocation."""
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


def main():
    expected=os.environ.get('VIDEO_RELEASE_MANIFEST_SHA256')
    if not expected or len(expected)!=64:raise ValueError('Use the hash-bound staged plain launcher')
    matches=list(Path('/kaggle/input').rglob('video-test-release-manifest.json'))
    if len(matches)!=1:raise ValueError('Expected exactly one private release manifest')
    path=matches[0]
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Staged manifest mismatch')
    manifest=json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('schema')!=1 or manifest.get('test_gold_parsed') is not False:raise ValueError('Invalid stage')
    workspace=Path('/kaggle/working/video-test-release-workspace')
    workspace.mkdir(exist_ok=False)
    seen=set()
    for entry in manifest['files']:
        relative=Path(entry['path']);flat=entry['flat_name']
        if relative.is_absolute() or '..' in relative.parts or relative.as_posix() in seen or Path(flat).name!=flat:
            raise ValueError('Unsafe or duplicate staged path')
        seen.add(relative.as_posix());content=(path.parent/flat).read_bytes()
        if hashlib.sha256(content).hexdigest()!=entry['sha256']:raise ValueError('Changed private artifact')
        target=workspace/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
    # Match the verified Transformers version; the encoder independently attests pinned CLIP weights.
    subprocess.run([sys.executable,'-m','pip','install','-q','transformers==4.57.6',
                    'huggingface-hub>=0.34,<1','pillow>=10,<13'],check=True)
    sys.path.insert(0,str(workspace));os.chdir(workspace)
    import torch
    torch.set_num_threads(2)
    from src.models.moment_evaluation import release, verified_inputs, POLICY, MARKER
    from src.models.moment_training import digest
    source=workspace/'prepared';cache=workspace/'cache';checkpoint=workspace/'selected.pt'
    identities,_,_,_,_=verified_inputs(source,cache,checkpoint)
    if identities!=manifest['identities'] or digest(source/POLICY)!=manifest['frozen_policy_sha256']:
        raise ValueError('Frozen launch identities changed')
    output=Path('/kaggle/working/video-final-test.json')
    receipt=Path('/kaggle/working/video-test-release-receipt.json')
    if receipt.exists():raise ValueError('Prior release evidence exists')
    start=time.monotonic();status='failed'
    try:
        # release creates its exclusive prepared-source marker BEFORE parsing official-test.json.
        result=release(source,cache,checkpoint,workspace/'official-test.json',output)
        status='complete'
        print(json.dumps({'status':status,'queries':result['selected']['queries'],'winner':result['winner']}))
    finally:
        marker=source/MARKER
        if marker.exists():shutil.copyfile(marker,Path('/kaggle/working')/MARKER)
        receipt.write_text(json.dumps({'status':status,'source_commit':manifest['source_commit'],
            'staging_manifest_sha256':expected,'frozen_policy_sha256':manifest['frozen_policy_sha256'],
            'identities':identities,'seconds':time.monotonic()-start,'python':platform.python_version(),
            'torch':str(torch.__version__),'threads':torch.get_num_threads(),'marker_preserved':marker.exists()},
            indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    # Workspace remains private, including the authoritative source-directory marker.
    # Never rerun automatically after failure or copy it away to obtain another test attempt.

if __name__=='__main__':main()
