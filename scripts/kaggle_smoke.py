"""Plain private Kaggle CPU smoke runner; source is mounted as readable Python."""
import hashlib,json,os,shutil,subprocess,sys
from pathlib import Path


def main(smoke=True):
    subprocess.run([sys.executable,'-m','pip','install','-q','transformers==4.57.6',
                    'huggingface-hub>=0.34,<1','imageio>=2.36,<3','imageio-ffmpeg>=0.6,<1','pillow>=10,<13'],check=True)
    candidates=list(Path('/kaggle/input').rglob('video-source-manifest.json'))
    if len(candidates)!=1:raise ValueError('Expected exactly one versioned readable source manifest')
    source=candidates[0].parent;manifest=json.loads(candidates[0].read_text())
    workspace=Path('/kaggle/working/video-workspace');workspace.mkdir(exist_ok=False)
    for row in manifest['files']:
        original=source/row['flat_name'];content=original.read_bytes()
        if hashlib.sha256(content).hexdigest()!=row['sha256']:raise ValueError('Mounted source hash mismatch')
        target=(workspace/row['path']).resolve()
        if not target.is_relative_to(workspace):raise ValueError('Unsafe source path')
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
    os.chdir(workspace);sys.path.insert(0,str(workspace))
    from src.data.didemo import download_one
    from src.models.moment_encoder import encode
    staged=workspace/'prepared';records=json.loads((staged/'media.json').read_text())
    chosen=[next(r for r in records if r['split']==s) for s in ('train','val')] if smoke else records
    def fetch(record):
        receipt=download_one(record,'media')
        if receipt['sha256']!=record['sha256']:raise ValueError('Cloud media differs from frozen source')
        return receipt
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as pool:receipts=list(pool.map(fetch,chosen))
    cache=Path('/kaggle/working/video-smoke' if smoke else '/kaggle/working/video-features-v1')
    result=encode(staged,'media',cache,smoke=smoke)
    if not smoke:
        from src.models.moment_training import train
        train(cache,Path('/kaggle/working/video-adapters-v1'))
    Path('/kaggle/working/smoke-environment.json').write_text(json.dumps({
        'python':sys.version,'source_commit':manifest['source_commit'],'result':result},indent=2))
    # Publish only the private cache/evidence as output, not copied raw data/source.
    shutil.rmtree(workspace)

if __name__=='__main__':main()
