"""Stage plain readable versioned Python and private real annotations for Kaggle."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path


def stage(output,prepared):
    root=Path(__file__).resolve().parents[1];output=Path(output).resolve()
    if output.exists():raise ValueError('Refuse to overwrite source snapshot')
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():raise ValueError('Commit reviewed source before staging')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    paths=['src/__init__.py','src/config.py','src/models/__init__.py','src/models/encoder.py',
           'src/models/moments.py','src/models/moment_encoder.py','src/models/moment_training.py','src/data/__init__.py','src/data/didemo.py']
    entries=[];output.mkdir(parents=True,exist_ok=False)
    for name in paths:
        content=(root/name).read_bytes();flat=name.replace('/','__')
        (output/flat).write_bytes(content);entries.append({'path':name,'flat_name':flat,'sha256':hashlib.sha256(content).hexdigest()})
    for name in ('train.json','val.json','media.json','manifest.json'):
        content=(Path(prepared)/name).read_bytes();flat='prepared__'+name
        (output/flat).write_bytes(content);entries.append({'path':'prepared/'+name,'flat_name':flat,'sha256':hashlib.sha256(content).hexdigest()})
    (output/'video-source-manifest.json').write_text(json.dumps({'source_commit':commit,'files':entries},indent=2),encoding='utf-8')
    (output/'dataset-metadata.json').write_text(json.dumps({'title':'Private Portfolio DiDeMo Source V1',
        'id':'aikyatansinha/portfolio-didemo-source-v1','licenses':[{'name':'other'}]},indent=2),encoding='utf-8')
    print(json.dumps({'commit':commit,'files':len(entries),'output':str(output)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--prepared',required=True)
    a=p.parse_args();stage(a.output,a.prepared)
