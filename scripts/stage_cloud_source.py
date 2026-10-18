"""Stage plain readable versioned Python and private real annotations for Kaggle."""
import argparse,hashlib,json,shutil,subprocess,sys
from pathlib import Path


def research_inputs(root,prepared,proof):
    """Carry source-only audit inputs at their recorded portable paths."""
    root=Path(root).resolve();proof=Path(proof).resolve();prepared=Path(prepared).resolve()
    if not proof.is_relative_to(root/'data') or prepared!=proof/'prepared':
        raise ValueError('Research proof and preparation must share the repository data directory')
    audit_path=(proof/'source-accounting.json').resolve()
    if not audit_path.is_relative_to(root/'data'):raise ValueError('Unsafe research proof file')
    audit=json.loads(audit_path.read_text(encoding='utf-8'))
    paths=[]
    for name,expected in audit['input_sha256'].items():
        relative=Path(name)
        if relative.is_absolute() or ':' in name or '..' in relative.parts or not relative.parts or relative.parts[0]!='data':
            raise ValueError('Portable repository data paths required for original audit inputs')
        source=(root/relative).resolve()
        if not source.is_relative_to(root/'data') or source.suffix not in ('.json','.jsonl'):
            raise ValueError('Unsafe research source input')
        if hashlib.sha256(source.read_bytes()).hexdigest()!=expected:
            raise ValueError('Original research audit input changed')
        paths.append((source,relative.as_posix()))
    for name in ('source-accounting.json','resolved-receipts.jsonl','decoder/preflight.json',
                 'prepared/manifest.json','prepared/train.json','prepared/val.json','prepared/media.json'):
        source=(proof/name).resolve()
        if not source.is_relative_to(root/'data'):raise ValueError('Unsafe research proof file')
        paths.append((source,source.relative_to(root).as_posix()))
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from src.data.decoder_recovery import checked_recovery
    decoded=json.loads((proof/'decoder/preflight.json').read_text(encoding='utf-8'))
    extra=checked_recovery(proof,decoded,hashlib.sha256((proof/'resolved-receipts.jsonl').read_bytes()).hexdigest(),root/'src/data/decoder_preflight.py') if decoded.get('recovery') is not None else []
    for source in extra:
        if not source.is_relative_to(root/'data'):raise ValueError('Unsafe decoder recovery artifact')
        paths.append((source,source.relative_to(root).as_posix()))
    for name in ('train_data.json','val_data.json'):
        source=(root/'data/external/didemo'/name).resolve()
        if not source.is_relative_to(root/'data'):raise ValueError('Unsafe official source file')
        paths.append((source,source.relative_to(root).as_posix()))
    return paths


def stage(output,prepared,research_preparation=None):
    root=Path(__file__).resolve().parents[1];output=Path(output).resolve()
    if output.exists():raise ValueError('Refuse to overwrite source snapshot')
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():raise ValueError('Commit reviewed source before staging')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    proof_paths=research_inputs(root,prepared,research_preparation) if research_preparation is not None else []
    paths=['src/__init__.py','src/config.py','src/models/__init__.py','src/models/encoder.py',
           'src/models/moments.py','src/models/moment_encoder.py','src/models/encoding_progress.py','src/models/moment_training.py','src/models/positive_index.py',
           'src/models/localization_followup.py','src/models/research_followup.py','src/models/resumable_training.py','src/models/moment_search.py',
           'src/data/__init__.py','src/data/didemo.py','src/data/research_receipts.py','src/data/decoder_preflight.py','src/data/decoder_recovery.py','src/data/prepare_moments.py',
           'scripts/verify_encoder_resume.py']
    entries=[];output.mkdir(parents=True,exist_ok=False)
    for name in paths:
        content=(root/name).read_bytes();flat=name.replace('/','__')
        (output/flat).write_bytes(content);entries.append({'path':name,'flat_name':flat,'sha256':hashlib.sha256(content).hexdigest()})
    for name in ('train.json','val.json','media.json','manifest.json'):
        content=(Path(prepared)/name).read_bytes();flat='prepared__'+name
        (output/flat).write_bytes(content);entries.append({'path':'prepared/'+name,'flat_name':flat,'sha256':hashlib.sha256(content).hexdigest()})
    if proof_paths:
        for source,name in proof_paths:
            content=source.read_bytes();flat=name.replace('/','__')
            if any(row['path']==name for row in entries):raise ValueError('Duplicate staged research path')
            (output/flat).write_bytes(content)
            entries.append({'path':name,'flat_name':flat,'sha256':hashlib.sha256(content).hexdigest()})
    (output/'video-source-manifest.json').write_text(json.dumps({'source_commit':commit,'files':entries},indent=2),encoding='utf-8')
    (output/'dataset-metadata.json').write_text(json.dumps({'title':'Private Portfolio DiDeMo Source V1',
        'id':'aikyatansinha/portfolio-didemo-source-v1','licenses':[{'name':'other'}]},indent=2),encoding='utf-8')
    print(json.dumps({'commit':commit,'files':len(entries),'output':str(output)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--prepared',required=True)
    p.add_argument('--research-preparation',help='Complete source accounting/decoder/prepared proof directory')
    a=p.parse_args();stage(a.output,a.prepared,a.research_preparation)
