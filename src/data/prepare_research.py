"""Prepare the whole selected cohort only after complete source accounting."""
import argparse
import hashlib
import json
from pathlib import Path
from src.data.research_receipts import reconcile
from src.data.decoder_preflight import preflight
from src.data.prepare_moments import stage


def prepare(selection,journals,directory,media,output,recoveries=None):
    selection=Path(selection);journals=[Path(p) for p in journals];output=Path(output)
    if output.exists():raise ValueError('Fresh research preparation required')
    recoveries=Path(recoveries) if recoveries is not None else None
    inputs=[selection,*journals,*([recoveries] if recoveries else [])]
    contents={str(path):path.read_bytes() for path in inputs}
    parsed=[]
    for path in journals:
        data=contents[str(path)]
        if data and not data.endswith(b'\n'):raise ValueError('Incomplete receipt journal tail')
        parsed.append([json.loads(line) for line in data.splitlines() if line.strip()])
    recovery_rows=None
    if recoveries is not None:
        data=contents[str(recoveries)]
        if data and not data.endswith(b'\n'):raise ValueError('Incomplete recovery journal tail')
        recovery_rows=[json.loads(line) for line in data.splitlines() if line.strip()]
    rows,audit=reconcile(json.loads(contents[str(selection)]),parsed,recovery_rows)
    audit['input_sha256']={path:hashlib.sha256(data).hexdigest() for path,data in contents.items()}
    # No output exists before every selected candidate has a terminal receipt.
    output.mkdir(parents=True)
    receipts=output/'resolved-receipts.jsonl'
    receipts.write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
    (output/'source-accounting.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    decoded=preflight(receipts,directory,media,output/'decoder')
    if decoded['total']!=len({r['sha256'] for r in rows if r['status']=='complete'}):
        raise ValueError('Incomplete decoder inspection')
    result=stage(output/'prepared',directory=directory,receipts=receipts,selection=selection,
                 decoder_preflight=output/'decoder'/'preflight.json')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',required=True);p.add_argument('--receipts',nargs='+',required=True)
    p.add_argument('--directory',default='data/external/didemo');p.add_argument('--media',default='data/raw/didemo')
    p.add_argument('--output',required=True)
    p.add_argument('--recoveries',help='Complete source-only extensionless ingestion recovery journal')
    a=p.parse_args();prepare(a.selection,a.receipts,a.directory,a.media,a.output,a.recoveries)


if __name__=='__main__':main()
