"""Require complete source accounting before preparing a research cohort."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def validate_receipt(row):
    if row.get('status') not in ('complete','failed'):raise ValueError('Invalid receipt status')
    if row['status']=='complete':
        if not re.fullmatch('[0-9a-f]{64}',row.get('sha256','')):raise ValueError('Invalid media hash')
        if type(row.get('bytes')) is not int or row['bytes']<=0:raise ValueError('Invalid media size')


def reconcile(plan, journals, recoveries=None):
    permitted={(split,row['video']) for split,rows in plan['splits'].items() for row in rows}
    if sum(map(len,plan['splits'].values()))!=len(permitted):
        raise ValueError('Duplicate selection identity')
    records={}
    for journal in journals:
        for row in journal:
            key=(row['split'],row['video'])
            if key not in permitted:raise ValueError('Receipt outside selection')
            validate_receipt(row)
            if key in records:raise ValueError('Duplicate receipt identity; resolve retries explicitly')
            records[key]=row
    missing=permitted-records.keys()
    if missing:raise ValueError(f'Incomplete source accounting: {len(missing)} candidates missing')
    recovery_keys=set()
    if recoveries is not None:
        eligible={key for key,row in records.items() if key[1].endswith('.') and
                  row['status']=='failed' and row.get('reason')=='Invalid source video filename'}
        for row in recoveries:
            validate_receipt(row);key=(row['split'],row['video'])
            if key in recovery_keys:raise ValueError('Duplicate recovery identity')
            if key not in eligible:raise ValueError('Recovery cannot replace an unrelated or successful source attempt')
            recovery_keys.add(key)
            records[key]={**row,'recovered_ingestion_failure':dict(records[key])}
        if recovery_keys!=eligible:raise ValueError('Incomplete extensionless source recovery accounting')
    groups={}
    for key,row in records.items():
        if row['status']=='complete':groups.setdefault(row['sha256'],[]).append(key)
    priority={'train':0,'val':1,'test':2}
    excluded=set()
    for group in groups.values():
        winning=max(priority[key[0]] for key in group)
        excluded.update(key for key in group if priority[key[0]]<winning)
    resolved=[]
    for key,row in sorted(records.items()):
        row=dict(row)
        if key in excluded:
            row.update(status='failed',reason='exact-media cross-split duplicate; retain test then validation then training')
        resolved.append(row)
    counts={split:{status:sum(r['split']==split and r['status']==status for r in resolved)
                   for status in ('complete','failed')} for split in ('train','val','test')}
    return resolved,{'candidate_count':len(permitted),'counts':counts,
                     'recovery_attempts':len(recovery_keys),
                     'cross_split_excluded_ids':[list(key) for key in sorted(excluded)],
                     'duplicate_policy':'retain test > validation > training; within-split aliases retain all queries'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',required=True)
    p.add_argument('--receipts',nargs='+',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    paths=[Path(a.selection),*[Path(v) for v in a.receipts]]
    journals=[]
    for path in paths[1:]:
        content=path.read_bytes()
        if content and not content.endswith(b'\n'):raise ValueError('Incomplete receipt journal tail')
        journals.append([json.loads(line) for line in content.splitlines() if line.strip()])
    rows,audit=reconcile(json.loads(paths[0].read_text(encoding='utf-8')),journals)
    audit['input_sha256']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    output=Path(a.output)
    if output.exists():raise ValueError('Refuse to overwrite frozen receipt reconciliation')
    output.mkdir(parents=True)
    (output/'receipts.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
    (output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
