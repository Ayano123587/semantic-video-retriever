"""Reserve one persistent local cloud dispatch before pushing a release kernel.

This writes a receipt only; it does not upload or launch. Preserve it on failure
or ambiguous push/status. Never remove it to retry; review the original kernel
and recover its report/marker instead. This is a consistency guard, not security.
"""
import argparse
import datetime
import json
import re
from pathlib import Path
from src.models.moment_evaluation import MARKER, POLICY
from src.models.moment_training import digest

DISPATCH = 'test-dispatch-receipt.json'


def reserve(source, dataset_id, kernel_id, manifest_sha256):
    source=Path(source)
    if (source/MARKER).exists():raise ValueError('Test release already attempted')
    if not re.fullmatch(r'[0-9a-f]{64}',manifest_sha256):raise ValueError('Invalid stage SHA256')
    for value in (dataset_id,kernel_id):
        if not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_-]+',value):raise ValueError('Expected account/slug identity')
    policy=json.loads((source/POLICY).read_text(encoding='utf-8'))
    if policy.get('schema')!=1 or policy.get('test_targets_used') is not False:raise ValueError('Invalid frozen policy')
    receipt={'schema':1,'status':'dispatch reserved before push; preserve on ambiguity or failure',
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'dataset_id':dataset_id,'kernel_id':kernel_id,'staging_manifest_sha256':manifest_sha256,
        'policy_sha256':digest(source/POLICY),'identities':policy,
        'automatic_retry_allowed':False,'test_gold_parsed_by_reservation':False}
    # Exclusive persistent LOCAL source-directory receipt; do not include in cloud inputs.
    with (source/DISPATCH).open('x',encoding='utf-8',newline='\n') as handle:
        json.dump(receipt,handle,indent=2,allow_nan=False);handle.write('\n')
    print(json.dumps({'reserved':str(source/DISPATCH),'dataset_id':dataset_id,'kernel_id':kernel_id}))
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','dataset-id','kernel-id','manifest-sha256']:p.add_argument('--'+name,required=True)
    a=p.parse_args();reserve(a.source,a.dataset_id,a.kernel_id,a.manifest_sha256)
