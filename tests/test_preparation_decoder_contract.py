import hashlib
import json
import pytest
from src.data import prepare_moments


def fixture(tmp_path,monkeypatch):
    metadata=tmp_path/'metadata';metadata.mkdir()
    selection=tmp_path/'selection.json';receipts=tmp_path/'receipts.jsonl';decoder=tmp_path/'decoder.json'
    hashes={};plan={'splits':{}};rows=[]
    for split in ('train','val','test'):
        name=split+'.mov';plan['splits'][split]=[{'video':name}]
        content=json.dumps([{'video':name,'num_segments':6}]).encode()
        (metadata/f'{split}_data.json').write_bytes(content)
        hashes[f'{split}_data.json']=hashlib.sha256(content).hexdigest()
        rows.append({'video':name,'split':split,'status':'complete','sha256':hashlib.sha256(split.encode()).hexdigest()})
    monkeypatch.setattr(prepare_moments,'HASHES',hashes)
    selection.write_text(json.dumps(plan));receipts.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report={'receipt_sha256':hashlib.sha256(receipts.read_bytes()).hexdigest(),
            'rows':[{**r,'num_segments':6,'duration':30.,'fps':25.} for r in rows]}
    return metadata,selection,receipts,decoder,report


@pytest.mark.parametrize('defect',['receipt','geometry'])
def test_decoder_receipt_and_annotation_identity_are_bound_before_staging(tmp_path,monkeypatch,defect):
    metadata,selection,receipts,decoder,report=fixture(tmp_path,monkeypatch)
    if defect=='receipt':report['receipt_sha256']='0'*64
    else:report['rows'][0]['num_segments']=5
    decoder.write_text(json.dumps(report));output=tmp_path/'prepared'
    with pytest.raises(ValueError,match='Decoder .* mismatch'):
        prepare_moments.stage(output,directory=metadata,receipts=receipts,selection=selection,decoder_preflight=decoder)
    assert not output.exists()
