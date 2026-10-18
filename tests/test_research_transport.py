"""Audit transport mechanics; fixtures are not scientific datasets."""
import hashlib
import json
import pytest
from scripts.stage_cloud_source import research_inputs


def fixture(tmp_path):
    proof=tmp_path/'data/processed/research';prepared=proof/'prepared';prepared.mkdir(parents=True)
    original=tmp_path/'data/interim/original.jsonl';original.parent.mkdir();original.write_bytes(b'{}\n')
    audit={'input_sha256':{'data/interim/original.jsonl':hashlib.sha256(original.read_bytes()).hexdigest()}}
    (proof/'source-accounting.json').write_text(json.dumps(audit))
    (proof/'decoder').mkdir();(proof/'decoder/preflight.json').write_text('{}')
    return proof,prepared,original,audit


def test_transport_keeps_recorded_paths_and_excludes_test_annotations(tmp_path):
    proof,prepared,_,_=fixture(tmp_path)
    paths=research_inputs(tmp_path,prepared,proof)
    destinations=[p for _,p in paths]
    assert 'data/interim/original.jsonl' in destinations
    assert 'data/external/didemo/train_data.json' in destinations
    assert 'data/external/didemo/val_data.json' in destinations
    assert not any('test_data.json' in p or p.endswith('/test.json') for p in destinations)


def test_transport_rejects_changed_original_before_packaging(tmp_path):
    proof,prepared,original,_=fixture(tmp_path);original.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):research_inputs(tmp_path,prepared,proof)


@pytest.mark.parametrize('name',['../outside.json','C:/private/keys.json','other/keys.json'])
def test_transport_cannot_follow_untrusted_audit_paths(tmp_path,name):
    proof,prepared,_,audit=fixture(tmp_path)
    audit['input_sha256']={name:'0'*64};(proof/'source-accounting.json').write_text(json.dumps(audit))
    with pytest.raises(ValueError,match='Portable'):research_inputs(tmp_path,prepared,proof)


def test_transport_carries_fixed_optional_recovery_evidence(tmp_path,monkeypatch):
    from src.data import decoder_recovery
    proof,prepared,_,_=fixture(tmp_path)
    (proof/'decoder/preflight.json').write_text(json.dumps({'recovery':{'fixture':True}}))
    (proof/'resolved-receipts.jsonl').write_bytes(b'{}\n')
    evidence=proof/'decoder/recovery';evidence.mkdir()
    artifacts=[]
    for name in decoder_recovery.FILES.values():
        p=evidence/name;p.write_bytes(b'fixture');artifacts.append(p)
    calls=[]
    def checked(*args):calls.append(args);return artifacts
    monkeypatch.setattr(decoder_recovery,'checked_recovery',checked)
    paths=research_inputs(tmp_path,prepared,proof)
    assert len(calls)==1
    assert {str(p.relative_to(tmp_path)).replace('\\','/') for p in artifacts}.issubset({name for _,name in paths})
