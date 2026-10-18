import json
import pytest
from src.data.prepare_research import prepare


def test_incomplete_downloads_cannot_create_frozen_preparation(tmp_path):
    selection=tmp_path/'selection.json';journal=tmp_path/'receipts.jsonl';output=tmp_path/'output'
    selection.write_text(json.dumps({'splits':{'train':[{'video':'missing.mov'}],'val':[],'test':[]}}))
    journal.write_text('')
    with pytest.raises(ValueError,match='missing'):
        prepare(selection,[journal],tmp_path/'metadata',tmp_path/'media',output)
    assert not output.exists()


def test_interrupted_journal_cannot_be_mistaken_for_finished_accounting(tmp_path):
    selection=tmp_path/'selection.json';journal=tmp_path/'receipts.jsonl';output=tmp_path/'output'
    selection.write_text(json.dumps({'splits':{'train':[],'val':[],'test':[]}}))
    journal.write_bytes(b'{"status":"complete"')
    with pytest.raises(ValueError,match='journal tail'):
        prepare(selection,[journal],tmp_path/'metadata',tmp_path/'media',output)
    assert not output.exists()
