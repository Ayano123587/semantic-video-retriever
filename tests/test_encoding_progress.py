"""Checkpoint integrity fixtures; these are not model-performance measurements."""
import hashlib
import json
import numpy as np
import pytest
from src.models.encoding_progress import EncodingProgress
from src.models.moment_encoder import frame_schedule


def fixture(tmp_path):
    records=[{'video':f'{i}.mov','sha256':str(i)*64,'split':s,'num_segments':1}
             for i,s in enumerate(('train','val'))]
    values=np.zeros((2,512),dtype=np.float32);values[:,0]=1
    receipt={'video':'0.mov','source_sha256':'0'*64,'split':'train',
             'duration':5.,'fps':30.,'samples':frame_schedule(1,5.,30.)}
    identities={'source':'fixed-source','encoder':'fixed-weights','algorithm':'fixed-code'}
    progress=EncodingProgress(tmp_path/'first',identities,records)
    progress.append(0,values,receipt)
    return progress,identities,records,values


def test_resume_reuses_exact_complete_arrays_and_appends_remaining_source(tmp_path):
    first,identities,records,values=fixture(tmp_path)
    resumed=EncodingProgress(tmp_path/'next',identities,records,first.directory)
    actual,receipt=resumed.get(0)
    np.testing.assert_array_equal(actual,values)
    assert resumed.get(1) is None
    receipt={**receipt,'video':'1.mov','source_sha256':'1'*64,'split':'val'}
    resumed.append(1,values,receipt)
    assert len(resumed.entries)==2 and len(first.entries)==1


def test_resume_refuses_source_model_or_algorithm_identity_change(tmp_path):
    first,identities,records,_=fixture(tmp_path)
    with pytest.raises(ValueError,match='identity changed'):
        EncodingProgress(tmp_path/'next',{**identities,'encoder':'other'},records,first.directory)


@pytest.mark.parametrize('mutation',['array','index','source','schedule'])
def test_resume_refuses_corrupt_array_or_changed_complete_entry(tmp_path,mutation):
    first,identities,records,_=fixture(tmp_path)
    path=first.directory/'complete.jsonl'
    entry=json.loads(path.read_text())
    if mutation=='array':
        (first.directory/entry['file']).write_bytes(b'corrupt')
    elif mutation=='index':entry['index']=1
    elif mutation=='source':entry['record']['sha256']='f'*64
    else:entry['frame_receipt']['fps']=15.
    path.write_text(json.dumps(entry)+'\n')
    with pytest.raises(ValueError):EncodingProgress(tmp_path/'next',identities,records,first.directory)


def test_interrupted_tail_and_orphan_array_are_preserved_but_not_treated_as_complete(tmp_path):
    first,identities,records,values=fixture(tmp_path)
    with (first.directory/'complete.jsonl').open('ab') as handle:handle.write(b'{"index":1')
    np.save(first.directory/'00000001.npy',values,allow_pickle=False)
    resumed=EncodingProgress(tmp_path/'next',identities,records,first.directory)
    assert resumed.interrupted_tail_bytes==10 and len(resumed.entries)==1
    assert resumed.get(1) is None and not (resumed.directory/'00000001.npy').exists()
    assert (resumed.directory/'interrupted-tail.part').read_bytes()==b'{"index":1'


def test_pickle_array_is_not_loaded_even_with_recomputed_transport_hash(tmp_path):
    first,identities,records,_=fixture(tmp_path)
    path=first.directory/'00000000.npy'
    np.save(path,np.array([{'untrusted':'object'}],dtype=object),allow_pickle=True)
    journal=first.directory/'complete.jsonl';entry=json.loads(journal.read_text())
    entry['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    journal.write_text(json.dumps(entry)+'\n')
    with pytest.raises(ValueError):EncodingProgress(tmp_path/'next',identities,records,first.directory)
