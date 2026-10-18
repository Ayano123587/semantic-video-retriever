import hashlib
import numpy as np
import pytest
from src.data.decoder_preflight import inspect


class Reader:
    def __init__(self,broken=False):self.closed=False;self.broken=broken;self.requested=[]
    def get_meta_data(self):return {'duration':30.,'fps':25.}
    def get_data(self,index):
        self.requested.append(index)
        if self.broken:raise IndexError('missing terminal frame')
        return np.zeros((2,3,3),dtype=np.uint8)
    def close(self):self.closed=True


def fixture(tmp_path):
    path=tmp_path/'source.mov';path.write_bytes(b'fixture-not-real-media')
    return path,{'split':'train','video':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'num_segments':6}


def test_inspects_last_scheduled_frame_and_closes_reader(tmp_path):
    path,record=fixture(tmp_path);reader=Reader()
    row=inspect(record,path,lambda _:reader)
    assert row['status']=='valid' and row['last_frame_decoded']
    # floor((25 + .75 * 5) * 25) = floor(718.75)
    assert reader.closed and reader.requested==[718]


def test_missing_terminal_frame_rejects_whole_source(tmp_path):
    path,record=fixture(tmp_path);reader=Reader(broken=True)
    row=inspect(record,path,lambda _:reader)
    assert row['status']=='invalid' and row['duration']==0 and reader.closed


def test_changed_source_stops_before_decoder(tmp_path):
    path,record=fixture(tmp_path);path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash mismatch'):
        inspect(record,path,lambda _:pytest.fail('decoder must not run'))
