import pytest
from src.models.moment_encoder import frame_schedule

def test_real_geometry_schedule_and_partial_last_bin():
    result=frame_schedule(6,27.,30.)
    assert len(result)==12
    assert result[0]=={'segment':0,'frame':37,'seconds':37/30}
    assert result[-2]['seconds']==25.5
    assert result[-1]['seconds']==26.5
    assert all(0<=r['seconds']<27 for r in result)

@pytest.mark.parametrize('duration,fps',[(25.,30.),(0.,30.),(27.,0.),(float('inf'),30.),(27.,float('nan'))])
def test_bad_media_geometry_fails(duration,fps):
    with pytest.raises(ValueError):frame_schedule(6,duration,fps)


def test_streaming_source_is_verified_and_cleanup_preserves_existing_media(tmp_path,monkeypatch):
    import hashlib
    from src.models import moment_encoder as module
    body=b'transport fixture, not a semantic benchmark'
    original=tmp_path/'existing.mov';original.write_bytes(b'user-owned media')
    record={'video':'source.mov','sha256':hashlib.sha256(body).hexdigest()}
    created=[]
    def fetch(row,directory):
        path=module.Path(directory)/row['video'];path.write_bytes(body);created.append(path)
        return {'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    monkeypatch.setattr(module,'download_one',fetch)
    for _ in range(3):
        with module.verified_media(record,tmp_path,True) as path:
            assert path.read_bytes()==body and len(list(tmp_path.iterdir()))==2
        assert not path.exists() and original.read_bytes()==b'user-owned media'
    assert len(set(created))==3 and list(tmp_path.iterdir())==[original]


def test_streamed_hash_failure_never_yields_and_cleans_only_owned_directory(tmp_path,monkeypatch):
    from src.models import moment_encoder as module
    original=tmp_path/'keep.txt';original.write_bytes(b'keep')
    def changed(row,directory):
        (module.Path(directory)/row['video']).write_bytes(b'changed source')
        return {'sha256':'0'*64}
    monkeypatch.setattr(module,'download_one',changed)
    with pytest.raises(ValueError,match='differs from frozen'):
        with module.verified_media({'video':'source.mov','sha256':'f'*64},tmp_path,True):
            raise AssertionError('Changed source must not reach decoder')
    assert list(tmp_path.iterdir())==[original]


def test_decoder_failure_still_cleans_owned_streamed_source(tmp_path,monkeypatch):
    from src.models import moment_encoder as module
    def fetch(row,directory):
        (module.Path(directory)/row['video']).write_bytes(b'fixture')
        return {'sha256':row['sha256']}
    monkeypatch.setattr(module,'download_one',fetch)
    with pytest.raises(RuntimeError,match='decoder failed'):
        with module.verified_media({'video':'source.avi','sha256':'a'*64},tmp_path,True):
            raise RuntimeError('decoder failed')
    assert list(tmp_path.iterdir())==[]
