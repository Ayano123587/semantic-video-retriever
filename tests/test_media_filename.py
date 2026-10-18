import hashlib
import pytest
from src.data.didemo import media_filename,decoder_source


def test_extensionless_official_identity_has_stable_windows_safe_storage():
    identity='54597574@N00_7376481506_c0a1789f18.'
    assert media_filename(identity)=='source-'+hashlib.sha256(identity.encode()).hexdigest()+'.mp4'
    assert media_filename(identity)!=media_filename(identity+'mp4')


@pytest.mark.parametrize('value',['../escape.mp4','folder/clip.mp4','x:stream.mp4','..','x.mp4.','x\\y.mp4'])
def test_storage_mapping_still_rejects_unsafe_identities(value):
    with pytest.raises(ValueError):media_filename(value)


def test_decoder_uses_same_bytes_and_cleans_only_its_owned_view(tmp_path):
    original=tmp_path/'original.3gp';original.write_bytes(b'unchanged source bytes')
    with decoder_source(original) as compatible:
        assert compatible.suffix=='.mp4' and compatible.read_bytes()==original.read_bytes()
        directory=compatible.parent
    assert original.read_bytes()==b'unchanged source bytes' and not directory.exists()


def test_decoder_view_cleanup_runs_when_reader_fails(tmp_path):
    original=tmp_path/'original.mov';original.write_bytes(b'preserve me')
    with pytest.raises(RuntimeError):
        with decoder_source(original) as compatible:
            directory=compatible.parent
            raise RuntimeError('reader failure')
    assert original.exists() and not directory.exists()
