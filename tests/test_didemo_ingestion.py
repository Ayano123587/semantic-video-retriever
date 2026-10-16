"""Ingestion fixtures check filenames and cohort coverage, not model quality."""
import io
import json
import pytest
from src.data import didemo


@pytest.mark.parametrize('extension',['mp4','mov','avi','mpg','3gp','3g2','wmv','m4v'])
def test_official_original_extension_does_not_block_transcoded_media(tmp_path,monkeypatch,extension):
    body=b'small transport fixture; decoder validation is separate'
    monkeypatch.setattr(didemo,'urlopen',lambda *a,**k:io.BytesIO(body))
    record={'video':f'40774255@N07_5898882016_4af574f40b.{extension}',
            'aws_url':'https://example.invalid/transcoded.mp4','license':'fixture','original_url':'fixture'}
    receipt=didemo.download_one(record,tmp_path)
    assert (tmp_path/record['video']).read_bytes()==body
    assert receipt['bytes']==len(body)


@pytest.mark.parametrize('name',['../bad.mov','sub/bad.avi','bad\\name.mpg','bad..mov','bad.mov/other','bad.mov:stream'])
def test_unsafe_basename_fails_before_network(tmp_path,monkeypatch,name):
    def forbidden(*a,**k):raise AssertionError('Unsafe path must not reach network')
    monkeypatch.setattr(didemo,'urlopen',forbidden)
    with pytest.raises(ValueError,match='filename'):
        didemo.download_one({'video':name},tmp_path)


def test_default_selection_retains_every_eligible_video(tmp_path,monkeypatch):
    monkeypatch.setattr(didemo,'fetch_metadata',lambda directory:{'fixture':True})
    (tmp_path/'video_licenses.txt').write_text('id url license\n'+'\n'.join(
        f'{i} https://example.invalid/{i} http://creativecommons.org/licenses/by/2.0/' for i in range(6)))
    (tmp_path/'yfcc100m_hash.txt').write_text('\n'.join(f'{i} {i:032x}' for i in range(6)))
    for split,ids in [('train',[0,1,2]),('val',[3,4]),('test',[5])]:
        (tmp_path/f'{split}_data.json').write_text(json.dumps([{'video':f'user_{i}_hash.mov'} for i in ids]))
    selected=didemo.selection(tmp_path,tmp_path/'plan.json')
    assert [len(selected['splits'][s]) for s in ('train','val','test')]==[3,2,1]
    assert selected['sampling_cap'] is None
    assert all(x['selected_videos']==x['license_eligible_videos'] for x in selected['coverage'].values())
