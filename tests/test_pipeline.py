import numpy as np
import pytest
from src.features.build_features import frame_schedule
from src.models.index import VectorIndex,deduplicate
from src.models.predict_model import search

def records():return [{'timestamp_seconds':t,'video_sha256':'a','scene_id':0} for t in [0.,1.,8.]]
def test_exact_frame_timestamps():
    assert frame_schedule(24.,5.,2.)==[(0,0.),(48,2.),(96,4.)]
    assert len(frame_schedule(2.,1.,.1))==2
    with pytest.raises(ValueError):frame_schedule(0,10,2)

def test_index_roundtrip_and_integrity(tmp_path):
    index=VectorIndex([[1,0],[.9,.1],[0,1]],records());index.save(tmp_path)
    loaded=VectorIndex.load(tmp_path);assert np.allclose(index.vectors,loaded.vectors)
    assert loaded.query([1,0],3,4)[0]['timestamp_seconds']==0
    assert len(loaded.query([1,0],3,4))==2
    with (tmp_path/'vectors.npy').open('ab') as f:f.write(b'bad')
    with pytest.raises(ValueError,match='integrity'):VectorIndex.load(tmp_path)

def test_invalid_shape_and_vectors():
    with pytest.raises(ValueError):VectorIndex([[0,0]],records()[:1])
    with pytest.raises(ValueError):VectorIndex([[1,0]],records())
    with pytest.raises(ValueError):VectorIndex([[float('nan'),0]],records()[:1])
    with pytest.raises(ValueError):VectorIndex([[1,0]],records()[:1]).query([1,2,3])

def test_scene_aware_dedup():
    v,r=deduplicate([[1,0],[1,0],[0,1]],records());assert len(r)==2
    rs=records();rs[1]['scene_id']=1
    assert len(deduplicate([[1,0],[1,0],[0,1]],rs)[1])==3

def test_empty_query_fails_before_model_load():
    with pytest.raises(ValueError,match='empty'):search(None,'   ')

def test_model_revision_guard():
    index=VectorIndex([[1,0]],records()[:1],{'model_id':'wrong'})
    with pytest.raises(ValueError,match='revision'):search(index,'bird')


def test_actual_video_sampling(tmp_path):
    import imageio.v2 as imageio
    from src.features.build_features import sample
    video=tmp_path/'small.mp4'
    with imageio.get_writer(video,fps=4,macro_block_size=1) as writer:
        for i in range(8):writer.append_data(np.full((32,32,3),i*20,dtype=np.uint8))
    frames=list(sample(video,interval=.5))
    assert [r['frame_number'] for _,r in frames]==[0,2,4,6]
    assert [r['timestamp_seconds'] for _,r in frames]==[0.,.5,1.,1.5]


def test_evaluation_requires_video_and_interval_match():
    from src.models.experiment import relevant,metrics
    query={'video':'a.mp4','relevant_intervals_seconds':[[4,8]]}
    assert relevant({'video':'a.mp4','timestamp_seconds':5},query)
    assert not relevant({'video':'b.mp4','timestamp_seconds':5},query)
    assert metrics([[{'video':'a.mp4','timestamp_seconds':0},{'video':'a.mp4','timestamp_seconds':5}]],[query])=={'hit_at_1':0.,'hit_at_3':1.,'mrr_at_3':.5}
