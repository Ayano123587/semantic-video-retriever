"""Small software fixtures verify routing; they are not benchmark evidence."""
import json
import numpy as np
import pytest
import torch
from src.config import MODEL_ID,MODEL_REVISION
from src.models.encoder import WEIGHTS_SHA256
from src.models.moment_training import Adapter,digest
from src.models.moment_search import MomentSearch


def fixture(tmp_path,adapted=False):
    values=np.zeros((2,512),dtype=np.float32);values[0,0]=1;values[1,1]=1
    np.save(tmp_path/'val-moments.npy',values,allow_pickle=False)
    (tmp_path/'val-candidates.json').write_text(json.dumps([{'video':'v','moment':[0,0]},{'video':'v','moment':[1,1]}]))
    (tmp_path/'frames.json').write_text(json.dumps([{'video':'v','split':'val','duration':7.25}]))
    receipt={'model_id':MODEL_ID,'revision':MODEL_REVISION,'weights_sha256':WEIGHTS_SHA256,'loading_info':{}}
    manifest={'encoder':receipt,'files':{name:digest(tmp_path/name) for name in ('val-moments.npy','val-candidates.json','frames.json')}}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    states=[]
    if adapted:
        model=Adapter()
        with torch.no_grad():model.up.bias[1]=2
        states=[model.state_dict()]*3
    checkpoint=tmp_path/'selected.pt'
    torch.save({'schema':1,'states':states,'adapted_score_weight':1. if adapted else 0.,'encoder':receipt,
                'cache_manifest_sha256':digest(tmp_path/'manifest.json')},checkpoint)
    return checkpoint,digest(checkpoint),values[0]


def test_baseline_and_adapter_route_change_ranking_with_clamped_real_duration(tmp_path):
    checkpoint,sha,query=fixture(tmp_path)
    baseline=MomentSearch(tmp_path,checkpoint,sha).search_vector(query)
    assert baseline[0]['start_seconds']==0 and baseline[0]['score']==1
    checkpoint,sha,query=fixture(tmp_path,True)
    selected=MomentSearch(tmp_path,checkpoint,sha).search_vector(query)
    assert selected[0]['start_seconds']==5 and selected[0]['end_seconds']==7.25
    assert selected[0]['score']>selected[1]['score']


@pytest.mark.parametrize('defect',['checkpoint_hash','feature_hash','cache_identity'])
def test_rejects_changed_artifacts_before_allocating_encoder(tmp_path,defect):
    checkpoint,sha,_=fixture(tmp_path)
    if defect=='checkpoint_hash':sha='0'*64
    elif defect=='feature_hash':(tmp_path/'val-moments.npy').write_bytes(b'changed')
    else:(tmp_path/'manifest.json').write_text((tmp_path/'manifest.json').read_text()+' ')
    with pytest.raises(ValueError):MomentSearch(tmp_path,checkpoint,sha)


def test_rejects_unknown_query_encoder_and_invalid_query_vector(tmp_path):
    checkpoint,sha,query=fixture(tmp_path);index=MomentSearch(tmp_path,checkpoint,sha)
    class WrongEncoder:receipt={}
    with pytest.raises(ValueError,match='identity'):index.search('query',encoder=WrongEncoder())
    with pytest.raises(ValueError):index.search_vector(query*2)
    with pytest.raises(ValueError):index.search_vector(query,top_k=0)
