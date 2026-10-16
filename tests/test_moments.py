import itertools
import numpy as np
import pytest
from src.models.moments import moments,iou,consensus_iou,evaluate,positive_mask

def test_inclusive_iou_matches_set_oracle():
    spans=moments(6)
    assert len(spans)==21
    for a,b in itertools.product(spans,repeat=2):
        left=set(range(a[0],a[1]+1));right=set(range(b[0],b[1]+1))
        assert iou(a,b)==len(left&right)/len(left|right)

def test_best_three_consensus_and_official_rank():
    c=[{'video':'a','moment':m} for m in moments(2)]
    q=[{'video':'a','times':[[0,0],[0,0],[0,1],[1,1]]}]
    result=evaluate([[3.,2.,1.]],c,q)
    assert result['within_video_official_miou']==pytest.approx(5/6)
    assert result['within_video_official_rank1']==0  # mean best ranks = 4/3
    assert result['corpus_consensus_r1_iou50']==1

def test_wrong_video_cannot_be_temporal_positive():
    c=[{'video':'wrong','moment':(0,0)},{'video':'a','moment':(0,0)}]
    q={'video':'a','times':[[0,0]]*4}
    assert positive_mask(c,q).tolist()==[False,True]
    out=evaluate([[2.,1.]],c,[q])
    assert out['corpus_consensus_r1_iou50']==0 and out['within_video_official_rank1']==1

@pytest.mark.parametrize('bad',[[float('nan')],[float('inf')]])
def test_nonfinite_fails_closed(bad):
    with pytest.raises(ValueError):evaluate([bad],[{'video':'a','moment':(0,0)}],[{'video':'a','times':[[0,0]]}])

def test_missing_geometry_and_duplicate_candidates_fail():
    q=[{'video':'a','times':[[1,1]]}];c=[{'video':'a','moment':(0,0)}]
    with pytest.raises(ValueError):evaluate([[1.]],c,q)
    with pytest.raises(ValueError):evaluate([[1.,1.]],c*2,q)

def test_stable_tie_uses_candidate_order_not_gold():
    c=[{'video':'a','moment':(0,0)},{'video':'a','moment':(1,1)}]
    q=[{'video':'a','times':[[1,1]]*4}]
    assert evaluate([[0.,0.]],c,q)['corpus_consensus_r1_iou50']==0
