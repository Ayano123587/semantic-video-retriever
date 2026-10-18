import numpy as np
import pytest
from src.models.moments import moments,positive_mask
from src.models.positive_index import PositiveIndex


def test_indexed_labels_equal_original_oracle_for_every_geometry_and_reference_pair():
    candidates=[{'video':str(n),'moment':list(m)} for n in range(1,7) for m in moments(n)]
    queries=[{'video':str(n),'times':[list(a),list(b),list(a),list(b)]}
             for n in range(1,7) for a in moments(n) for b in moments(n)]
    index=PositiveIndex(candidates,queries)
    for start in range(0,len(queries),17):
        ids=list(range(start,min(start+17,len(queries))))
        expected=np.stack([positive_mask(candidates,queries[i]) for i in ids])
        np.testing.assert_array_equal(index.mask(ids),expected)
        np.testing.assert_array_equal(index.valid[ids],expected.any(axis=1))
        expected_local=np.asarray([[queries[i]['video']==candidate['video'] for candidate in candidates] for i in ids])
        np.testing.assert_array_equal(index.same_video_mask(ids),expected_local)


def test_preserves_empty_positive_queries_and_batch_order():
    candidates=[{'video':'v','moment':list(m)} for m in moments(6)]
    queries=[{'video':'v','times':[[0,0],[2,2],[5,5]]},
             {'video':'v','times':[[0,0],[0,0],[0,0]]}]
    index=PositiveIndex(candidates,queries)
    assert not index.valid[0] and index.valid[1]
    np.testing.assert_array_equal(index.mask([1,0,1]),np.stack([positive_mask(candidates,queries[i]) for i in (1,0,1)]))
    with pytest.raises(ValueError):index.mask([-1])


def test_rejects_duplicate_geometry_and_missing_reference():
    c=[{'video':'v','moment':[0,0]}]
    q=[{'video':'v','times':[[0,0]]}]
    with pytest.raises(ValueError,match='Duplicate'):PositiveIndex(c+c,q)
    with pytest.raises(ValueError,match='reference'):PositiveIndex(c,[{'video':'v','times':[[1,1]]}])
