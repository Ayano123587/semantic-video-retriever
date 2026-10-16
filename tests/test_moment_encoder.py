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
