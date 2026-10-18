import pytest
from src.data.research_receipts import reconcile


def fixture():
    plan={'splits':{s:[{'video':s+'-a.mp4'},{'video':s+'-b.mp4'}] for s in ('train','val','test')}}
    rows=[{'split':s,'video':r['video'],'status':'complete','bytes':10,'sha256':str(i)*64}
          for i,(s,r) in enumerate((s,r) for s,rs in plan['splits'].items() for r in rs)]
    return plan,rows


def test_all_candidates_accounted_before_preparation():
    plan,rows=fixture()
    with pytest.raises(ValueError,match='missing'):reconcile(plan,[rows[:-1]])
    result,audit=reconcile(plan,[rows[:3],rows[3:]])
    assert len(result)==audit['candidate_count']==6


def test_duplicate_priority_preserves_heldout_media_and_same_split_aliases():
    plan,rows=fixture()
    for i in (0,2,4,5):rows[i]['sha256']='f'*64
    result,audit=reconcile(plan,[rows])
    assert len(audit['cross_split_excluded_ids'])==2
    assert all(r['status']=='complete' for r in result if r['split']=='test')


@pytest.mark.parametrize('defect',['foreign','duplicate','hash','size','status'])
def test_invalid_receipts_are_rejected(defect):
    plan,rows=fixture()
    if defect=='foreign':rows[0]['video']='foreign.mp4'
    elif defect=='duplicate':rows.append(dict(rows[0]))
    elif defect=='hash':rows[0]['sha256']='invalid'
    elif defect=='size':rows[0]['bytes']=True
    else:rows[0]['status']='pending'
    with pytest.raises(ValueError):reconcile(plan,[rows])


def test_failed_attempt_remains_in_total_accounting():
    plan,rows=fixture();rows[-1]={'split':'test','video':'test-b.mp4','status':'failed','error':'404'}
    _,audit=reconcile(plan,[rows])
    assert audit['counts']['test']=={'complete':1,'failed':1}


def recovery_fixture():
    plan={'splits':{'train':[{'video':'official.'}],'val':[],'test':[]}}
    failed={'split':'train','video':'official.','status':'failed','reason':'Invalid source video filename'}
    recovered={**failed,'status':'complete','bytes':12,'sha256':'a'*64}
    recovered.pop('reason')
    return plan,failed,recovered


def test_recovery_preserves_original_attempt_and_restores_source():
    plan,failed,recovered=recovery_fixture()
    rows,audit=reconcile(plan,[[failed]],[recovered])
    assert rows[0]['status']=='complete' and rows[0]['recovered_ingestion_failure']==failed
    assert failed['status']=='failed' and audit['recovery_attempts']==1


@pytest.mark.parametrize('defect',['foreign','duplicate','success','unrelated','missing','invalid_hash'])
def test_recovery_cannot_change_unrelated_sources_or_hide_incomplete_attempts(defect):
    plan,failed,recovered=recovery_fixture();recoveries=[recovered]
    if defect=='foreign':recovered['video']='foreign.'
    elif defect=='duplicate':recoveries.append(dict(recovered))
    elif defect=='success':failed={**recovered}
    elif defect=='unrelated':failed['reason']='HTTP Error 404: Not Found'
    elif defect=='missing':recoveries=[]
    else:recovered['sha256']='invalid'
    with pytest.raises(ValueError):reconcile(plan,[[failed]],recoveries)
