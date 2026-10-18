"""Actual tiny CPU optimization verifies mechanics, never benchmark quality."""
import copy
import json
import random
from pathlib import Path
import numpy as np
import pytest
import torch
from src.models import resumable_training as r
from src.models.positive_index import PositiveIndex
from src.models.moments import evaluate
from src.models.moment_training import scores_for


def data_fixture():
    generator=torch.Generator().manual_seed(31)
    videos=['a','b'];candidates=[{'video':v,'moment':m} for v in videos for m in ([0,0],[0,1],[1,1])]
    queries=[{'video':videos[i%2],'times':[[i%2,i%2]]} for i in range(7)]
    values=torch.nn.functional.normalize(torch.randn(6,512,generator=generator),dim=1)
    x=torch.nn.functional.normalize(torch.randn(7,512,generator=generator),dim=1)
    item={'x':x,'v':values,'queries':queries,'candidates':candidates,'positive_index':PositiveIndex(candidates,queries)}
    return {'train':item,'val':item}


def seed_everything():
    torch.manual_seed(222);random.seed(333);np.random.seed(444)


def same(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def last_state(output):
    path=sorted(output.glob('checkpoint-*.pt'))[-1]
    return torch.load(path,weights_only=True,map_location='cpu')


def baseline(data):
    item=data['val']
    return evaluate(scores_for([],item['x'],item['v']),item['candidates'],item['queries'])


def tiny_spec(phase):
    spec=r.production_spec(phase);spec.update(maximum_epochs=3,minimum_epochs=1,patience=2,batch=2)
    spec['configs']=spec['configs'][:2]
    return spec


@pytest.mark.parametrize('phase',['phase1','phase2'])
def test_batch_resume_exact_weights_optimizer_rng_history_and_selection(tmp_path,phase):
    data=data_fixture();spec=tiny_spec(phase);base=baseline(data)
    identity={'phase':phase,'cache_manifest_sha256':'cache','protocol_sha256':'protocol','source':'fixed'}
    seed_everything()
    full=r.engine(data,{'encoder':{}},base,tmp_path/'full',identity,spec,checkpoint_every_batches=10000)
    seed_everything();previous=None;previous_sha=None;windows=[]
    for i,limit in enumerate([1,7,13,None]):
        out=tmp_path/f'window{i}'
        prior_hash=r.digest(previous) if previous is not None else None
        result=r.engine(data,{'encoder':{}},base,out,identity,spec,resume=previous,resume_sha256=previous_sha,
                        checkpoint_every_batches=10000,max_batches=limit)
        if previous is not None:assert r.digest(previous)==prior_hash
        windows.append(out)
        if limit is not None:
            assert result['status']=='paused_service_window'
            assert not (out/'results.json').exists() and not (out/'selected.pt').exists()
            previous=out/result['snapshot'];previous_sha=result['snapshot_sha256']
        else:continued=result
    final=windows[-1]
    assert full['completed_fits']==continued['completed_fits']==6
    for key in ['baseline','runs','selected','selected_validation','policies','random_control','chronological_control']:
        if key in full:assert same(full[key],continued[key]),key
    for name in sorted((tmp_path/'full').glob('config*-seed*.pt')):
        a=torch.load(name,weights_only=True);b=torch.load(final/name.name,weights_only=True)
        assert same(a,b),name.name
    a=last_state(tmp_path/'full');b=last_state(final)
    assert same(a['rng'],b['rng'])
    assert same(a['winner_states'],b['winner_states'])
    assert full['checkpoint_replay_max_abs_error']==continued['checkpoint_replay_max_abs_error']==0


@pytest.mark.parametrize('defect',['identity','snapshot_hash','completed_hash','missing_completed','early_done','permutation','permutation_shape','permutation_range','cursor','epoch','history','best_state','stage_seed','optimizer_nonfinite','optimizer_shape','optimizer_config'])
def test_resume_rejects_identity_hash_and_state_corruption(tmp_path,defect):
    data=data_fixture();spec=tiny_spec('phase2');identity={'phase':'phase2','cache_manifest_sha256':'cache','protocol_sha256':'protocol'}
    first=r.engine(data,{'encoder':{}},baseline(data),tmp_path/'first',identity,spec,max_batches=14,checkpoint_every_batches=10000)
    path=tmp_path/'first'/first['snapshot'];sha=first['snapshot_sha256']
    if defect=='identity':identity=dict(identity,protocol_sha256='changed')
    elif defect=='snapshot_hash':sha='0'*64
    elif defect=='completed_hash':(tmp_path/'first/config0-seed42.pt').write_bytes(b'changed')
    elif defect=='missing_completed':(tmp_path/'first/config0-seed42.pt').unlink()
    else:
        state=torch.load(path,weights_only=True)
        if defect=='early_done':state.update(stage='done',grid=0,seed_index=0,completed=[],active=None)
        elif defect=='permutation':state['active']['permutation'].fill_(0)
        elif defect=='permutation_shape':state['active']['permutation']=torch.zeros(2,dtype=torch.int64)
        elif defect=='permutation_range':state['active']['permutation'][0]=999
        elif defect=='cursor':state['active']['cursor']=1
        elif defect=='epoch':state['active']['epoch']=99
        elif defect=='history':state['active']['history']=[{'epoch':8}]
        elif defect=='best_state':state['active']['best_state']={'wrong':torch.zeros(1)}
        elif defect=='stage_seed':state['stage']='candidate'
        elif defect=='optimizer_config':state['active']['optimizer']['param_groups'][0]['lr']=999.
        else:
            first_state=next(iter(state['active']['optimizer']['state'].values()))
            if defect=='optimizer_shape':first_state['exp_avg']=torch.zeros(1)
            else:first_state['exp_avg'].fill_(float('nan'))
        path.write_bytes(b'')
        with path.open('wb') as stream:torch.save(state,stream)
        sha=r.digest(path);receipt=json.loads(path.with_suffix('.json').read_text());receipt['snapshot_sha256']=sha
        path.with_suffix('.json').write_text(json.dumps(receipt))
    with pytest.raises((ValueError,RuntimeError)):r.engine(data,{'encoder':{}},baseline(data),tmp_path/'next',identity,spec,resume=path,resume_sha256=sha)


def test_fresh_outputs_source_binding_and_production_recipe():
    from src.models.research_followup import SOURCES
    assert 'models/resumable_training.py' in SOURCES
    for phase,count in [('phase1',3),('phase2',4)]:
        spec=r.production_spec(phase)
        assert len(spec['configs'])==count and spec['seeds']==[42,1337,2026]
        assert (spec['maximum_epochs'],spec['minimum_epochs'],spec['patience'],spec['batch'],spec['temperature'])==(50,5,8,64,.07)


@pytest.mark.parametrize('phase',['phase1','phase2'])
def test_mechanical_engine_matches_original_production_loops(tmp_path,monkeypatch,phase):
    from src.models import moment_training as first
    from src.models import localization_followup as second
    data=data_fixture();base=baseline(data);manifest={'encoder':{}}
    cache=tmp_path/'cache';cache.mkdir();(cache/'manifest.json').write_text('{}')
    protocol=tmp_path/'protocol.json';protocol.write_text(json.dumps({'baseline':base}))
    identity={'phase':phase,'cache_manifest_sha256':r.digest(cache/'manifest.json'),'protocol_sha256':r.digest(protocol)}
    seed_everything()
    if phase=='phase1':
        monkeypatch.setattr(first,'load_cache',lambda *a,**k:(data,manifest))
        legacy=first.train(cache,tmp_path/'legacy')
    else:
        monkeypatch.setattr(second,'checked_cache',lambda *a,**k:(data,manifest,{'baseline':base}))
        legacy=second.train(cache,tmp_path/'legacy',protocol)
    seed_everything()
    current=r.engine(data,manifest,base,tmp_path/'current',identity,r.production_spec(phase),checkpoint_every_batches=10000)
    for key in ['baseline','runs','selected','selected_validation','policies','random_control','chronological_control']:
        if key in legacy:assert same(legacy[key],current[key]),key
    old=torch.load(tmp_path/'legacy/selected.pt',weights_only=True)
    new=torch.load(tmp_path/'current/selected.pt',weights_only=True)
    assert same(old['states'],new['states']) and old['validation']==new['validation']


def test_best_state_shape_after_validation_is_checked_before_continuing(tmp_path):
    data=data_fixture();spec=tiny_spec('phase1');identity={'phase':'phase1','cache_manifest_sha256':'cache','protocol_sha256':'protocol'}
    stopped=r.engine(data,{'encoder':{}},baseline(data),tmp_path/'first',identity,spec,max_batches=22,checkpoint_every_batches=10000)
    path=tmp_path/'first'/stopped['snapshot'];state=torch.load(path,weights_only=True)
    assert state['active']['history'] and state['active']['best_state'] is not None
    state['active']['best_state']['down.weight']=torch.zeros(1)
    with path.open('wb') as stream:torch.save(state,stream)
    receipt=json.loads(path.with_suffix('.json').read_text());receipt['snapshot_sha256']=r.digest(path)
    path.with_suffix('.json').write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError,match='size mismatch'):
        r.engine(data,{'encoder':{}},baseline(data),tmp_path/'next',identity,spec,resume=path,resume_sha256=r.digest(path))


def test_startup_time_counts_and_can_save_an_empty_verified_window(tmp_path,monkeypatch):
    data=data_fixture();base=baseline(data);spec=tiny_spec('phase1')
    # Public timer began before checked_protocol; no wall-clock sleep required.
    result=r.engine(data,{'encoder':{}},base,tmp_path/'empty',{'cache_manifest_sha256':'cache','protocol_sha256':'protocol'},spec,
                    max_seconds=1.,service_started=r.time.monotonic()-5.)
    assert result['status']=='paused_service_window' and result['batches']==0 and result['completed_seeds']==0
    assert not (tmp_path/'empty/results.json').exists()
    state=torch.load(tmp_path/'empty'/result['snapshot'],weights_only=True)
    assert state['service_seconds']>=5. and state['active'] is None
    # The public wrapper starts the clock before source/cache verification.
    from src.models import research_followup as research
    clock={'now':100.};monkeypatch.setattr(r.time,'monotonic',lambda:clock['now'])
    monkeypatch.setattr(research,'checked_protocol',lambda *args:(data,{'encoder':{}},{'baseline':base,'source_manifest_sha256':'cache','implementation_sha256':{},'coverage':{},'encoding_identity':{}}))
    captured={}
    def engine(*args,**kwargs):captured.update(kwargs);return {'verified':True}
    monkeypatch.setattr(r,'engine',engine)
    assert r.train('prep','cache',tmp_path/'public','protocol','sha','phase1',max_seconds=10.)=={'verified':True}
    assert captured['service_started']==100. and captured['max_seconds']==10.
