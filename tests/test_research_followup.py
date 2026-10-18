"""Tiny contract fixtures are software tests, never retrieval benchmarks."""
import json
from pathlib import Path
from unittest.mock import Mock
import numpy as np
import pytest
import torch
from src.models import research_followup as r
from src.models import localization_followup as legacy
from src.data.research_receipts import reconcile


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')
    return path


def lines(path,rows):
    path=Path(path);path.write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8',newline='\n')
    return path


def preparation(tmp_path,monkeypatch,recovery=False):
    monkeypatch.chdir(tmp_path)
    (tmp_path/'data').mkdir(exist_ok=True)
    root=tmp_path/'preparation';root.mkdir();metadata=tmp_path/'metadata'
    videos={'train':'train.','val':'val.mp4','test':'test.mp4'}
    source_rows={s:[{'video':v,'num_segments':1,'annotation_id':7,'description':'input action','times':[[0,0]]}] for s,v in videos.items()}
    hashes={}
    for split in ('train','val'):
        file=put(metadata/f'{split}_data.json',source_rows[split]);hashes[file.name]=r.digest(file)
    hashes['test_data.json']='unopened-test-hash'
    monkeypatch.setattr(r,'HASHES',hashes)
    splits={s:[{'video':v,'license':'by-nc','aws_url':'https://source/'+s,'original_url':'https://original/'+s}] for s,v in videos.items()}
    plan={'source':{'revision':r.REVISION,'sha256':hashes},'splits':splits,'use_terms':{'noncommercial_research_only':True}}
    selection=put(tmp_path/'data/selection.json',plan)
    monkeypatch.setattr(r,'SELECTION_CANONICAL_SHA256',r.canonical_sha(plan))
    monkeypatch.setattr(r,'SELECTION_COUNTS',{s:1 for s in videos})
    rows=[{'split':s,'video':v,'status':'complete','sha256':str(i+1)*64,'bytes':11} for i,(s,v) in enumerate(videos.items())]
    if recovery:
        failure={'split':'train','video':videos['train'],'status':'failed','reason':'Invalid source video filename'}
        first=lines(tmp_path/'data/original1.jsonl',[failure]);second=lines(tmp_path/'data/original2.jsonl',rows[1:]);retry=lines(tmp_path/'data/recovery.jsonl',rows[:1])
        journals=[[failure],rows[1:]];resolved,audit=reconcile(plan,journals,rows[:1]);inputs=[selection,first,second,retry]
    else:
        first=lines(tmp_path/'data/original.jsonl',rows);resolved,audit=reconcile(plan,[rows]);inputs=[selection,first]
    audit['input_sha256']={str(path.relative_to(tmp_path)).replace('/', '\\'):r.digest(path) for path in inputs}
    put(root/'source-accounting.json',audit);receipt=lines(root/'resolved-receipts.jsonl',resolved)
    decoder_rows=[{**row,'num_segments':1,'status':'valid','duration':5.,'fps':25.,'last_frame_decoded':True} for row in rows]
    decoded={'rows':decoder_rows,'total':3,'valid':3,'targets_used_for_filtering':False,'receipt_sha256':r.digest(receipt),'source_sha256':r.digest(Path(r.__file__).parents[1]/'data/decoder_preflight.py')}
    decoder=put(root/'decoder/preflight.json',decoded)
    media=[{'split':s,'video':videos[s],'num_segments':1,'sha256':str(i+1)*64,'bytes':11,**{k:splits[s][0][k] for k in ('license','aws_url','original_url')}} for i,s in enumerate(videos)]
    prepared=root/'prepared';files={}
    for name,value in [('media.json',media),*[(f'{s}.json',[{'id':f'{s}:0:7','video':videos[s],'source_video':videos[s],'text':'input action','times':[[0,0]]}]) for s in ('train','val')]]:
        path=put(prepared/name,value);files[name]=r.digest(path)
    put(prepared/'manifest.json',{'source_revision':r.REVISION,'official_metadata_sha256':hashes,'files':files,'receipt_sha256':r.digest(receipt),'selection_file_sha256':r.digest(selection),'decoder_preflight_sha256':r.digest(decoder),'excluded_decoder_media_sha256':[]})
    return root,metadata


@pytest.mark.parametrize('recovery',[False,True])
def test_complete_journal_routes_and_no_test_annotation_read(tmp_path,monkeypatch,recovery):
    root,metadata=preparation(tmp_path,monkeypatch,recovery)
    manifest,coverage=r.checked_preparation(root,metadata)
    assert coverage=={'candidate_count':3,'retained_media':3,'decoder_sources':3,'decoder_excluded':0}
    assert not (metadata/'test_data.json').exists()


@pytest.mark.parametrize('defect',['input','resolved','decoder_source','decoder_coverage','prepared_hash','query_subset','test_annotations'])
def test_rejects_incomplete_or_changed_proof(tmp_path,monkeypatch,defect):
    root,metadata=preparation(tmp_path,monkeypatch)
    if defect=='input':(tmp_path/'data/original.jsonl').write_text('changed')
    elif defect=='resolved':lines(root/'resolved-receipts.jsonl',[])
    elif defect.startswith('decoder'):
        p=root/'decoder/preflight.json';value=r.read(p)
        if defect=='decoder_source':value['source_sha256']='wrong'
        else:value['rows'].pop()
        put(p,value)
    elif defect=='prepared_hash':(root/'prepared/train.json').write_text('[]')
    elif defect=='query_subset':
        path=put(root/'prepared/train.json',[]);manifest=r.read(root/'prepared/manifest.json');manifest['files']['train.json']=r.digest(path);put(root/'prepared/manifest.json',manifest)
    else:put(root/'prepared/test.json',[{'gold':'forbidden'}])
    with pytest.raises(ValueError):r.checked_preparation(root,metadata)


def test_historical_guard_remains_strict(tmp_path):
    protocol=put(tmp_path/'protocol.json',{'source_manifest_sha256':'not-historical'})
    with pytest.raises(ValueError,match='manifest mismatch'):legacy.checked_cache(tmp_path,protocol)


def fake_data():
    values=torch.zeros(1,512);values[0,0]=1
    item={'x':values,'v':values,'candidates':[{'video':'v','moment':[0,0]}],'queries':[{'video':'v','times':[[0,0]]}]}
    return {'train':item,'val':item}


def bound_fixture(tmp_path,monkeypatch):
    root,metadata=preparation(tmp_path,monkeypatch);cache=tmp_path/'cache';put(cache/'manifest.json',{'fixture':True})
    data=fake_data();manifest={'encoder':{'fixture':True},'encoding_identity':{'fixture':True}};coverage={'candidate_count':3}
    checker=Mock(return_value=(data,manifest,coverage));monkeypatch.setattr(r,'checked_research_cache',checker)
    monkeypatch.setattr(r,'scores_for',lambda *_:np.asarray([[.1]],dtype=np.float32))
    def metrics(scores,*_):return {'corpus_consensus_r1_iou50':float(scores[0,0]),'within_video_official_miou':.5}
    monkeypatch.setattr(r,'evaluate',metrics)
    protocol=tmp_path/'protocol.json';r.freeze(root,cache,protocol,metadata)
    return root,metadata,cache,protocol,checker


def test_freeze_is_fresh_and_training_checks_exact_protocol_before_data(tmp_path,monkeypatch):
    root,metadata,cache,protocol,checker=bound_fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='Fresh'):r.freeze(root,cache,protocol,metadata)
    checker.reset_mock()
    with pytest.raises(ValueError,match='fingerprint'):r.train(root,cache,tmp_path/'out',protocol,'wrong',metadata)
    checker.assert_not_called()
    core=Mock(return_value={'done':True});monkeypatch.setattr(r,'_train_checked',core)
    result=r.train(root,cache,tmp_path/'out',protocol,r.digest(protocol),metadata)
    assert result=={'done':True}
    assert core.call_args.args[-1]==r.digest(cache/'manifest.json')
    assert legacy.MANIFEST=='8b97485f9dcc1ad243103b0cc4298c82cc3f20ceafa52cad863d6d050f53a3df'


@pytest.mark.parametrize('phase1_score,phase2_score,expected',[(.4,.3,'phase1'),(.3,.4,'phase2'),(.3,.3,'phase1'),(.1,.1,'frozen_clip')])
def test_global_chooser_keeps_best_eligible_policy_and_fixed_ties(tmp_path,monkeypatch,phase1_score,phase2_score,expected):
    root,metadata,cache,protocol,_=bound_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(r,'score_policy',lambda bundle,*_:np.asarray([[bundle.get('adapted_score_weight',0.) or .1]],dtype=np.float32))
    p=r.read(protocol);paths=[];hashes=[]
    for name,score in [('phase1',phase1_score),('phase2',phase2_score)]:
        directory=tmp_path/name;directory.mkdir()
        measured=r.evaluate(np.asarray([[score]],dtype=np.float32))
        bundle={'schema':1,'states':[],'adapted_score_weight':score,'winner':name,'encoder':{'fixture':True},'cache_manifest_sha256':p['source_manifest_sha256'],'validation':measured,'test_targets_used':False}
        torch.save(bundle,directory/'selected.pt')
        report={'selected':name,'selected_validation':measured,'baseline':p['baseline'],'cache_manifest_sha256':p['source_manifest_sha256'],'selected_checkpoint_sha256':r.digest(directory/'selected.pt'),'protocol_sha256':r.digest(protocol),'test_targets_used':False}
        file=put(directory/'results.json',report);paths.append(directory);hashes.append(r.digest(file))
    result=r.choose(root,cache,*paths,tmp_path/'choice',protocol,r.digest(protocol),*hashes,metadata)
    assert result['selected'].startswith(expected)
    assert result['test_targets_used'] is False and result['checkpoint_replay_max_abs_error']==0


@pytest.mark.parametrize('name',['/data/a.json','D:\\data\\a.json','../data/a.json','data/../a.json','\\\\server\\a.json','elsewhere/a.json'])
def test_accounting_resolver_rejects_absolute_drive_traversal_and_nondata_paths(tmp_path,monkeypatch,name):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError,match='accounting path'):r.accounting_path(name)


def test_accounting_resolver_normalizes_windows_without_rewriting_proof(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path);path=put(tmp_path/'data/a.json',{'proof':'unchanged'})
    before=path.read_bytes()
    assert r.accounting_path('data\\a.json')==path.resolve()
    assert path.read_bytes()==before


def encoded_fixture(tmp_path,monkeypatch):
    root,metadata=preparation(tmp_path,monkeypatch)
    cache=tmp_path/'encoded';cache.mkdir()
    files={};frames=[]
    for media in r.read(root/'prepared/media.json'):
        split=media['split'];values=np.zeros((1,512),np.float32);values[0,0]=1
        np.save(cache/f'{split}-moments.npy',values,allow_pickle=False)
        put(cache/f'{split}-candidates.json',[{'video':media['video'],'moment':[0,0]}])
        frames.append({'video':media['video'],'split':split,'source_sha256':media['sha256'],'duration':5.,'fps':25.,'samples':r.frame_schedule(1,5.,25.)})
        if split!='test':
            np.save(cache/f'{split}-text.npy',values,allow_pickle=False)
            put(cache/f'{split}-queries.json',r.read(root/f'prepared/{split}.json'))
    put(cache/'frames.json',frames)
    files={path.name:r.digest(path) for path in cache.iterdir()}
    receipt={'model_id':r.MODEL_ID,'revision':r.MODEL_REVISION,'weights_sha256':r.WEIGHTS_SHA256,'loading_info':{}}
    media=r.read(root/'prepared/media.json')
    ordered=[row for split in ('train','val','test') for row in media if row['split']==split]
    identity={'source_manifest_sha256':r.digest(root/'prepared/manifest.json'),'encoder':receipt,
              'ordered_records_sha256':r.hashlib.sha256(json.dumps(ordered,sort_keys=True).encode()).hexdigest(),
              'encoder_source_sha256':r.digest(Path(r.__file__).with_name('moment_encoder.py')),
              'progress_source_sha256':r.digest(Path(r.__file__).with_name('encoding_progress.py')),
              'torch':str(torch.__version__),'device':'cpu','model_dtype':'torch.float32',
              'transformers':'fixture-version','numpy':np.__version__,'ffmpeg':'fixture-version','smoke_only':False,
              'pillow':'fixture-version','imageio':'fixture-version','imageio_ffmpeg':'fixture-version',
              'torch_build_sha256':'a'*64,'cpu_identity':{'machine':'fixture','processor':''}}
    put(cache/'manifest.json',{'status':'complete','smoke_only':False,'source_manifest_sha256':r.digest(root/'prepared/manifest.json'),'test_text_or_targets_encoded':False,'encoder':receipt,'encoding_identity':identity,'files':files})
    return root,metadata,cache


def test_complete_encoded_cache_without_test_annotations(tmp_path,monkeypatch):
    root,metadata,cache=encoded_fixture(tmp_path,monkeypatch)
    data,_,coverage=r.checked_research_cache(root,cache,metadata)
    assert data['train']['positive_index'].valid.tolist()==[True]
    assert coverage['retained_media']==3
    assert not (cache/'test-queries.json').exists()


@pytest.mark.parametrize('defect',['source_binding','features','candidate_subset','frame_timing','test_annotations'])
def test_encoded_cache_rejects_partial_tampered_or_test_annotation_inputs(tmp_path,monkeypatch,defect):
    root,metadata,cache=encoded_fixture(tmp_path,monkeypatch)
    manifest=r.read(cache/'manifest.json')
    if defect=='source_binding':manifest['source_manifest_sha256']='wrong'
    elif defect=='features':(cache/'val-moments.npy').write_bytes(b'changed')
    elif defect=='candidate_subset':
        path=put(cache/'test-candidates.json',[]);manifest['files'][path.name]=r.digest(path)
    elif defect=='frame_timing':
        frames=r.read(cache/'frames.json');frames[0]['duration']=4.
        path=put(cache/'frames.json',frames);manifest['files'][path.name]=r.digest(path)
    else:
        path=put(cache/'test-queries.json',[{'gold':'forbidden'}]);manifest['files'][path.name]=r.digest(path)
    put(cache/'manifest.json',manifest)
    with pytest.raises((ValueError,EOFError)):r.checked_research_cache(root,cache,metadata)


def test_phase1_requires_bound_protocol_before_calling_existing_trainer(tmp_path,monkeypatch):
    root,metadata,cache,protocol,_=bound_fixture(tmp_path,monkeypatch)
    trainer=Mock();monkeypatch.setattr(r,'phase1_train',trainer)
    with pytest.raises(ValueError,match='fingerprint'):
        r.train_phase1(root,cache,tmp_path/'out',protocol,'wrong',metadata)
    trainer.assert_not_called()
    p=r.read(protocol)
    def terminal(cache_path,output):
        Path(output).mkdir()
        return {'baseline':p['baseline'],'cache_manifest_sha256':p['source_manifest_sha256']}
    trainer.side_effect=terminal
    result=r.train_phase1(root,cache,tmp_path/'out',protocol,r.digest(protocol),metadata)
    assert trainer.call_count==1 and result['protocol_sha256']==r.digest(protocol)
    assert r.read(tmp_path/'out/results.json')['full_coverage']==p['coverage']


def test_shared_core_preserves_nonhistorical_binding_and_baseline_metadata(tmp_path,monkeypatch):
    data=fake_data()
    class Index:valid=np.asarray([True])
    data['train']['positive_index']=Index()
    baseline=legacy.evaluate(legacy.scores_for([],data['val']['x'],data['val']['v']),data['val']['candidates'],data['val']['queries'])
    protocol=put(tmp_path/'protocol.json',{'fixture':'no fits'})
    monkeypatch.setattr(legacy,'LAMBDAS',[])
    result=legacy._train_checked(data,{'encoder':{}},{'baseline':baseline},tmp_path/'out',protocol,'full-cache-binding')
    assert result['cache_manifest_sha256']=='full-cache-binding'
    assert result['selected']=='frozen_clip'
    assert torch.load(tmp_path/'out/selected.pt',weights_only=True)['cache_manifest_sha256']=='full-cache-binding'


@pytest.mark.parametrize('defect',['missing','encoder_source_sha256','progress_source_sha256','ordered_records_sha256',
                                 'source_manifest_sha256','encoder','device','model_dtype','smoke_only',
                                 'pillow','imageio','imageio_ffmpeg','torch_build_sha256','cpu_identity'])
def test_research_cache_requires_bound_encoding_execution_identity(tmp_path,monkeypatch,defect):
    root,metadata,cache=encoded_fixture(tmp_path,monkeypatch)
    manifest=r.read(cache/'manifest.json')
    if defect=='missing':manifest.pop('encoding_identity')
    elif defect=='smoke_only':manifest['encoding_identity'][defect]=True
    elif defect=='encoder':manifest['encoding_identity'][defect]={}
    elif defect in ('pillow','imageio','imageio_ffmpeg'):manifest['encoding_identity'][defect]=''
    else:manifest['encoding_identity'][defect]='changed'
    put(cache/'manifest.json',manifest)
    with pytest.raises(ValueError,match='identity'):r.checked_research_cache(root,cache,metadata)

def test_checked_preparation_validates_optional_decoder_recovery(tmp_path,monkeypatch):
    import copy
    from src.data import decoder_recovery as d
    root,metadata=preparation(tmp_path,monkeypatch)
    path=root/'decoder/preflight.json';decoded=r.read(path)
    for row in decoded['rows']:row['last_frame']=93
    parent=copy.deepcopy(decoded);parent['rows'][0].update(status='invalid',error='PermissionError: [WinError 32] locked',duration=0.,fps=0.,last_frame_decoded=False);parent['rows'][0].pop('last_frame');parent['valid']-=1
    evidence=root/'decoder/recovery';evidence.mkdir()
    put(evidence/'parent-preflight.json',parent);lines(evidence/'parent-rows.jsonl',parent['rows'])
    (evidence/'recover.py').write_text('# Private source-only helper snapshot\n',encoding='utf-8')
    source=Path(r.__file__).parents[1]/'data/decoder_preflight.py';old=parent['rows'][0];new=decoded['rows'][0]
    container={'mode':'verified_mp4_copy','original_video':old['video'],'raw_sha256':old['sha256'],'source_bytes':11,'copy_name':'source.mp4','copy_sha256':old['sha256'],'copy_bytes':11,'retained_private':True}
    attempt={'outcome':'decoded_valid','probe':None,'input_container':container,'parent_row_sha256':d.canonical(old),'raw_sha256':old['sha256'],'attempt':1,'result':new,'parent_preflight_sha256':d.digest(evidence/'parent-preflight.json'),'source_file_sha256':d.digest(source),'source_function_sha256':d.function_hash(source)}
    lines(evidence/'attempts.jsonl',[attempt])
    recovery={field:d.digest(evidence/name) for field,name in d.FILES.items()}
    recovery.update(unchanged_inspect_source_sha256=d.digest(source),inspect_function_sha256=d.function_hash(source),targets_used_for_filtering=False,replacements=[{'index':0,'parent_row_sha256':d.canonical(old),'replacement_row_sha256':d.canonical(new),'raw_sha256':old['sha256'],'outcome':'decoded_valid'}])
    decoded['recovery']=recovery;put(path,decoded)
    manifest=r.read(root/'prepared/manifest.json');manifest['decoder_preflight_sha256']=r.digest(path);put(root/'prepared/manifest.json',manifest)
    assert r.checked_preparation(root,metadata)[1]['decoder_sources']==3
    (evidence/'recover.py').write_text('changed',encoding='utf-8')
    with pytest.raises(ValueError,match='artifact'):r.checked_preparation(root,metadata)
