"""CPU service-window continuation for the unchanged full-research adapter grids.

Snapshots and completed seeds are private artifacts. A stopped window cannot
produce a selected model or benchmark report. Every continuation uses a fresh
output directory and verifies its parent snapshot, code, cache and runtime.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
from src.models.moment_training import Adapter, CONFIGS, SEEDS, digest, scores_for, multipositive_loss
from src.models.localization_followup import LAMBDAS, ALPHAS, conditional_loss, score_policy
from src.models.moments import evaluate


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def cpu_identity():
    result={'machine':platform.machine(),'processor':platform.processor(),'logical_cpus':os.cpu_count()}
    path=Path('/proc/cpuinfo')
    if path.is_file():
        rows={}
        for line in path.read_text(encoding='utf-8').splitlines():
            if ':' in line:
                key,value=line.split(':',1)
                if key.strip() in ('model name','vendor_id','flags','Features') and key.strip() not in rows:rows[key.strip()]=value.strip()
        result['linux_cpu']=rows
    return result


def runtime():
    return {'python':platform.python_version(),'platform':platform.platform(),'processor':platform.processor(),'cpu':cpu_identity(),
            'torch':str(torch.__version__),'numpy':np.__version__,'device':'cpu','dtype':str(torch.get_default_dtype()),
            'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),
            'deterministic':torch.are_deterministic_algorithms_enabled(),'mkldnn':torch.backends.mkldnn.enabled,
            'matmul_precision':torch.get_float32_matmul_precision(),
            'torch_build':hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
            'environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','PYTHONHASHSEED')}}


def rng_state():
    algorithm,keys,position,has_gauss,cached=np.random.get_state()
    return {'torch':torch.get_rng_state(),'python':random.getstate(),
            'numpy':{'algorithm':algorithm,'keys':torch.tensor(keys.astype(np.int64)),
                     'position':position,'has_gauss':has_gauss,'cached_gaussian':cached}}


def restore_rng(saved):
    torch.set_rng_state(saved['torch']);random.setstate(saved['python'])
    value=saved['numpy']
    np.random.set_state((value['algorithm'],value['keys'].numpy().astype(np.uint32),
                         value['position'],value['has_gauss'],value['cached_gaussian']))


def finite_tree(value):
    if isinstance(value,torch.Tensor):return bool(torch.isfinite(value).all())
    if isinstance(value,dict):return all(finite_tree(v) for v in value.values())
    if isinstance(value,(tuple,list)):return all(finite_tree(v) for v in value)
    if isinstance(value,float):return math.isfinite(value)
    return True


def save_new(path,value):
    path=Path(path)
    with path.open('xb') as stream:
        torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
    return digest(path)


def json_new(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())


def primary(model,validation):
    model.eval()
    with torch.inference_mode():ranked=(model(validation['x'])@validation['v'].T).argmax(dim=1)
    return float(torch.tensor([int(candidate) in positives for candidate,positives in
                              zip(ranked,validation['positive_index'].indices)],dtype=torch.float32).mean())


def eligible(metrics,baseline):
    return (metrics['corpus_consensus_r1_iou50']-baseline['corpus_consensus_r1_iou50']>=.01-1e-12 and
            metrics['within_video_official_miou']>=baseline['within_video_official_miou']-.01-1e-12)


def production_spec(phase):
    if phase=='phase1':configs=copy.deepcopy(CONFIGS)
    elif phase=='phase2':configs=[{'name':f'lambda{w}','lambda':w,'lr':.0003,'weight_decay':.01} for w in LAMBDAS]
    else:raise ValueError('Unknown training phase')
    return {'phase':phase,'configs':configs,'seeds':list(SEEDS),'batch':64,'temperature':.07,
            'maximum_epochs':50,'minimum_epochs':5,'patience':8,
            'alphas':list(ALPHAS) if phase=='phase2' else [1.]}


def seed_identity(identity,spec,grid,seed):
    return {'run':identity,'config_index':grid,'config':spec['configs'][grid],'seed':seed}


def checked_active(active,spec,count):
    if not isinstance(active,dict):raise ValueError('Invalid active seed state')
    required={'model','optimizer','generator','epoch','cursor','permutation','total','history','best','best_epoch','best_state'}
    if set(active)!=required:raise ValueError('Active checkpoint fields differ')
    epoch,cursor,history=active['epoch'],active['cursor'],active['history']
    if type(epoch) is not int or not 1<=epoch<=spec['maximum_epochs'] or not isinstance(history,list) or len(history)!=epoch-1:
        raise ValueError('Invalid epoch/history cursor')
    for i,row in enumerate(history,1):
        if row.get('epoch')!=i or set(row)!={'epoch','train_loss','validation_corpus_consensus_r1_iou50'} or not math.isfinite(row['train_loss']) or row['train_loss']<0 or not 0<=row['validation_corpus_consensus_r1_iou50']<=1:
            raise ValueError('Malformed completed epoch history')
    if type(cursor) is not int or not 0<=cursor<=count or (cursor!=count and cursor%spec['batch']):raise ValueError('Invalid batch cursor')
    order=active['permutation']
    if order is None:
        if cursor!=0 or active['total']!=0:raise ValueError('Invalid new epoch accumulator')
    elif not isinstance(order,torch.Tensor) or order.dtype!=torch.int64 or order.shape!=(count,) or not torch.equal(order.sort().values,torch.arange(count)):
        raise ValueError('Invalid saved epoch permutation')
    if not isinstance(active['total'],(int,float)) or not math.isfinite(active['total']) or active['total']<0:
        raise ValueError('Invalid epoch loss accumulator')
    if not history:
        if active['best']!=-1. or active['best_epoch']!=0 or active['best_state'] is not None:raise ValueError('Unexpected pre-validation best state')
    else:
        values=[row['validation_corpus_consensus_r1_iou50'] for row in history];best=max(values);best_epoch=values.index(best)+1
        if active['best']!=best or active['best_epoch']!=best_epoch or not isinstance(active['best_state'],dict):raise ValueError('Best checkpoint does not match earliest validation maximum')
    # Strict restoration below verifies parameter shapes and optimizer ordering.
    if not isinstance(active['generator'],torch.Tensor) or active['generator'].dtype!=torch.uint8:
        raise ValueError('Invalid saved permutation generator')


def checked_resume(resume,resume_sha256,identity,spec,output):
    resume=Path(resume)
    if not resume_sha256 or digest(resume)!=resume_sha256:raise ValueError('Resume snapshot fingerprint mismatch')
    receipt=json.loads(resume.with_suffix('.json').read_text(encoding='utf-8'))
    if receipt.get('snapshot_sha256')!=resume_sha256 or receipt.get('identity_sha256')!=fingerprint(identity):
        raise ValueError('Resume receipt identity mismatch')
    state=torch.load(resume,map_location='cpu',weights_only=True)
    if state.get('schema')!=1 or state.get('identity')!=identity or state.get('spec')!=spec or not finite_tree(state):
        raise ValueError('Resume source/cache/protocol/runtime/state mismatch')
    if state.get('stage') not in ('fit','candidate','final'):
        raise ValueError('Invalid continuation stage')
    if type(state.get('grid')) is not int or not 0<=state['grid']<=len(spec['configs']):
        raise ValueError('Invalid continuation grid cursor')
    if type(state.get('seed_index')) is not int or not 0<=state['seed_index']<=len(spec['seeds']):raise ValueError('Invalid seed cursor')
    if state['stage']=='fit' and (state['grid']==len(spec['configs']) or state['seed_index']==len(spec['seeds'])):raise ValueError('Invalid active fit cursor')
    if state['stage']=='candidate' and (state['grid']==len(spec['configs']) or state['seed_index']!=len(spec['seeds']) or state['active'] is not None):raise ValueError('Invalid candidate cursor')
    if state['stage']=='final' and (state['grid']!=len(spec['configs']) or state['seed_index']!=0 or state['active'] is not None):raise ValueError('Invalid final cursor')
    if len(state['runs'])!=state['grid'] or (spec['phase']=='phase2' and len(state['policies'])!=state['grid']*len(spec['alphas'])+state['policy_index']):raise ValueError('Incomplete candidate policy coverage')
    expected_count=state['grid']*len(spec['seeds'])+state['seed_index']
    if len(state['completed'])!=expected_count or not 0<=state['seed_index']<=len(spec['seeds']):
        raise ValueError('Completed seed/cursor coverage mismatch')
    for i,record in enumerate(state['completed']):
        grid,position=divmod(i,len(spec['seeds']));seed=spec['seeds'][position]
        name=f'config{grid}-seed{seed}.pt'
        if record.get('name')!=name or record.get('identity')!=seed_identity(identity,spec,grid,seed):
            raise ValueError('Completed seed identity mismatch')
        path=resume.parent/name
        if not path.is_file():raise ValueError('Missing completed seed artifact')
        if digest(path)!=record.get('sha256'):raise ValueError('Completed seed hash mismatch')
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if saved.get('identity')!=record['identity'] or not finite_tree(saved):raise ValueError('Completed seed payload mismatch')
        shutil.copyfile(path,output/name)
    return state


def engine(data,manifest,baseline,output,identity,spec,*,resume=None,resume_sha256=None,
           max_seconds=None,checkpoint_every_batches=None,checkpoint_every_seconds=300.,max_batches=None,service_started=None):
    """Internal engine; custom specs are only for mechanical tests, not the CLI."""
    started=time.monotonic() if service_started is None else service_started
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    if torch.get_default_dtype()!=torch.float32 or str(torch.get_default_device())!='cpu':raise ValueError('CPU float32 training required')
    output=Path(output)
    if output.exists():raise ValueError('Fresh continuation output required')
    if max_seconds is not None and (isinstance(max_seconds,bool) or not math.isfinite(max_seconds) or max_seconds<=0):
        raise ValueError('Positive finite service window required')
    if checkpoint_every_batches is not None and (type(checkpoint_every_batches) is not int or checkpoint_every_batches<1):raise ValueError('Positive checkpoint interval required')
    if isinstance(checkpoint_every_seconds,bool) or not math.isfinite(checkpoint_every_seconds) or checkpoint_every_seconds<=0:raise ValueError('Positive checkpoint time interval required')
    if max_batches is not None and (type(max_batches) is not int or max_batches<1):raise ValueError('Positive batch limit required')
    if (resume is None)!=(resume_sha256 is None):raise ValueError('Resume path and fingerprint required together')
    identity=copy.deepcopy(identity);identity['runtime']=runtime();identity['spec_sha256']=fingerprint(spec)
    training,validation=data['train'],data['val'];valid=torch.from_numpy(training['positive_index'].valid)
    query_ids=np.flatnonzero(training['positive_index'].valid);x=training['x'][valid];v=training['v']
    if not len(x):raise ValueError('No training query has a consensus-positive moment')
    output.mkdir(parents=True)
    if resume is None:
        measured=evaluate(scores_for([],validation['x'],validation['v']),validation['candidates'],validation['queries'])
        from src.models.research_followup import metrics_match
        if not metrics_match(measured,baseline):raise ValueError('Frozen validation baseline mismatch')
        state={'schema':1,'identity':identity,'spec':spec,'baseline':baseline,'grid':0,'seed_index':0,'stage':'fit',
               'active':None,'completed':[],'runs':[],'policies':[],'policy_index':0,'candidate_metrics':[],
               'winner_name':'frozen_clip','winner_states':[],'winner_alpha':0.,'winner_metrics':baseline,
               'best_promoted':-1.,'batches':0,'rng':rng_state(),'service_seconds':0.,'controls':{}}
    else:state=checked_resume(resume,resume_sha256,identity,spec,output)
    if state['baseline']!=baseline:raise ValueError('Resume baseline differs')
    window_batches=0;snapshots=0;model=None;optimizer=None;generator=None;last_checkpoint=time.monotonic()
    # Validation and completed-policy replay are intentionally indivisible work
    # units. Budget checks occur before them, never halfway through a metric.
    def checkpoint(stopped):
        nonlocal snapshots,last_checkpoint
        if state['active'] is not None:
            state['active'].update(model=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(optimizer.state_dict()),generator=generator.get_state())
        state['rng']=rng_state();saved=copy.deepcopy(state)
        saved['service_seconds']=state['service_seconds']+time.monotonic()-started
        snapshots+=1;path=output/f'checkpoint-{snapshots:06d}.pt';sha=save_new(path,saved)
        last_checkpoint=time.monotonic()
        json_new(path.with_suffix('.json'),{'snapshot_sha256':sha,'identity_sha256':fingerprint(identity),
                                         'completed_seeds':len(state['completed']),'batches':state['batches'],'stage':state['stage']})
        if stopped:
            result={'status':'paused_service_window','snapshot':path.name,'snapshot_sha256':sha,
                    'completed_seeds':len(state['completed']),'batches':state['batches'],'test_targets_used':False}
            json_new(output/'continuation.json',result);return result
    def boundary():
        return ((max_seconds is not None and time.monotonic()-started>=max_seconds) or
                (max_batches is not None and window_batches>=max_batches))
    # Loading constructors must not advance the resumed random streams.
    if state['active'] is not None:
        active=state['active'];checked_active(active,spec,len(x));config=spec['configs'][state['grid']]
        model=Adapter();model.load_state_dict(active['model'],strict=True)
        if active['best_state'] is not None:
            model.load_state_dict(active['best_state'],strict=True);model.load_state_dict(active['model'],strict=True)
        optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
        expected_groups=copy.deepcopy(optimizer.state_dict()['param_groups'])
        optimizer.load_state_dict(active['optimizer'])
        if optimizer.state_dict()['param_groups']!=expected_groups:raise ValueError('Saved optimizer hyperparameters differ')
        expected_steps=(active['epoch']-1)*math.ceil(len(x)/spec['batch'])+math.ceil(active['cursor']/spec['batch'])
        for group in optimizer.param_groups:
            for parameter in group['params']:
                saved=optimizer.state.get(parameter,{})
                if set(saved)!={'step','exp_avg','exp_avg_sq'} or saved['step'].numel()!=1 or float(saved['step'])!=expected_steps or any(saved[key].shape!=parameter.shape or saved[key].dtype!=parameter.dtype for key in ('exp_avg','exp_avg_sq')):
                    raise ValueError('Saved optimizer moments/step geometry differ')
        generator=torch.Generator();generator.set_state(active['generator'])
    restore_rng(state['rng'])
    while state['grid']<len(spec['configs']):
        if boundary():return checkpoint(True)
        grid=state['grid'];config=spec['configs'][grid]
        if state['stage']=='fit':
            seed=spec['seeds'][state['seed_index']]
            if state['active'] is None:
                torch.manual_seed(seed);generator=torch.Generator().manual_seed(seed);model=Adapter()
                optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
                state['active']={'epoch':1,'cursor':0,'permutation':None,'total':0.,'history':[],
                                 'best':-1.,'best_epoch':0,'best_state':None}
            active=state['active']
            if active['permutation'] is None:active['permutation']=torch.randperm(len(x),generator=generator)
            if active['cursor']<len(x):
                model.train();ix=active['permutation'][active['cursor']:active['cursor']+spec['batch']]
                optimizer.zero_grad();logits=model(x[ix])@v.T/spec['temperature']
                positive=torch.from_numpy(training['positive_index'].mask(query_ids[ix.numpy()]))
                if spec['phase']=='phase1':loss=multipositive_loss(logits,positive)
                else:
                    loss=conditional_loss(logits,positive,torch.ones_like(positive))
                    if config['lambda']:
                        allowed=torch.from_numpy(training['positive_index'].same_video_mask(query_ids[ix.numpy()]))
                        loss=loss+config['lambda']*conditional_loss(logits,positive,allowed)
                loss.backward()
                if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise ValueError('Nonfinite training')
                optimizer.step();active['total']+=float(loss.detach())*len(ix);active['cursor']+=len(ix)
                state['batches']+=1;window_batches+=1
                if (checkpoint_every_batches is not None and state['batches']%checkpoint_every_batches==0) or time.monotonic()-last_checkpoint>=checkpoint_every_seconds:checkpoint(False)
                continue
            value=primary(model,validation);epoch=active['epoch']
            active['history'].append({'epoch':epoch,'train_loss':active['total']/len(x),'validation_corpus_consensus_r1_iou50':value})
            if value>active['best']:
                active['best']=value;active['best_epoch']=epoch;active['best_state']=copy.deepcopy(model.state_dict())
            print(json.dumps({'phase':spec['phase'],'config':config['name'],'seed':seed,**active['history'][-1]}),flush=True)
            done=epoch>=spec['maximum_epochs'] or (epoch>=spec['minimum_epochs'] and epoch-active['best_epoch']>=spec['patience'])
            if done:
                bound=seed_identity(identity,spec,grid,seed);name=f'config{grid}-seed{seed}.pt'
                payload={'identity':bound,'state':active['best_state'],'best_epoch':active['best_epoch'],'history':active['history'],
                         'final_model':copy.deepcopy(model.state_dict()),'optimizer':copy.deepcopy(optimizer.state_dict()),
                         'generator':generator.get_state(),'rng':rng_state()}
                sha=save_new(output/name,payload);state['completed'].append({'name':name,'sha256':sha,'identity':bound})
                state['seed_index']+=1;state['active']=None;model=None;optimizer=None;generator=None
                if state['seed_index']==len(spec['seeds']):state['stage']='candidate'
                checkpoint(False)
            else:
                active.update(epoch=epoch+1,cursor=0,permutation=None,total=0.)
            continue
        # Every completed seed is hash-bound and re-used; no partial seed/grid
        # is eligible for a selected model.
        records=state['completed'][grid*len(spec['seeds']):(grid+1)*len(spec['seeds'])]
        completed=[torch.load(output/r['name'],map_location='cpu',weights_only=True) for r in records]
        states=[item['state'] for item in completed]
        alpha=spec['alphas'][state['policy_index']]
        bundle={'schema':1,'states':states,'adapted_score_weight':alpha}
        measured=evaluate(score_policy(bundle,validation['x'],validation['v']),validation['candidates'],validation['queries'])
        adopted=eligible(measured,baseline);pvalue=measured['corpus_consensus_r1_iou50']
        if spec['phase']=='phase2':state['policies'].append({'lambda':config['lambda'],'adapted_score_weight':alpha,'validation':measured,'promotion_eligible':adopted})
        state['candidate_metrics'].append(measured)
        if adopted and pvalue>state['best_promoted']:
            state.update(best_promoted=pvalue,winner_states=states,winner_alpha=alpha,winner_metrics=measured,
                         winner_name=config['name'] if spec['phase']=='phase1' else f"lambda{config['lambda']}-alpha{alpha}")
        state['policy_index']+=1
        if state['policy_index']==len(spec['alphas']):
            seed_reports=[{'seed':seed,'best_epoch':item['best_epoch'],'history':item['history']} for seed,item in zip(spec['seeds'],completed)]
            if spec['phase']=='phase1':state['runs'].append({'config':config,'seeds':seed_reports,'validation':measured,'promotion_eligible':adopted})
            else:state['runs'].append({'lambda':config['lambda'],'seeds':seed_reports})
            state.update(grid=grid+1,seed_index=0,stage='fit',policy_index=0,candidate_metrics=[])
        checkpoint(False)
    state['stage']='final'
    if boundary():return checkpoint(True)
    if spec['phase']=='phase1':
        for name in ('random_control','chronological_control'):
            if name in state['controls']:continue
            if boundary():return checkpoint(True)
            shape=(len(validation['x']),len(validation['v']))
            values=np.random.default_rng(42).random(shape) if name=='random_control' else np.broadcast_to(-np.arange(shape[1],dtype=float),shape)
            state['controls'][name]=evaluate(values,validation['candidates'],validation['queries']);checkpoint(False)
    required=len(spec['configs'])*len(spec['seeds'])
    if len(state['completed'])!=required or len(state['runs'])!=len(spec['configs']) or (spec['phase']=='phase2' and len(state['policies'])!=len(spec['configs'])*len(spec['alphas'])):
        raise ValueError('Cannot finalize an incomplete fit/policy grid')
    for record in state['completed']:
        if not (output/record['name']).is_file() or digest(output/record['name'])!=record['sha256']:raise ValueError('Cannot finalize missing/changed completed seed')
    bundle={'schema':1,'winner':state['winner_name'],'states':state['winner_states'],'adapted_score_weight':state['winner_alpha'],
            'encoder':manifest['encoder'],'cache_manifest_sha256':identity['cache_manifest_sha256'],
            'validation':state['winner_metrics'],'training_consensus_no_positive_queries':int((~valid).sum()),'test_targets_used':False}
    sha=save_new(output/'selected.pt',bundle);restored=torch.load(output/'selected.pt',map_location='cpu',weights_only=True)
    delta=float(np.max(np.abs(score_policy(bundle,validation['x'][:3],validation['v'])-score_policy(restored,validation['x'][:3],validation['v']))))
    if delta>1e-6:raise ValueError('Saved policy replay mismatch')
    state['stage']='done';checkpoint(False)
    result={'baseline':baseline,'runs':state['runs'],'selected':state['winner_name'],'selected_validation':state['winner_metrics'],
            'protocol_sha256':identity['protocol_sha256'],'cache_manifest_sha256':identity['cache_manifest_sha256'],
            'selected_checkpoint_sha256':sha,'checkpoint_replay_max_abs_error':delta,'test_targets_used':False,
            'service_seconds':state['service_seconds']+time.monotonic()-started,'completed_fits':len(state['completed']),
            'required_fits':len(spec['configs'])*len(spec['seeds']),'continuation_identity':identity,**state['controls']}
    if spec['phase']=='phase2':result['policies']=state['policies']
    json_new(output/'results.json',result)
    return result


def train(preparation,cache,output,protocol,protocol_sha256,phase,metadata='data/external/didemo',
          resume=None,resume_sha256=None,max_seconds=None,checkpoint_every_batches=None,checkpoint_every_seconds=300.):
    started=time.monotonic()
    if max_seconds is not None and (isinstance(max_seconds,bool) or not math.isfinite(max_seconds) or max_seconds<=0):raise ValueError('Positive finite service window required')
    from src.models.research_followup import checked_protocol
    if Path(output).exists():raise ValueError('Fresh continuation output required')
    data,manifest,p=checked_protocol(preparation,cache,protocol,protocol_sha256,metadata)
    identity={'schema':1,'phase':phase,'protocol_sha256':protocol_sha256,'cache_manifest_sha256':p['source_manifest_sha256'],
              'implementation_sha256':p['implementation_sha256'],'coverage':p['coverage'],'encoding_identity':p['encoding_identity']}
    return engine(data,manifest,p['baseline'],output,identity,production_spec(phase),resume=resume,resume_sha256=resume_sha256,
                  max_seconds=max_seconds,checkpoint_every_batches=checkpoint_every_batches,checkpoint_every_seconds=checkpoint_every_seconds,service_started=started)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['phase1','phase2'],required=True)
    for name in ('preparation','cache','output','protocol','protocol-sha256'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--metadata',default='data/external/didemo');parser.add_argument('--resume');parser.add_argument('--resume-sha256')
    parser.add_argument('--max-seconds',type=float);parser.add_argument('--checkpoint-every-batches',type=int);parser.add_argument('--checkpoint-every-seconds',type=float,default=300.)
    args=vars(parser.parse_args());print(json.dumps(train(**args),indent=2))


if __name__=='__main__':main()
