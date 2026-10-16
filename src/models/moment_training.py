"""Frozen-protocol residual text adapters, selected without test targets."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from src.models.moments import evaluate,positive_mask

CONFIGS=[{'name':'lr001','lr':.001,'weight_decay':.01},
         {'name':'lr0003','lr':.0003,'weight_decay':.01},
         {'name':'decay01','lr':.001,'weight_decay':.1}]
SEEDS=[42,1337,2026]

class Adapter(nn.Module):
    def __init__(self):
        super().__init__();self.down=nn.Linear(512,64);self.up=nn.Linear(64,512)
        nn.init.zeros_(self.up.weight);nn.init.zeros_(self.up.bias)
    def forward(self,x):return nn.functional.normalize(x+self.up(nn.functional.gelu(self.down(x))),dim=-1)

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def load_cache(path,allow_smoke=False):
    path=Path(path);manifest=json.loads((path/'manifest.json').read_text())
    if manifest['smoke_only'] and not allow_smoke:raise ValueError('Smoke cache cannot support benchmark training')
    for name,expected in manifest['files'].items():
        if digest(path/name)!=expected:raise ValueError('Changed feature cache')
    result={}
    for split in ('train','val'):
        x=np.load(path/f'{split}-text.npy',allow_pickle=False);v=np.load(path/f'{split}-moments.npy',allow_pickle=False)
        q=json.loads((path/f'{split}-queries.json').read_text(encoding='utf-8'))
        c=json.loads((path/f'{split}-candidates.json').read_text())
        if x.shape!=(len(q),512) or v.shape!=(len(c),512):raise ValueError('Feature shape mismatch')
        if not np.isfinite(x).all() or not np.isfinite(v).all():raise ValueError('Nonfinite cached features')
        if not np.allclose(np.linalg.norm(x,axis=1),1,atol=1e-4) or not np.allclose(np.linalg.norm(v,axis=1),1,atol=1e-4):raise ValueError('Unnormalized CLIP features')
        result[split]={'x':torch.from_numpy(x),'v':torch.from_numpy(v),'queries':q,'candidates':c,
                       'positive':torch.from_numpy(np.stack([positive_mask(c,row) for row in q]))}
    return result,manifest

def multipositive_loss(scores,positives):
    if scores.shape!=positives.shape or not positives.any(dim=1).all():raise ValueError('Missing contrastive positive')
    return (torch.logsumexp(scores,dim=1)-torch.logsumexp(scores.masked_fill(~positives,float('-inf')),dim=1)).mean()

def scores_for(states,x,v):
    if not states:return (x@v.T).numpy()
    scores=[]
    for state in states:
        model=Adapter();model.load_state_dict(state,strict=True);model.eval()
        with torch.inference_mode():scores.append((model(x)@v.T).numpy())
    return np.mean(scores,axis=0)

def train(cache,output):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    output=Path(output)
    if output.exists():raise ValueError('Preserve prior training evidence; use a fresh output directory')
    data,manifest=load_cache(cache);output.mkdir(parents=True,exist_ok=False)
    training=data['train'];validation=data['val'];valid=training['positive'].any(dim=1)
    x=training['x'][valid];positive=training['positive'][valid];v=training['v']
    if len(x)==0:raise ValueError('No training query has a consensus-positive moment')
    baseline_scores=scores_for([],validation['x'],validation['v'])
    baseline=evaluate(baseline_scores,validation['candidates'],validation['queries'])
    baseline_name='frozen_clip';winner_states=[];winner_metrics=baseline;winner_name=baseline_name
    reports=[];start=time.monotonic();best_promoted=-1.
    for config in CONFIGS:
        states=[];seed_reports=[]
        for seed in SEEDS:
            torch.manual_seed(seed);rng=torch.Generator().manual_seed(seed)
            model=Adapter();optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
            best=-1.;best_state=None;best_epoch=0;history=[]
            for epoch in range(1,51):
                model.train();order=torch.randperm(len(x),generator=rng);total=0.
                for indices in order.split(64):
                    optimizer.zero_grad();logits=model(x[indices])@v.T/.07
                    loss=multipositive_loss(logits,positive[indices]);loss.backward();optimizer.step()
                    total+=float(loss.detach())*len(indices)
                model.eval()
                with torch.inference_mode():ranked=(model(validation['x'])@validation['v'].T).argmax(dim=1)
                primary=float(validation['positive'][torch.arange(len(ranked)),ranked].float().mean())
                history.append({'epoch':epoch,'train_loss':total/len(x),'validation_corpus_consensus_r1_iou50':primary})
                if primary>best:
                    best=primary;best_epoch=epoch;best_state={k:t.detach().clone() for k,t in model.state_dict().items()}
                print(json.dumps({'config':config['name'],'seed':seed,**history[-1]}),flush=True)
                if epoch>=5 and epoch-best_epoch>=8:break
            states.append(best_state);torch.save({'state':best_state,'config':config,'seed':seed,'best_epoch':best_epoch},output/f'{config["name"]}-{seed}.pt')
            seed_reports.append({'seed':seed,'best_epoch':best_epoch,'history':history})
        score=scores_for(states,validation['x'],validation['v']);metrics=evaluate(score,validation['candidates'],validation['queries'])
        primary=metrics['corpus_consensus_r1_iou50']
        eligible=primary-baseline['corpus_consensus_r1_iou50']>=.01-1e-12 and metrics['within_video_official_miou']>=baseline['within_video_official_miou']-.01-1e-12
        reports.append({'config':config,'seeds':seed_reports,'validation':metrics,'promotion_eligible':eligible})
        if eligible and primary>best_promoted:
            best_promoted=primary;winner_name=config['name'];winner_states=states;winner_metrics=metrics
    rng=np.random.default_rng(42)
    random=evaluate(rng.random(baseline_scores.shape),validation['candidates'],validation['queries'])
    chronology=evaluate(np.broadcast_to(-np.arange(baseline_scores.shape[1],dtype=float),baseline_scores.shape),validation['candidates'],validation['queries'])
    selection={'schema':1,'winner':winner_name,'states':winner_states,'encoder':manifest['encoder'],
        'cache_manifest_sha256':digest(Path(cache)/'manifest.json'),'validation':winner_metrics,
        'training_consensus_no_positive_queries':int((~valid).sum()),'test_targets_used':False}
    torch.save(selection,output/'selected.pt');reloaded=torch.load(output/'selected.pt',map_location='cpu',weights_only=True)
    original=scores_for(winner_states,validation['x'][:3],validation['v'])
    replay=scores_for(reloaded['states'],validation['x'][:3],validation['v'])
    delta=float(np.max(np.abs(original-replay)))
    if delta>1e-6:raise ValueError('Saved adapter replay mismatch')
    report={'baseline':baseline,'random_control':random,'chronological_control':chronology,'runs':reports,
        'selected':winner_name,'selected_validation':winner_metrics,'seconds':time.monotonic()-start,
        'cache_manifest_sha256':selection['cache_manifest_sha256'],'selected_checkpoint_sha256':digest(output/'selected.pt'),
        'checkpoint_replay_max_abs_error':delta,'training_no_positive_queries':int((~valid).sum()),
        'test_targets_used':False,'limits':['Available licensed subset, not full official benchmark.',
        'Unknown encoder pretraining overlap; validation reused for stopping/selection.']}
    (output/'results.json').write_bytes((json.dumps(report,indent=2,allow_nan=False)+'\n').encode())
    print(json.dumps({'selected':winner_name,'validation':winner_metrics,'seconds':report['seconds']},indent=2));return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();train(a.cache,a.output)


def smoke_train(cache,output):
    """Two real-query optimization steps and weights-only reload; no benchmark claim."""
    torch.set_num_threads(2);torch.manual_seed(42)
    output=Path(output)
    if output.exists():raise ValueError('Refuse to overwrite smoke evidence')
    data,manifest=load_cache(cache,allow_smoke=True)
    item=data['train'];valid=item['positive'].any(dim=1)
    if not valid.any():raise ValueError('Real smoke queries have no consensus-positive moments')
    model=Adapter();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    losses=[]
    for epoch in range(2):
        optimizer.zero_grad();loss=multipositive_loss(model(item['x'][valid])@item['v'].T/.07,item['positive'][valid])
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise ValueError('Nonfinite gradients')
        optimizer.step();losses.append(float(loss.detach()))
    output.mkdir(parents=True,exist_ok=False)
    model.eval()
    with torch.inference_mode():before=model(data['val']['x'])
    torch.save(model.state_dict(),output/'smoke.pt');restored=Adapter()
    restored.load_state_dict(torch.load(output/'smoke.pt',weights_only=True,map_location='cpu'),strict=True);restored.eval()
    with torch.inference_mode():after=restored(data['val']['x'])
    delta=float((before-after).abs().max())
    if delta>1e-6:raise ValueError('Trained smoke checkpoint replay mismatch')
    result={'real_training_queries':int(valid.sum()),'epochs':2,'losses':losses,'checkpoint_replay_max_abs_error':delta,
            'checkpoint_sha256':digest(output/'smoke.pt'),'benchmark_or_test_score':False}
    (output/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2));return result
