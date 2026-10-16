"""Prespecified same-video contrastive follow-up; no held-out test access."""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from src.models.moment_training import Adapter, digest, load_cache, scores_for
from src.models.moments import evaluate

SEEDS = [42, 1337, 2026]
LAMBDAS = [0, .5, 1, 2]
ALPHAS = [.25, .5, .75, 1.]
MANIFEST = '8b97485f9dcc1ad243103b0cc4298c82cc3f20ceafa52cad863d6d050f53a3df'


def conditional_loss(scores, positives, allowed):
    """Negative log positive probability mass conditional on allowed candidates."""
    if scores.ndim != 2 or scores.numel() == 0 or scores.shape != positives.shape or scores.shape != allowed.shape:
        raise ValueError('Nonempty aligned score and mask matrices required')
    if positives.dtype != torch.bool or allowed.dtype != torch.bool:
        raise ValueError('Boolean masks required')
    if not torch.isfinite(scores).all() or not positives.any(dim=1).all() or (positives & ~allowed).any():
        raise ValueError('Finite scores and positive subset of allowed candidates required')
    denominator = torch.logsumexp(scores.masked_fill(~allowed, -torch.inf), dim=1)
    numerator = torch.logsumexp(scores.masked_fill(~positives, -torch.inf), dim=1)
    return (denominator - numerator).mean()


def same_video_mask(queries, candidates):
    return torch.tensor([[q['video'] == c['video'] for c in candidates] for q in queries], dtype=torch.bool)


def score_policy(bundle, x, v):
    """Exact convex score mixture, preserving equal-seed cosine averaging."""
    alpha = bundle.get('adapted_score_weight', 1. if bundle.get('states') else 0.)
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('Invalid adapted score weight')
    states = bundle.get('states')
    if bundle.get('schema') != 1 or not isinstance(states, list) or len(states) not in (0, 3):
        raise ValueError('Expected schema1 with zero or three adapter states')
    if not states and alpha != 0:
        raise ValueError('Baseline cannot have positive adapted weight')
    for value in (x, v):
        if value.ndim != 2 or value.shape[1] != 512 or not len(value) or not torch.isfinite(value).all():
            raise ValueError('Finite nonempty 512-dimensional features required')
        if not torch.allclose(value.norm(dim=1), torch.ones(len(value), device=value.device), atol=1e-4):
            raise ValueError('Unit features required')
    for state in states:
        if any(not torch.isfinite(t).all() for t in state.values()):
            raise ValueError('Nonfinite adapter state')
    frozen = scores_for([], x, v)
    result = frozen if alpha == 0 else (1-alpha)*frozen + alpha*scores_for(states, x, v)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite policy scores')
    return result


def checked_cache(cache, protocol):
    p = json.loads(Path(protocol).read_text(encoding='utf-8'))
    if p['source_manifest_sha256'] != MANIFEST or digest(Path(cache)/'manifest.json') != MANIFEST:
        raise ValueError('Protocol/source manifest mismatch')
    if p['local_loss_weights'] != LAMBDAS or p['seeds'] != SEEDS or p['adapted_score_weights'] != ALPHAS:
        raise ValueError('Protocol candidate grid mismatch')
    if p['optimizer'] != {'name':'AdamW','lr':.0003,'weight_decay':.01,'temperature':.07,'batch':64}:
        raise ValueError('Protocol optimizer mismatch')
    if p['epochs'] != {'maximum':50,'minimum':5,'patience':8,'selection':'per-seed validation primary; earliest exact ties'}:
        raise ValueError('Protocol stopping mismatch')
    if p['promotion'] != {'minimum_absolute_primary_gain_over_frozen_clip':.01,'maximum_within_video_miou_drop':.01}:
        raise ValueError('Protocol promotion mismatch')
    data, manifest = load_cache(cache)
    return data, manifest, p


def smoke(cache, output, protocol):
    torch.set_num_threads(2); torch.manual_seed(42)
    data, _, _ = checked_cache(cache, protocol)
    out = Path(output)
    if out.exists(): raise ValueError('Fresh output required')
    item = data['train']; valid = item['positive'].any(1)
    indices = valid.nonzero().flatten()[:8]
    x = item['x'][indices]; positive = item['positive'][indices]
    local = same_video_mask(item['queries'], item['candidates'])[indices]
    model = Adapter(); opt = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.01)
    losses = []
    for _ in range(2):
        opt.zero_grad(); logits = model(x)@item['v'].T/.07
        loss = conditional_loss(logits, positive, torch.ones_like(positive)) + conditional_loss(logits, positive, local)
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise ValueError('Nonfinite gradient')
        opt.step(); losses.append(float(loss.detach()))
    state = {k:v.detach().clone() for k,v in model.state_dict().items()}
    bundle = {'schema':1, 'states':[state,state,state], 'adapted_score_weight':.5}
    out.mkdir(parents=True); torch.save(bundle, out/'smoke.pt')
    replay = torch.load(out/'smoke.pt', weights_only=True, map_location='cpu')
    delta = float(np.max(np.abs(score_policy(bundle,x,item['v'])-score_policy(replay,x,item['v']))))
    result = {'queries':len(indices),'losses':losses,'replay_max_abs_error':delta,'test_access':False}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    if delta > 1e-6: raise ValueError('Replay mismatch')
    return result


def train(cache, output, protocol):
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    data, manifest, p = checked_cache(cache, protocol)
    out = Path(output)
    if out.exists(): raise ValueError('Fresh output required')
    out.mkdir(parents=True)
    t = data['train']; val = data['val']; valid = t['positive'].any(1)
    x = t['x'][valid]; positive = t['positive'][valid]; v = t['v']
    local = same_video_mask(t['queries'],t['candidates'])[valid]
    baseline = evaluate(scores_for([],val['x'],val['v']),val['candidates'],val['queries'])
    for key, expected in p['baseline'].items():
        if abs(baseline[key]-expected)>1e-12: raise ValueError('Historical baseline mismatch')
    reports=[]; policies=[]; best=-1.; winner=[]; winner_alpha=0.; winner_name='frozen_clip'; winner_metrics=baseline
    start=time.monotonic()
    for weight in LAMBDAS:
        states=[]; seeds=[]
        for seed in SEEDS:
            torch.manual_seed(seed); rng=torch.Generator().manual_seed(seed)
            model=Adapter(); opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01)
            peak=-1.; peak_epoch=0; state=None; history=[]
            for epoch in range(1,51):
                model.train(); total=0.
                for ix in torch.randperm(len(x),generator=rng).split(64):
                    opt.zero_grad(); logits=model(x[ix])@v.T/.07
                    loss=conditional_loss(logits,positive[ix],torch.ones_like(positive[ix]))
                    if weight: loss=loss+weight*conditional_loss(logits,positive[ix],local[ix])
                    loss.backward()
                    if not torch.isfinite(loss) or not all(torch.isfinite(a.grad).all() for a in model.parameters() if a.grad is not None):
                        raise ValueError('Nonfinite training')
                    opt.step(); total+=float(loss.detach())*len(ix)
                model.eval()
                with torch.inference_mode(): ranked=(model(val['x'])@val['v'].T).argmax(1)
                primary=float(val['positive'][torch.arange(len(ranked)),ranked].float().mean())
                row={'epoch':epoch,'train_loss':total/len(x),'validation_corpus_consensus_r1_iou50':primary};history.append(row)
                print(json.dumps({'lambda':weight,'seed':seed,**row}),flush=True)
                if primary>peak:
                    peak=primary;peak_epoch=epoch;state={k:a.detach().clone() for k,a in model.state_dict().items()}
                if epoch>=5 and epoch-peak_epoch>=8:break
            states.append(state); torch.save({'state':state,'lambda':weight,'seed':seed,'best_epoch':peak_epoch},out/f'lambda{weight}-{seed}.pt')
            seeds.append({'seed':seed,'best_epoch':peak_epoch,'history':history})
        reports.append({'lambda':weight,'seeds':seeds})
        for alpha in ALPHAS:
            bundle={'schema':1,'states':states,'adapted_score_weight':alpha}
            metrics=evaluate(score_policy(bundle,val['x'],val['v']),val['candidates'],val['queries'])
            primary=metrics['corpus_consensus_r1_iou50']
            eligible=primary-baseline['corpus_consensus_r1_iou50']>=.01-1e-12 and metrics['within_video_official_miou']>=baseline['within_video_official_miou']-.01-1e-12
            policies.append({'lambda':weight,'adapted_score_weight':alpha,'validation':metrics,'promotion_eligible':eligible})
            if eligible and primary>best:
                best=primary;winner=states;winner_alpha=alpha;winner_name=f'lambda{weight}-alpha{alpha}';winner_metrics=metrics
    bundle={'schema':1,'winner':winner_name,'states':winner,'adapted_score_weight':winner_alpha,'encoder':manifest['encoder'],
            'cache_manifest_sha256':MANIFEST,'validation':winner_metrics,'training_consensus_no_positive_queries':int((~valid).sum()),'test_targets_used':False}
    torch.save(bundle,out/'selected.pt'); replay=torch.load(out/'selected.pt',weights_only=True,map_location='cpu')
    delta=float(np.max(np.abs(score_policy(bundle,val['x'][:3],val['v'])-score_policy(replay,val['x'][:3],val['v']))))
    if delta>1e-6:raise ValueError('Saved policy replay mismatch')
    result={'baseline':baseline,'runs':reports,'policies':policies,'selected':winner_name,'selected_validation':winner_metrics,
            'seconds':time.monotonic()-start,'torch_version':str(torch.__version__),'threads':2,'protocol_sha256':digest(protocol),
            'cache_manifest_sha256':MANIFEST,'selected_checkpoint_sha256':digest(out/'selected.pt'),'checkpoint_replay_max_abs_error':delta,'test_targets_used':False}
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cache',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--protocol',default='docs/localization-v2-protocol.json');parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();print(json.dumps((smoke if args.smoke else train)(args.cache,args.output,args.protocol),indent=2))
