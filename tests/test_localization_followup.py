import numpy as np
import pytest
import torch
from src.models.localization_followup import conditional_loss, same_video_mask, score_policy
from src.models.moment_training import Adapter, scores_for


def test_conditional_probability_oracle_and_excluded_gradient():
    scores=torch.tensor([[0.,1.,2.,8.]],requires_grad=True)
    positives=torch.tensor([[True,False,True,False]])
    allowed=torch.tensor([[True,True,True,False]])
    result=conditional_loss(scores,positives,allowed)
    expected=-torch.log(torch.softmax(scores[0,:3],0)[[0,2]].sum())
    assert result.item()==pytest.approx(expected.item())
    result.backward()
    assert scores.grad[0,3]==0 and torch.isfinite(scores.grad).all()
    assert scores.grad[0,1]>0 and scores.grad[0,0]<0


def test_conditional_guards_and_singleton_zero():
    s=torch.tensor([[1.,2.]],requires_grad=True);p=torch.tensor([[True,False]])
    loss=conditional_loss(s,p,p);loss.backward()
    assert loss.item()==0 and torch.equal(s.grad,torch.zeros_like(s))
    for scores,positive,allowed in [(s,~p,p),(s,torch.zeros_like(p),p),(s*float('nan'),p,p),(s,p.float(),p),(s[:,:1],p,p)]:
        with pytest.raises(ValueError):conditional_loss(scores,positive,allowed)
    assert same_video_mask([{'video':'a'}],[{'video':'b'},{'video':'a'}]).tolist()==[[False,True]]


def test_exact_score_mixture_and_serialized_replay(tmp_path):
    torch.manual_seed(7)
    x=torch.nn.functional.normalize(torch.randn(3,512),dim=1);v=torch.nn.functional.normalize(torch.randn(5,512),dim=1)
    states=[]
    for _ in range(3):
        m=Adapter()
        with torch.no_grad():m.up.weight.normal_(std=.03)
        states.append(m.state_dict())
    b={'schema':1,'states':states,'adapted_score_weight':.25}
    expected=.75*scores_for([],x,v)+.25*scores_for(states,x,v)
    assert np.allclose(score_policy(b,x,v),expected,atol=1e-7)
    torch.save(b,tmp_path/'p.pt')
    restored=torch.load(tmp_path/'p.pt',weights_only=True)
    assert np.array_equal(score_policy(restored,x,v),score_policy(b,x,v))
    for alpha in [True,-.1,1.1,float('nan'),float('inf')]:
        with pytest.raises(ValueError):score_policy({**b,'adapted_score_weight':alpha},x,v)
    with pytest.raises(ValueError):score_policy({**b,'states':states[:1]},x,v)
    with pytest.raises(ValueError):score_policy({**b,'states':[]},x,v)
    with pytest.raises(ValueError):score_policy(b,x*2,v)
    assert np.array_equal(score_policy({'schema':1,'states':[]},x,v),scores_for([],x,v))
