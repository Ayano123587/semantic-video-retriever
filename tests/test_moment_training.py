import pytest
import torch
from src.models.moment_training import Adapter,multipositive_loss

def test_adapter_initially_identity_and_learns_real_gradients():
    torch.manual_seed(42);x=torch.nn.functional.normalize(torch.randn(4,512),dim=-1);model=Adapter()
    assert torch.allclose(model(x),x,atol=1e-6)
    value=model(x)@x.T
    loss=multipositive_loss(value/.07,torch.eye(4,dtype=torch.bool));loss.backward()
    assert torch.isfinite(model.up.weight.grad).all()
    assert float(model.up.weight.grad.abs().sum())>0

def test_multiple_positive_loss_matches_probability_mass_oracle():
    scores=torch.tensor([[0.,1.,2.]])
    positive=torch.tensor([[True,False,True]])
    actual=multipositive_loss(scores,positive)
    expected=-torch.log(torch.softmax(scores,dim=1)[0,[0,2]].sum())
    assert float(actual)==pytest.approx(float(expected))
    with pytest.raises(ValueError):multipositive_loss(scores,torch.zeros_like(positive))
