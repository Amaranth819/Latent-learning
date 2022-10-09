from worldmodel_a2c import WorldModelA2C
from torchviz import make_dot
import torch

bs = 3
seq_len = 10
obs_dim = 44
act_dim = 17
state_dim = 128
hidden_dim = 256
o_curr = torch.zeros((bs, obs_dim))
a_prev = torch.zeros((bs, act_dim))
h_prev = torch.zeros((bs, state_dim))
s_prev = torch.zeros((bs, state_dim))
model = WorldModelA2C(obs_dim = 44, act_dim = 17, state_dim = 128, hidden_dim = 256)
posterior, a, a_logprob, v, (h_curr, s_curr) = model.step(o_curr, a_prev, h_prev, s_prev)

make_dot(h_curr.mean(), params=dict(model.named_parameters()), show_attrs=True, show_saved=True).render("rnn_torchviz", format="png")