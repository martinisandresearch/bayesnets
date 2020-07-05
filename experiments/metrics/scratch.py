import torch
from torch import nn
import pendulum
from swarm import core, activations, animator, networks, regimes
from experiments.goofy import utils
import numpy as np

#%%
x = torch.linspace(-5, 5, 61)
y = torch.sin(x)
h, w = 1, 10
nepoch = 100
bee_trainer = regimes.make_bee(regimes.default_train, x, y, h, w, num_epochs = nepoch)
results = core.swarm_train(bee_trainer, num_bees=20, fields="ypred,loss", seed=10)

#%%
losses = results['loss']
ypreds = results['ypred']

#%%
len(losses)
np.min(losses[2])
