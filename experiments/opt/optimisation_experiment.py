from swarm import core
from swarm import metrics
from swarm import regimes
from torch import nn
import torch
from hyperopt import hp, tpe, fmin, Trials
from hyperopt.pyll import scope
from swarm import activations
import numpy as np
import pandas as pd

def get_swarm_results(h, w, lr, mom, act, x, y):
    nepoch = 100
    num_bees = 50
    bee_trainer = regimes.make_bee(
        regimes.default_train, x, y, h, w, num_epochs=nepoch, lr=lr, momentum=mom, activation = act
    )
    results = core.swarm_train(bee_trainer, num_bees=num_bees, fields="ypred,loss", seed=10)
    return results

#%%
# Just a quick example to check
x = torch.linspace(-3.5, 3.5, 21)
y = torch.sin(x)
results = get_swarm_results(2, 10, 0.01, .9, nn.ReLU, x, y)
ypreds = results['ypred']
losses = results['loss']

#%%
space = {"h": scope.int(hp.quniform("h", 1, 5, 1)),
         "w": scope.int(hp.quniform("w", 1, 50, 1)),
         "lr": hp.uniform("lr", 0.0001, 0.2),
         "mom": hp.uniform("mom", 0.5, 0.995),
         "act": hp.choice("act", [nn.ReLU, nn.Tanh, nn.Tanhshrink, torch.nn.Softmin, torch.nn.GELU, torch.nn.CELU, torch.nn.Sigmoid])}

#%%
def objective(params):
    """The objective function that returns a single score (to minimise) for a swarm"""
    x = torch.linspace(-3.5, 3.5, 61)
    y = torch.sin(x)
    try:
        results = get_swarm_results(params['h'], params['w'], params['lr'], params['mom'], params['act'], x, y)
    except:
        return 200
    ypreds = results['ypred']
    losses = results['loss']
    y = y.numpy()
    mean_preds = np.mean(ypreds, axis=0)
    loss = metrics.mse_loss(mean_preds, y)
    epoch = metrics.iteration_threshold(loss, 0.005)
    if epoch is None:
        return 200
    else:
        return epoch

#%%
tpe_algo = tpe.suggest
tpe_trials = Trials()
tpe_best = fmin(fn=objective, space=space, algo=tpe_algo, trials=tpe_trials, max_evals=200)

#%%
def unlist(vals):
    return {key:val[0] for key, val in vals.items()}
def get_vals(atrial):
    return atrial['misc']['vals']
def map_acts(ind):
    act_map = ["ReLU", "Tanh", "Tanhshrink", "Softmin", "GELU", "CELU", "Sigmoid"]
    return act_map[ind]

#%%
trial_results = [unlist(get_vals(atrial)) for atrial in tpe_trials.trials]
trials_df = pd.DataFrame(trial_results)
trials_df['loss'] = [atrial['result']['loss'] for atrial in tpe_trials.trials]
trials_df.sort_values('loss', inplace=True)
trials_df['act'] = trials_df['act'].apply(map_acts)