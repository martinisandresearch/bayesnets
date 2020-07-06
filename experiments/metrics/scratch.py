import torch
from torch import nn
import pendulum
from swarm import core, activations, animator, networks, regimes
from experiments.goofy import utils
import numpy as np
from matplotlib import pyplot as plt
from swarm import metrics

#%%
x = torch.linspace(-3.5, 3.5, 21)
y = torch.sin(x)
h, w = 1, 10
nepoch = 60
bee_trainer = regimes.make_bee(regimes.default_train, x, y, h, w, num_epochs = nepoch, lr=0.01, momentum=0.94)
results = core.swarm_train(bee_trainer, num_bees=10, fields="ypred,loss", seed=10)

#%%
losses = results['loss']
ypreds = results['ypred']


#%%
# This is just getting the point predictions
point_means = np.apply_along_axis(np.mean, 0, ypreds)
point_maxs = np.apply_along_axis(np.max, 0, ypreds)
point_mins = np.apply_along_axis(np.min, 0, ypreds)
point_meds = np.apply_along_axis(np.median, 0, ypreds)
point_sds = np.apply_along_axis(np.std, 0, ypreds)
point_ranges = point_maxs - point_mins

#%%
out = np.square(np.subtract(point_means, y.numpy()))
loss = np.apply_along_axis(np.mean, 1, out)

#%%
out0 = np.square(np.subtract(ypreds[0], y.numpy()))
loss0 = np.apply_along_axis(np.mean, 1, out0)

#%%
mean_point_preds = metrics.summarise_across(ypreds)
loss_mean_preds = metrics.rms_error(mean_point_preds, y)
sd_point_preds = metrics.summarise_across(ypreds, "std")
range_sd_point_preds = metrics.summarise_across(sd_point_preds, "range", 'x')
#%%
plt.plot(loss_mean_preds)
plt.show()

#%%
reverse_scores = 1/np.array(range_sd_point_preds)
plt.plot(reverse_scores/reverse_scores[3])
plt.show()
#%%
epoch = metrics.iteration_finder(reverse_scores, 1.1, "nominal", "always", "above", 3)
#epoch = metrics.iteration_finder(range_sd_point_preds, 0.5, "ratio", "first", "below")