import numpy as np
import torch
from swarm import core, regimes
from swarm import metrics
import pytest

def get_simple_results():
    x = torch.linspace(-3.5, 3.5, 21)
    y = torch.sin(x)
    h, w = 1, 10
    nepoch = 60
    bee_trainer = regimes.make_bee(regimes.default_train, x, y, h, w, num_epochs=nepoch, lr=0.01, momentum=0.94)
    results = core.swarm_train(bee_trainer, num_bees=10, fields="ypred,loss", seed=10)
    return results

@pytest.fixture
def ypreds():
    results = get_simple_results()
    ypreds = results['ypred']
    return ypreds

@pytest.fixture
def losses():
    results = get_simple_results()
    losses = results['loss']
    return losses

@pytest.fixture
def y():
    x = torch.linspace(-3.5, 3.5, 21)
    y = torch.sin(x)
    return y.numpy()

def test_compare_averages(ypreds, losses, y):

    """This is a possible logical sequence of inquiry calling a range of metrics in sequence
    The logic behind the final comparisons can be illustrated easily by viewing the plot:
    plt.plot(derived_loss_median_bee)
    plt.plot(loss_mean_preds)
    plt.show()
    """
    # calculate the 'mean' prediction of the swarm, at each x point, at each epoch.
    # This is done by averaging over the first axis, which represents the bees
    mean_preds = np.mean(ypreds, axis=0)
    # This should have shape (iterations,x_steps)
    assert mean_preds.shape == (60, 21)

    # Calculate the mean loss of this new derived prediction over the whole of x
    loss_mean_preds = metrics.mse_loss(mean_preds, y)
    # This should have shape (iterations,)
    assert loss_mean_preds.shape == (60,)

    # Find the loss of the median 'bee' in terms of loss at each step as assessed in training.
    # This is of interest as a hypothetical 'if we'd only trained once' likely outcome
    loss_median_bee = np.median(losses, axis=0)
    # This should have shape (iterations,)
    assert loss_median_bee.shape == (60,)

    # To check that our loss function is similar to training, we can calculate the loss on each bee ourselves
    losses_all_bees = metrics.mse_loss(ypreds, y)
    # This should have shape (bees, iterations)
    assert losses_all_bees.shape == (10, 60)
    # And then find the median bee at each step
    derived_loss_median_bee = np.median(losses_all_bees, axis=0)
    # Which should have the same shape, (iterations,)
    assert derived_loss_median_bee.shape == loss_median_bee.shape
    # And be very close to the same
    assert np.max(np.abs(derived_loss_median_bee - loss_median_bee)) < 0.00001

    # Find at which epoch the loss of the prediction based on the mean of the swarm reaches 0.3
    mean_pred_epoch_nom = metrics.iteration_threshold(loss_mean_preds, 0.3)
    # Find at which epoch the median bee would reach loss of 0.3
    median_bee_epoch_nom = metrics.iteration_threshold(derived_loss_median_bee, 0.3)
    # In this case, the prediction based on mean of the entire swarm reaches this threshold faster
    assert mean_pred_epoch_nom < median_bee_epoch_nom

    # Find at which epoch the loss of the prediction based on the mean of the swarm reaches 0.3 it's initial loss
    mean_pred_epoch_ratio = metrics.iteration_threshold_ratio(loss_mean_preds, 0.3)
    # Find at which epoch the median bee has reached a loss 0.3 of the loss of the median initial bee
    median_bee_epoch_ratio = metrics.iteration_threshold_ratio(derived_loss_median_bee, 0.3)
    # In this case the median reaches the proportional improvement faster, since the mean prediction begins better
    assert mean_pred_epoch_ratio > median_bee_epoch_ratio











