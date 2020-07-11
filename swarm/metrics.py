__author__ = "Aidan Morrison <aidandmorrison@gmail.com>"

import numpy as np
from typing import Union, Optional

def mse_loss(swarm_ypreds: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    This is a function that calculates the root-mean sqared error for a prediction array, given true y.
    It averages along the all the point predictions in x for a set of predictions.
    Args:
        swarm_ypreds: A numpy array of predictions. This can have different numbers of dimensions,
            eg [bee, epoch, x_step] or [epoch, x_step], so long as the final dimension represents the
            predictions at different x values, and has same length as y.
        y: The true values predictions. A 1-d array.

    Returns: A numpy array of dimension one less than the input `swarm_ypreds`, with the evaluated root-mean-squared
    loss across all predictions.
    """
    dims = len(swarm_ypreds.shape)
    squares = np.square(np.subtract(swarm_ypreds, y))
    losses = np.apply_along_axis(np.mean, dims-1, squares)
    return losses

def iteration_threshold(epoch_scores: Union[np.ndarray, list],
                        threshold: float,
                        criterion: str = "first",
                        direction: str = "below") -> Optional[int]:
    """
    This is a function intended to find when a certain score of interest is achieved during a training process.
    A single vector of scores representing the assessment of the whole swarm at a given epoch/batch (ie iteration)
    is required.

    In a typical case, you might want to find when a loss is below a suitable threshold, eg .05.
    An alternative might be that you want to find when the range of standard deviations at all x values has
    falled below a threshold. (This could indicate knots/nodes eliminated.) Or the median/mean/max standard deviation
    has fallen below a threshold. (This would indicate a level of coherence within the swarm.)

    This function also allows you to set a criterion, based on whether it should return the first epoch/batch at which
    the threshold was crossed, or the first time it was crossed and never thereafter exceeded. (I.e. always remains
    below that threshold in all tested epochs/batches.)
    Args:
        epoch_scores: A 1-d Numpy array or list of floats representing the scores at each epoch/batch.
        threshold: The nominal value you want the score to first get below or above
        criterion: "first" or "always". Whether you want to return the epoch at which this threshold is first crossed, or
            always remains below/above (defaults to first)
        direction: "above" or "below". Choose whether you want get below or above the threshold (defaults below)

    Returns: An integer specifying which epoch/batch meets this criterion, or None if never met

    """
    # First check for suitable inputs. I wish there was a neater way of doing this.
    valid_criteria = {"first", "always"}
    valid_directions = {"above", "below"}
    if criterion not in valid_criteria:
        raise ValueError("argument `criterion` must be in one of %r." % valid_criteria)
    if direction not in valid_directions:
        raise ValueError("argument `direcction` must be in one of %r." % valid_directions)

    # Convert to numpy for convenience
    if type(epoch_scores) == list:
        epoch_scores = np.array(epoch_scores)

    # Create boolean mask indicating whether crossed above or below
    if direction == "above":
        boolean = epoch_scores >= threshold
    elif direction == "below":
        boolean = epoch_scores <= threshold

    # If criterion never met, return None, print a text warning. (could consider incorporating warning module?)
    if np.sum(boolean) == 0:
        return None
    else:
        if criterion == "first":
            return np.min(np.where(boolean))
        elif criterion == "always":
            diff = np.diff(boolean)
            return np.max(np.where(diff)) + 1

def iteration_threshold_ratio(epoch_scores: Union[np.ndarray, list],
                     threshold: float,
                     criterion: str = "first",
                     direction: str = "below",
                     reference_iteration: int = 0) -> Optional[int]:
    """
    This function is a variation on `iteration_threshold` which instead identifies the iteration after which
    the metric reaches a threshold varation from the same metric at some reference point. For different regimes of target functions the nominal evaluation score
    may vary widely widely in terms of magnitude. So the threshold 0.05 can be used to indicate that the score reaches
    5% of whatever the score was at the first epoch/batch. For ascending scores, a value greater than one may be
    helpful. i.e. 20 would indicate improving starting value 20 times over.

    Args:
        epoch_scores: A 1-d Numpy array or list of floats representing the scores at each epoch/batch.
        threshold: The ratio threshold that you're looking for. Eg 0.5 for half the loss of the reference iteration.
        criterion: "first" or "always". Whether you want to return the epoch at which this threshold is first crossed, or
            always remains below/above (defaults to first)
        direction: "above" or "below". Choose whether you want get below or above the threshold (defaults below)
        reference_iteration: Integer for the epoch/batch at which you want to set your reference to be improved upon.
            Defaults to 0. Ignored if threshold_type is "nominal".

    Returns: An integer specifying which epoch/batch meets this criterion, or None if never met

    """

    # Convert to numpy for convenience
    if type(epoch_scores) == list:
        epoch_scores = np.array(epoch_scores)

    if np.min(epoch_scores) < 0 and np.max(epoch_scores) > 0:
        raise ValueError("These scores cross over zero, ratios cannot be used. "
                         "Specify nominal threshold in `iteration_threshold` instead.")

    reference_score = epoch_scores[reference_iteration]
    used_threshold = reference_score * threshold

    out = iteration_threshold(epoch_scores, used_threshold, criterion, direction)
    return out
