__author__ = "Aidan Morrison <aidandmorrison@gmail.com>"

import numpy as np


def summarise_across(data:np.ndarray, func_string: str = 'mean', across: str = "bees") -> np.ndarray:
    """
    This function aggregates along the first dimension of a multidimensional array.
    In this library, it has two main use cases:

    1: This function returns a summary statistic regarding predictions at a particular x point,
    for a particular epoch.  Hence, it summarises the predictions made across different bees
    in a swarm.

    2: This function returns a summary statistic regarding the losses of different bees at each epoch.

    3. This function returns a summary statistic regarding of another summary statistic recorded at each
    point in the domain. In this case it should be used in with argument `across` set to `x`.

    Note: Could use a refactor to generalise so that any one-d function can be passed in
    as an argument. (didn't do this initially because the range concept was important, and
    I wanted this to be accessible along the same lines as mean, max, min etc.)

    Args:
        data: In use case 1 this should be a numpy array of predictions, as yeilded by core.swarm_train.
            It will have three dimensions, of structure [bee, epoch/batch, x_step].
            In use case 2 this should be a numpy array of losses, which can also be yeilded by core.swarm_train
            It will have two dimensions, of structure [bee, epoch].
            In use case 3 this should be a numpy array of summary statistics across bees at all points in domain x.
            It will have two dimensions, of structure [epoch/batch, x_step]
        func_string: A string indicating the type of summary statistic desired.
            Currently allowable includes 'min', 'max', 'mean', 'median', 'std', and 'range'
        across: Takes ony 'Indicate whether you'd like to summarise across the bees (as in use case 1 and 2),
            or summarise across the a value for each point in domain x.

    Returns: a numpy array of two dimensions, of structure [epoch/batch, x_step] that gives the
        summary evaluation for each point in x space for all the bees at that epoch of training.

    """
    valid_axes = {'bees', 'x'}
    if across not in valid_axes:
        raise ValueError("argument `across` must be one of %r." % valid_axes)

    # Create a simple dict of likely functions
    funcs = {
        "mean": np.mean,
        "std": np.std,
        "median": np.median,
        "min": np.min,
        "max": np.max
    }

    # Identify the suitable axis to summarise across
    if across == "bees":
        axis_choice = 0
    elif across == "x":
        axis_choice = len(data.shape) - 1

    # Create summary of data
    if func_string in funcs.keys():
        return np.apply_along_axis(funcs[func_string], axis_choice, data)
    elif func_string == 'range':
        point_maxs = np.apply_along_axis(np.max, axis_choice, data)
        point_mins = np.apply_along_axis(np.min, axis_choice, data)
        return point_maxs - point_mins
    else:
        raise ValueError("Function not in currently allowable of min', 'max', 'mean', 'median', 'std', and 'range'")


def rms_error(swarm_ypreds: np.ndarray, y: np.ndarray) -> np.ndarray:
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


def iteration_finder(epoch_scores: [np.ndarray, list],
                     threshold: float,
                     threshold_type: str = "nominal",
                     criterion: str = "first",
                     direction: str = "below",
                     reference_iteration: int = 0) -> [int, None]:
    """
    This is a function intended to find when a certain score of interest is achieved during a training process.
    A single vector of scores representing the assessment of the whole swarm at a given epoch/batch (ie iteration)
    is required.

    In a typical case, you might want to find when a loss is below a suitable threshold, eg .05.
    An alternative might be that you want to find when the range of standard deviations at all x values has
    falled below a threshold. (This could indicate knots/nodes eliminated.) Or the median/mean/max standard deviation
    has fallen below a threshold. (This would indicate a level of coherence within the swarm.)

    The threshold can have two different interpretations, either as a nominal value or a ratio relative to a reference
    value at a given iteration (often the first).  For different regimes of target functions the nominal evaluation score
    may vary widely widely in terms of magnitude. So the threshold 0.05 can be used to indicate that the score reaches
    5% of whatever the score was at the first epoch/batch. For ascending scores, a value greater than one may be
    helpful. i.e. 20 would indicate improving starting value 20 times over.

    This function also allows you to set a criterion, based on whether it should return the first epoch/batch at which
    the threshold was crossed, or the first time it was crossed and never thereafter exceeded. (I.e. always remains
    below that threshold in all tested epochs/batches.)
    Args:
        epoch_scores: A list or 1-d Numpy array of floats representing the scores at each epoch/catch.
        threshold: The nominal value you want the score to first get below or above
        threshold_type: "ratio" or "nominal". Indicate whether you would like to use the threshold as a nominal value, or a ratio that
            reflects the change from a reference_iteration. Defaults to "nominal".
        criterion: "first" or "always". Whether you want to return the epoch at which this threshold is first crossed, or
            always remains below/above (defaults to first)
        direction: "above" or "below". Choose whether you want get below or above the threshold (defaults below)
        reference_iteration: Integer for the epoch/batch at which you want to set your reference to be improved upon.
            Defaults to 0. Ignored if threshold_type is "nominal".

    Returns: An integer specifying which epoch/batch meets this criterion

    """
    # First check for suitable inputs. I wish there was a neater way of doing this.
    valid_threshold_types = {"nominal", "ratio"}
    valid_criteria = {"first", "always"}
    valid_directions = {"above", "below"}
    if threshold_type not in valid_threshold_types:
        raise ValueError("argument `threshold_type` must be one of %r." % valid_threshold_types)
    if criterion not in valid_criteria:
        raise ValueError("argument `criterion` must be in one of %r." % valid_criteria)
    if direction not in valid_directions:
        raise ValueError("argument `direcction` must be in one of %r." % valid_directions)

    # Convert to numpy for convenience
    if type(epoch_scores) == list:
        epoch_scores = np.array(epoch_scores)

    # Choose what sort of threshold to use, and obtain it
    if threshold_type == "ratio":
        reference_score = epoch_scores[reference_iteration]
        used_threshold = reference_score * threshold
    elif threshold_type == "nominal":
        used_threshold = threshold

    # Create boolean mask indicating whether crossed above or below
    if direction == "above":
        boolean = epoch_scores >= used_threshold
    elif direction == "below":
        boolean = epoch_scores <= used_threshold

    # If criterion never met, return None, print a text warning. (could consider incorporating warning module?)
    if np.sum(boolean) == 0:
        print("Warning: Criterion never met in training, returning None")
        return None
    else:
        if criterion == "first":
            return np.min(np.where(boolean))
        elif criterion == "always":
            diff = np.diff(boolean)
            return np.max(np.where(diff)) + 1
