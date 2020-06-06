# Contributing

## Folder Structure

The basic idea here is that we want to balance between flexibility and stability while ensuring anything we do produce is easily understood, used and modified by another person. We basically do this by having a core library, `swarm` that can be relied upon and an `experiments` dir that builds on that base in any direction it feels.

1. `swarm` contains all the core code that we rely upon - any changes here should be done via a Merge Request
    - `tests` are checks for `swarm`'s capabilities and should be treated as an extension of `swarm`
    - conceptually any code shared across experiments should go into swarm

2. `experiments` is where the research goes - this is where we can test our ideas in self-contained folders and generate animations + writeups on the results. Experiments is used as a way to share code/ideas rather than to produce libraries and reusable code- anything good should be merged into swarm and a deletion of an experiment should have no flow on effects.

3. `sample` is a place to put animations/notebooks that give a sense of what we're doing here. It's a non critical folder, but there is value in having some sort of minimal examples to share instantly.
    - There is a potential to run a folder like this to produce the animations/writeups in a nice format.

## Suggested Git Workflow

General Philosophies:

1. Many small changes are easier than few big changes.
2. Notebooks are not reviewable and it is discouraged as a way of collaborating. They work as finished products and places to try ideas quickly, rather than something that can be shared and worked collaboratively.
    - Colaboratory provides a way to work on notebooks together, but this doesn't work well for git

Specific:

1. Generally avoid pushing to master unless you're absolutely sure. It's fraught with peril and is a point of commonality for everyone else so making changes to master should have some overview, preferably through a PR.
2. Separate out your changes to `swarm` and changes to `experiments` as much as possible. `swarm` as a point of common use requires more stability and thus will have a more conservative approach to PR - it should be necessary or a significant Qol change rather than a whatever you feel like.
3. Experiments should work on their own branch and merge into master occasionally. My preferred choice for this is to do this via a PR - it'll do the merging cleanly for you. This does mean you need to reset your remote branch forcibly after a merge though, but I think it'll be a good way to ensure that what's in master is always working correctly and keeps main history clean.
4. Use your name in the experiment branch i.e. `varun/dropout` or `ben/activations` so it doesn't get deleted by accident.

## Setting up Python

Swarm needs to be on your `PYTHONPATH` to be importable. I'd suggest using PyCharm since it does the paths for you automatically. However if you'd like to be a bit more agnostic, try [`direnv`](https://direnv.net/) for this. `brew install direnv` + and follow instructions to get it working.

And copy this into `.envrc ` in the project.

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
# if you use a venv do this too
# source venv/bin/activate
```
(You will need to run `direnv allow` for it to work - you should see an error otherwise)
This will execute every time you `cd` into the project and no matter where you're executing code in that directory it will work. Additionally, it will undo itself when in another directory which is pretty great, especially if you're using a virtualenv.

## Running an Experiment with Swarm

Swarm primarily provides a core that runs experiments with the same seed for reproducibility and provides an easy way of recording neural net results without need to write a ton of code.

See the
- [simple_experiment](experiments/sample/simple_experiment.py) to get a sense of how to use it
- [intermediate_experiment](experiments/sample/intermediate_experiment.py) for another example
- [hive experiment](experiments/sample/hive_experiment.py) for doing sweeps across params

The basic idea is this:

1. Write a bee_trainer - a function that trains a neural net from start to finish and yields data of interest each epoch.
2. Make a dictionary of it's kwargs (if it needs any)
3. Use `swarm.core.swarm_train` to generate the results from the experiments.
4. Get a dictionary back which has each data collected as a `np.array` with the dims represented (bee, epoch, \*data)
5. Analyse the data or generate an animation from it!
