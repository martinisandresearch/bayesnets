# Bayesnets

[![GitHub license](https://img.shields.io/github/license/nayyarv/bayesnets.svg)](https://github.com/nayyarv/bayesnets/blob/master/LICENSE)
![Tests](https://github.com/nayyarv/bayesnets/workflows/Tests/badge.svg)
[![codecov](https://codecov.io/gh/martinisandresearch/bayesnets/branch/master/graph/badge.svg)](https://codecov.io/gh/martinisandresearch/bayesnets)

Some experiments and visualisations into Neural Nets inspired by Bayesian thinking.

See our [Contributing Guide](CONTRIBUTING.md) for an overview on the structure of this project + guidelines on doing things.

See our [Blog](https://martinisandresearch.github.io/bayesnets/intro.html) for more info on our experiments,
written in a way for others to consume.

## Nomenclature

 - Bee : A neural net training sequence
 - Swarm : A group of networks trained the same way, with the only difference defined by starting conditions
 - Hive : A set of swarms with some some training/initialisation parameter varied.


## Animation Usage
Use the `./animate_training.py` script for your a quick viz. You can specify depth (hidden layers)
and width, even functions at the commandline.
`./animate_training.py --help` should give you some guidance on what can be done.

Example usages
```bash
./animate_training.py -h 3 -w 10  --func exp
./animate_training.py -h 3 -w 10  --func sin -n 800 --xdomain 0:6.2
./animate_training.py -h 3 -w 10  --func exp --numtrains 3
./animate_training.py -h 3 -w 15  --func sin -n 800 --xdomain -6.1:6.2 --lr 0.004
```

Most of the time is spent in generating the animation ~ 30s for the default settings


## Getting Started


### Tooling

1. pre-commit - Follow, https://pre-commit.com/.
    - Preferred method is to install using brew or to system python
    - `pre-commit install` so it will run and `pre-commit run --all-files` will initialise everything
    - This runs some formatting and simple static checks on the code.
    - This simply manages git hooks and is optional

### Environment

If you use a venv, the `--user` flag won't be necessary
```bash
pip3 install --user -r requirements.txt
pip3 install --user -r requirements-dev.txt
```

### For the Animation Backend (FFMPEG)

```bash
# OS X
brew install ffmpeg

# Ubuntu
sudo apt install ffmpeg
```
