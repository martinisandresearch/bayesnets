# Bayesnets

[![GitHub license](https://img.shields.io/github/license/nayyarv/bayesnets.svg)](https://github.com/nayyarv/bayesnets/blob/master/LICENSE)
![Tests](https://github.com/nayyarv/bayesnets/workflows/Tests/badge.svg)

Some experiments and visualisations into Neural Nets inspired by Bayesian thinking.

See our [Contributing Guide](CONTRIBUTING.md) for an overview on the structure of this project + guidelines on doing things.

## Nomenclature

 - Bee : A neural net trainign sequence
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


## Installation

If you use a venv, adapt as necessary. 
```bash
pip3 install --user -r requirements.txt
```

### For the Animation Backend (FFMPEG)

```bash
# OS X
brew install ffmpeg

# Ubuntu
sudo apt install ffmpeg
```

