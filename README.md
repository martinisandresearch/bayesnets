# Bayesnets

Some experiments and visualisations into Neural Nets inspired by Bayesian thinking.

## Usage
Use the `./animate_training.py` script for your experiments. You can specify depth (hidden layers)
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

Additional config.
1. Loss defaults to  MSE, specify in Trainer
2. Optimiser defaults to SGD, specify in Trainer
3. Custom nets - just change the `net =` in the loop of `animate_training`
4. Custom domain - call `Trainer(func, xt)` directly with your custom xt


## Installation

If you use a venv, adapt as necessary. 
```bash
pip3 install --user -r requirements.txt
```

### For the Animation 

```bash
# OS X
brew install ffmpeg

# Ubuntu
sudo apt install ffmpeg
```

