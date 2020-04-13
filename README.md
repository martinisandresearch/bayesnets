# Bayesnets

Some experiments and visualisations into Neural Nets inspired by Bayesian thinking.

## Usage
Use the `./animate_training.py` script for your experiments. You can specify depth (hidden layers)
and width, even functions at the commandline.
`./animate_training.py --help` should give you some guidance on what can be done.


```bash
./animate_training.py -h 3 -w 10  --func exp 
```

Most of the time is spent in generating the animation ~ 30s for the default settings

Config in code
1. Loss defaults to  MSE
2. Optimiser defaults to SGD
3. Custom nets - just change the `net =` in the loop of `animate_training`


Code changes necessary
1. Custom domain - x is assumed to be linearly spaced and is a built in assumption


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

