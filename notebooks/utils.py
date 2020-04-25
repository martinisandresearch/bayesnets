import torch
from torch import nn
from torch import optim
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

class Swarm_Simulator:
    def __init__(self):
        
        self.simple_funcs = {
            'exponential':torch.exp,
            'sinusoidal':torch.sin
        }
        
        self.debug = False
        
        self.params = {
            'domain_min': -1,
            'domain_max': 3,
            'num_points': 81,
            'function': 'exponential',
            'hidden': 1,
            'width': 5,
            'nepoch': 100,
            'lr': 0.002,
            'momentum': 0.9,
            'swarm_size': 20
            
        }
        
        self.simulations = []
        
        
        self.bees = [{'loss_list':[], 'pred_list': []} for i in range(self.params['swarm_size'])]
        
        self.xt = torch.linspace(self.params['domain_min'], self.params['domain_max'], self.params['num_points']).unsqueeze(-1)
        self.chosen_func = self.simple_funcs[self.params['function']]
        self.yt = self.chosen_func(self.xt)
        self.xd = self.xt.detach()
        self.yd = self.yt.detach()
        self.get_pred = self.make_predictor(self.xt, self.yt)
        
        
    def reset_swarm(self):
        self.bees = [{'loss_list':[], 'pred_list': []} for i in range(self.params['swarm_size'])]
        
    def rebuild_predictor(self):
        self.xt = torch.linspace(self.params['domain_min'], self.params['domain_max'], self.params['num_points']).unsqueeze(-1)
        self.chosen_func = self.simple_funcs[self.params['function']]
        self.yt = self.chosen_func(self.xt)
        self.xd = self.xt.detach()
        self.yd = self.yt.detach()
        self.get_pred = self.make_predictor(self.xt, self.yt)
        self.reset_swarm()
        

    def make_net(self, hidden_depth, width):
        assert hidden_depth >= 1
        yield nn.Linear(1, width)
        yield nn.ReLU()
        for i in range(hidden_depth-1):
            yield nn.Linear(width, width)
            yield nn.ReLU()
        yield nn.Linear(width, 1)
    
    def make_predictor(self, x, y):
        xt = x.clone()
        yt = y.clone()
        def get_pred(num, hidden = self.params['hidden'], width = self.params['width'], nepoch=self.params['nepoch'], lr=self.params['lr'], momentum=self.params['momentum'], debug=self.debug):
            net = nn.Sequential(*self.make_net(hidden, width))
            lossfunc = nn.MSELoss()
            optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

            og_loss = lossfunc(net(xt), yt)
            for epoch in range(nepoch):
                optimizer.zero_grad()
                ypred = net(xt)
                self.bees[num]['pred_list'].append(np.array(ypred.detach()))
                loss = lossfunc(ypred, yt)
                self.bees[num]['loss_list'].append(loss.item())
                if debug: print(epoch, loss)
                loss.backward()
                optimizer.step()
                
            if debug: print(f"First loss {og_loss} v final {loss}")
            return ypred.detach(), loss.item()
        return get_pred

    def run_sim(self, debug = False):
        
                  
        for num in range(len(self.bees)):
            yval, loss = self.get_pred(num)
            if np.isnan(loss):
                raise RuntimeError("Nan loss")
            if debug:
                print(num, loss)
        
        sim_dict = {
            'params':self.params.copy(), 
            'bees': self.bees.copy(), 
            'summ_stats': self.summ_swarm(self.bees),
            'xd': self.xd,
            'yd': self.yd
        }
        self.simulations.append(sim_dict)
        self.reset_swarm()
        
    def summ_swarm(self, bees):
        summ_stats = {}
        summ_stats['mean_final_loss'] = np.mean([b['loss_list'][-1] for b in bees])
        summ_stats['min_final_loss'] = np.min([b['loss_list'][-1] for b in bees])
        summ_stats['max_final_loss'] = np.max([b['loss_list'][-1] for b in bees])
        return summ_stats
                  
              
def plot_preds(sim):
    plt.title(f"Approximating {sim['params']['function']} with {sim['params']['hidden']} hidden layer, {sim['params']['width']} width. Average final loss {round(sim['summ_stats']['mean_final_loss'],5)}")
    plt.plot(sim['xd'], sim['yd'], ".", label="Actual")
    for i in range(len(sim['bees'])):
        plt.plot(sim['xd'],sim['bees'][i]['pred_list'][-1])
              
def plot_losses(sim):
    plt.title(f"Loss per epoch for {sim['params']['function']} with {sim['params']['hidden']} hidden layer, {sim['params']['width']} width")
    for i in range(len(sim['bees'])):
        plt.plot(sim['bees'][i]['loss_list'])