import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
import animate_training
from swarm import activations
from swarm import networks
from swarm import regimes
import pendulum
import numpy as np
import pandas as pd



def run_sim(hidden = 3, width = 10, swarm_size = 20, nepoch = 200, xdomain = [-3,3], density = 10, funcname = 'sin', lr = 0.002, momentum = 0.9):
    params = {
        'hidden': hidden,
        'width': width,
        'swarm_size': swarm_size,
        'nepoch': nepoch,
        'xdomain': str(xdomain[0]) + '_' + str(xdomain[1]),
        'density': density,
        'funcname': funcname,
        'lr': lr,
        'momentum': momentum
    }
    data_list = []
    loss_list = []
    train = animate_training.Trainer.from_domain(funcname, xdomain, density)
    train.optimkwargs["lr"] = lr
    train.optimkwargs["momentum"] = momentum
    xd = train.xt.detach().numpy()
    yd = train.yt.detach().numpy()
    print("Starting training")
    tr_start = pendulum.now()
    for i in range(swarm_size):
        net = networks.flat_net(hidden, width)
        data, loss = train.get_training_results(net, nepoch)
        if np.any(np.isnan(loss.numpy())):
            raise RuntimeError(f"Nan loss found, drop lr. Currently lr={lr}")
        data_list.append(data.numpy())
        loss_list.append(loss.numpy())
    tm = pendulum.now() - tr_start
    print("Finished training in {}".format(tm.in_words()))
    return params, data_list, loss_list, xd, yd

def add_anindex(an_array, anindex):
    return np.c_[an_array, np.array([anindex for a in range(len(an_array))])]

def flatten_sim(data_list, loss_list, xd, yd):
    xy = np.c_[xd, yd]
    data_list = [add_anindex(data_list[a], a) for a in range(len(data_list))]
    flat_data = data_list[0]
    for i in range(1, len(data_list)):
        flat_data = np.vstack((flat_data, data_list[i]))
    loss_list = [add_anindex(loss_list[a], a) for a in range(len(loss_list))]
    flat_loss = loss_list[0]
    for i in range(1, len(loss_list)):
        flat_loss = np.vstack((flat_loss, loss_list[i]))
    
    return flat_data, flat_loss, xy

def run_save_sim(unique_string, dest_dir, param_list = None,):
    if param_list == None:
        param_list = [{
        'hidden': 2,
        'width': 10,
        'swarm_size': 10,
        'nepoch': 400,
        'xdomain': [-3,3],
        'density': 10,
        'funcname': 'sin',
        'lr': 0.002,
        'momentum': 0.9
    }]
    
    counter = 0
    for this_params in param_list:
        params, data_list, loss_list, xd, yd = run_sim(
            this_params['hidden'],
            this_params['width'],
            this_params['swarm_size'],
            this_params['nepoch'],
            this_params['xdomain'],
            this_params['density'],
            this_params['funcname'],
            this_params['lr'],
            this_params['momentum']
        )
        flat_data, flat_loss, xy = flatten_sim(data_list, loss_list, xd, yd)
        sim_sig = np.random.randint(9999999999)
        params['sim_sig']= sim_sig
        #params['simex_sig'] = float(str(sim_sig) + '.' + str(ex_sig))
        flat_data = add_anindex(flat_data, sim_sig)
        flat_loss = add_anindex(flat_loss, sim_sig)
        xy = add_anindex(xy, sim_sig)
        if counter == 0:
            final_data = flat_data
            final_loss = flat_loss
            final_xy = xy
            param_list = [params]
        else:
            final_data = np.vstack([final_data, flat_data])
            final_loss = np.vstack([final_loss, flat_loss])
            final_xy = np.vstack([final_xy, xy])
            param_list.append(params)
        print("finished sim"  + str(counter))
        counter += 1
        
    base_string = dest_dir + "/sim_" + unique_string
    np.savetxt(base_string + "_data" + ".csv", final_data, delimiter=",")
    np.savetxt(base_string + "_loss" + ".csv", final_loss, delimiter=",")
    np.savetxt(base_string + "_xy" + ".csv", final_xy, delimiter=",")
    final_params = pd.DataFrame(param_list)
    final_params.to_csv(base_string + "_params" + ".csv", index = False)


        
        
                        
            
    



    
    
    
    