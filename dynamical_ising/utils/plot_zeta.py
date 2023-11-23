import matplotlib.pyplot as plt
from torchdiffeq import odeint_adjoint as odeint
import torch
from torch import nn
from math import exp, log
import numpy as np
import matplotlib as mpl
import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mu_ode_solve_ising import Data
from utils.weighted_model import AvgsODEFunc, ODEFunc 
from utils.plot_beta import *
if __name__ == "__main__":
    with torch.no_grad():
        device = "cpu"
        n_train = 37
        Ts = np.array([
        2.0,
        2.014157894736842,
        2.0283157894736843,
        2.042473684210526,
        2.0566315789473686,
        2.0707894736842105,
        2.084947368421053,
        2.099105263157895,
        2.1132631578947367,
        2.127421052631579,
        2.141578947368421,
        2.1557368421052634,
        2.1698947368421053,
        2.1840526315789477,
        2.1982105263157896,
        2.2123684210526315,
        2.226526315789474,
        2.240684210526316,
        2.254842105263158,
        #2.269,
                ])
        plt.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
            "font.size":11
         })
        dataset = "./dataset"

        funcs = nn.ModuleList([AvgsODEFunc(n_train,torch.zeros(n_train)) for _ in range(len(Ts))])
        x_mins=torch.zeros(len(Ts))
        y_mins=torch.zeros(len(Ts))
        fig,( ax,ax1) = plt.subplots(1, 2)
        fig.set_size_inches(w=4.7747, h=2.9841875/2*1.7)
        fig.tight_layout(pad=3.)
        cmap = mpl.cm.get_cmap('rainbow')
        rgba = cmap(np.linspace(0, 1, len(Ts)))
        times = torch.zeros(len(Ts))
        times_true = torch.zeros(len(Ts))
        for idx in range(len(Ts)):
                model_dir = "./dump/t_100000_ising_mu_%.3f_N_128"%Ts[idx]
                model_name = "weighted_mu_net.ckpt"
                model_path = os.path.join(model_dir,model_name)
                device = "cpu"
                ckpt = torch.load(model_path,map_location=torch.device('cpu'))
                funcs[idx].load_state_dict(ckpt['model'])
                funcs[idx].weights = ckpt['weights']
                x = torch.linspace(0,1.0,200).reshape(200,1)
                mu = funcs[idx].netto(x.to(device)).flatten().cpu()
                y = torch.cumsum(-1/200* mu ,dim=0)
                x_mins[idx] = x[torch.argmin(y)]
                y_mins[idx] = y[torch.argmin(y)]
                file_name =  "./dataset/N_128_T_%.4f_num_traj_1000t_max_100000.h5"%Ts[idx]
                t_max = 100000
                data = Data(file_name,device = device,t_max = t_max,every=1)
                pred_y = odeint(funcs[idx], data.x[0].to(device), data.time.to(device))
                for t in range(t_max-1):
                        if abs(  data.x[t]-    x_mins[idx])<0.01:
                                times_true[idx] = data.time[t]
                                break
                for t in range(t_max-1):
                        if abs(  pred_y[t]-    x_mins[idx])<0.01:
                                times[idx] = data.time[t]
                                break
                ax.plot( data.time[:t],pred_y[:t], color=rgba[ idx  ] )
                ax.plot( data.time[:t],data.x[:t], color=rgba[ idx  ] )
                ax.scatter(times[idx], x_mins[idx],color = "grey",s=5)
                ax.set_xlabel(r"$t$")
                ax.set_ylabel(r"$m(t)$")
                
                ax1.plot( torch.log(data.time[:t]),torch.log(pred_y[:t]), color=rgba[ idx  ] )
                ax1.plot( torch.log(data.time[:t]),torch.log(data.x[:t]), color=rgba[ idx  ] )
                ax1.scatter(torch.log(times[idx]),      torch.log(x_mins[idx]),color = "grey",s=5)
                ax1.set_xlabel(r"$t$")
                ax1.set_ylabel(r"$m(t)$")
                plt.savefig("./images/trajs.pdf")



