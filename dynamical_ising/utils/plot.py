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
from mu_ode_solve_ising import Data, ODEFunc
#from utils.weighted_model import AvgsODEFunc, ODEFunc 
def plot_dyn(T,train1,train2,device,dataset,file_name):
        """
        plot the true dynamics and the one retrieved by the network for two different training train_1 and rain2
        """
        plt.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
            "font.size":11
         })
        data = dataset#Data(file_name,device = device)
        model_dir = "./dump/ising_mu_%.3f_N_128"%T
        model_dir = "./dump/t_100000_ising_mu_%.3f_N_128"%T
        model_name = "global_step_%d.ckpt"%train1
        model_path = os.path.join(model_dir,model_name)
        mult = 1.0
        device = "cpu"
        func = ODEFunc()
        ckpt = torch.load(model_path,map_location=torch.device('cpu'))
        func.load_state_dict(ckpt['model'])
        x = torch.linspace(0,1.0,200).reshape(200,1)
        mu = func.netto(x.to(device)).flatten().cpu()
        y = torch.cumsum(-1/200* mu ,dim=0)
        true_y = data.x
        pred_y = odeint(func, data.x[0].to(device), data.time.to(device))
        print(pred_y.shape)
        fig, (ax1,ax2) = plt.subplots(1, 2)
        fig.set_size_inches(w=4.7747/1.5, h=2.9841875/2*1.5)
        fig.tight_layout(pad=2.)
        ax1.plot(data.time.to(device).flatten(),true_y.flatten(),label = "Data")
        ax1.plot(data.time.to(device).flatten(),pred_y.flatten()  ,label = "Network",linestyle="dashed")
        ax1.set_xlabel(r"$t$")
        ax1.set_ylabel(r"$\bar{m}$")
        ax1.legend()
        ax1.text(-0.1, 1.1, "(a)", transform=ax1.transAxes)

        model_name = "global_step_%d.ckpt"%train2
        model_path = os.path.join(model_dir,model_name)
        mult = 1.0
        device = "cpu"
        func = ODEFunc()
        ckpt = torch.load(model_path,map_location=torch.device('cpu'))
        func.load_state_dict(ckpt['model'])
        x = torch.linspace(0,1.0,200).reshape(200,1)
        mu = func.netto(x.to(device)).flatten().cpu()
        y = torch.cumsum(-1/200* mu ,dim=0)
        true_y = data.x
        pred_y = odeint(func, data.x[0].to(device), data.time.to(device))

        ax2.plot(data.time.to(device).flatten(),true_y.flatten(),label = "Data")
        ax2.plot(data.time.to(device).flatten(),pred_y.flatten()  ,label = "Network",linestyle="dashed")
        ax2.set_xlabel(r"$t$")
        ax2.set_ylabel(r"$\bar{m}$")
        #ax2.legend()
        ax2.text(-0.1, 1.1, "(b)", transform=ax2.transAxes)
        plt.savefig("./images/dyn_Ising.pdf")
        plt.close()

        t_max = 500
        fig, (ax) = plt.subplots(1, 1)
        fig.set_size_inches(w=4.7747, h=2.9841875/2*1.5)
        fig.tight_layout(pad=3.)
        x_dot = (-data.x.flatten()[:-1]+data.x.flatten()[1:])/0.01
        ax.plot(data.x.flatten().flip((0)[0::100] ),x_dot.flatten().flip((0)[0::100]  ),label = "Data")
        x = torch.linspace( data.x.min(),data.x.max() ,t_max ).reshape(t_max,1)
        mu = func.netto(x.to(device)).flatten().cpu()
        ax.plot(x,mu.flatten(),label = "Network",linestyle="dashed")
        ax.legend()
        ax.set_xlabel(r"$m(t)$")
        ax.set_ylabel(r"$m'(t)$")

        plt.savefig("./images/x_dot.pdf")

if __name__ == "__main__":
    with torch.no_grad():
        device = "cpu"
        N = 128
        T = 2.0566 
        train1 = 30
        train2 = 31
        file_name =  "./dataset/N_128_T_%.4f_num_traj_1000.h5"%T
        t_max = 50000
        every = 1
        dataset = Data(file_name,device = device,t_max = t_max,every=every)
        plot_dyn(T,train1,train2,device,dataset,file_name)











