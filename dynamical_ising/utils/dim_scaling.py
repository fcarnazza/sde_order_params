import matplotlib.pyplot as plt
import h5py
from matplotlib.ticker import FormatStrFormatter
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
from mu_diff import Mu 


def scatter_beta(dim_list,
                Ts,
                n_dx = 200,
                ):

        funcs = [nn.ModuleList([Mu() for _ in range(len(Ts))]) for _ in range(len(dim_list))]

        x_0=torch.zeros(len(dim_list),len(Ts))
        x_mins=torch.zeros(len(dim_list),len(Ts))
        y_mins=torch.zeros(len(dim_list),len(Ts))

        for idx_d in range(len(dim_list)):
                for idx_T in range(len(Ts)):
                        model_name = "mu_net.ckpt"
                        model_dir = "./dump/t_100000_ising_mu_%.3f_N_%d"%(Ts[idx_T], dim_list[idx_d] )
                        model_path = os.path.join(model_dir,model_name)
                        device = "cpu"
                        ckpt = torch.load(model_path,map_location=torch.device('cpu'))
                        funcs[idx_d][idx_T].load_state_dict(ckpt['model'])
                        x = torch.linspace(0,1.0,n_dx).reshape(n_dx,1)
                        mu = funcs[idx_d][idx_T](x.to(device)).flatten().cpu()
                        y = torch.cumsum(-1/n_dx* mu ,dim=0)
                        x_0[idx_d][idx_T] = x[torch.argmin(torch.abs(mu))]
                        x_mins[idx_d][idx_T] = x[torch.argmin(y   )]
                        y_mins[idx_d][idx_T] = y[torch.argmin(y)]
        
        fig, (ax) = plt.subplots(1, 1)
        fig.set_size_inches(w=4.7747/1.35, h=2.9841875*0.65)
        fig.tight_layout(pad=2.5)
        for idx_d in range(len(dim_list)):
                ax.scatter(Ts,x_mins[idx_d])
        ax.set_xlabel(r"$T$")
        ax.set_ylabel(r"$\bar m _{\rm stat}$")
        plt.savefig("./images/scatter_dim_scalinf.pdf")

        plt.close()



if __name__ == "__main__":
        plt.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
            "font.size":11
         })
        name_file = "./dataset/temperaures_from_2.2214_to_2.2966_N_28.h5"
        Ts = np.array(h5py.File(name_file,'r' )["temps"])
        dim_list = [128]
        scatter_beta(dim_list,Ts)



