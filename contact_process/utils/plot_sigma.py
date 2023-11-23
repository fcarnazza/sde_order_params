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
from sigma2 import *
from utils.weighted_model import AvgsODEFunc, ODEFunc 
from utils.cp_loader import *
parser = argparse.ArgumentParser('plot sigma cp')
parser.add_argument('--input_dim', type=int, default=1)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--output_dim', type=int, default=1)
parser.add_argument('--n_train', type=int, default=2)
parser.add_argument('--kappa', type=float, default=3.29)
parser.add_argument('--pre_trained', type=bool, default=False)
parser.add_argument('--pre_kappa', type=float, default=0.255)
parser.add_argument('--N', type=int, default=100)

args = parser.parse_args()

if __name__ == "__main__":
        data_dir = "../../Autoencoder/HGN/nn_sde/c_time_con_proc/" 
        device = "cpu"
        file_name  = data_dir + "N_"+str(args.N)+"_perc_1.0_nsteps_10000_inf_rate_"+str(args.kappa)+"_rec_rate_1_surv_prob_False_.h5" 
        mult=1.
        train_dir = "dump/N_"+str(args.N)+"_cp_mu_"+str(args.kappa )+"/" 
        model_path=train_dir+"weighted_sigma_net.ckpt"
        model = WeightedSigma(n_train=args.n_train)
        ckpt = torch.load(model_path,map_location=torch.device('cpu'))
        model.load_state_dict(ckpt['model'])
        model.weight=ckpt['weights']
        x = torch.linspace(0,1,100).reshape(100,1)
        dataset = Data(
            file_name = file_name, 
            t_final = 1000,
            device = device,
            #every=100,
            every=50,
            t_step = 0.01)
        with torch.no_grad():
            plt.rcParams.update({
                "pgf.texsystem": "pdflatex",
                'font.family': 'serif',
                'text.usetex': True,
                'pgf.rcfonts': False,
                "font.size":11
             })
            fig, (ax1,ax2) = plt.subplots(1, 2)
            fig.tight_layout(pad=5)
            fig.set_size_inches(w=4.7747/1.4, h=2.9841875/2*1.5)
            ax1.plot(x.flatten(),model(mult*x).flatten()/mult)
            ax1.set_xlabel(r"$\rho$")
            ax1.set_ylabel(r"$\sigma_\theta$")
 
 
            qv = []
            dt_qv = []
            t = []
            pred = []
            x = []
            for i in range(0, 100):
                v = dataset[i]
                qv.append(v[0])
                t.append(v[3])
                dt_qv.append((v[1]-v[0])/v[3]/mult**2)
                pred.append((model(v[2])**2/mult**2).flatten())
                x.append(v[2])
            t = torch.cumsum(torch.tensor(t),0)
            ax2.plot(t,torch.tensor(pred).flatten(),color = "green",label= r"$\sigma_\theta^2(\rho^\textrm{true})$"  )
            ax2.plot(t,dt_qv,color = "red",label =r"$\partial_t [\rho^\textrm{true}]_t$")
            ax1.text(-0.1, 1.1, "(b)", transform=ax1.transAxes)
            ax2.text(-0.1, 1.1, "(a)", transform=ax2.transAxes)

            box = ax2.get_position()
            ax2.set_position([box.x0, box.y0, box.width, box.height*0.8])

            ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5))

            ax2.set_xlabel(r"$t$")
            ax2.set_ylabel(r"$\sigma^2_t$")
            name_fig = "./images/w_sigma2_kappa_%.3f.pdf"%args.kappa
            plt.savefig(os.path.join(train_dir,"w_scatter.pdf")  )
            plt.savefig( name_fig)
            plt.close()
