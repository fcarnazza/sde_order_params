import torch
import os
import sys
import argparse
import random
import matplotlib.pyplot as plt
from torch import optim
from torch import nn
import tqdm
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mu_dynkin import *

from utils.cp_loader import *
parser = argparse.ArgumentParser('ODE sigma cp')
parser.add_argument('--kappa', type=float, default=3.29)
parser.add_argument('--pre_trained', type=bool, default=False)
parser.add_argument('--pre_kappa', type=float, default=0.255)
parser.add_argument('--N', type=int, default=100)
def plot(model,dataset):
    with torch.no_grad():
            train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)
            infinite_train_dataloader = (elem for it in iter(lambda: train_dataloader, None) for elem in it)
            x_ = dataset.x
            xs = next(iter(infinite_train_dataloader))
            print(xs.shape,model(xs).shape)
            x_t = torch.mean(model(xs),0)
            plt.rcParams.update({
                "pgf.texsystem": "pdflatex",
                'font.family': 'serif',
                'text.usetex': True,
                'pgf.rcfonts': False,
                "font.size":11
             })
            fig, (ax1,ax2) = plt.subplots(1,2 )
            fig.tight_layout(pad=5)
            fig.set_size_inches(w=4.7747, h=2.9841875/2*1.5)
            ax1.plot(x_t,label = r"$\langle\sum_t \mu_\theta(\rho(t))\rangle$")
            ax1.plot(x_,label = r"$\langle\rho(t)\rangle$")
            ax1.set_xlabel(r"$t$")
            ax1.set_ylabel(r"$\bar{\rho}(t) $")
            ax1.legend()
            rho = torch.linspace(0,1,100).reshape(100,1)
            ax2.plot( rho.flatten(),model.mu_net(rho).flatten()  )
            ax2.set_ylabel(r"$\mu_\theta$")
            ax2.set_xlabel(r"$\rho $")
            plt.savefig("./images/plot_mu_dynkin.pdf")


if __name__ == "__main__":
    device="cpu"
    kappa = args.kappa 
    t_final = 1000
    input_size = 1
    hidden_size = 64
    output_size = 1
    n_train=10
    num_iters=500
    data_dir = "../../Autoencoder/HGN/nn_sde/c_time_con_proc/" 
    file_name  = data_dir + "N_"+str(args.N)+"_perc_1.0_nsteps_10000_inf_rate_"+str(args.kappa)+"_rec_rate_1_surv_prob_False_.h5" 
    train_dir = "dump/N_"+str(args.N)+"_cp_mu_dynkin_"+str(args.kappa )+"_num_iter_"+str(num_iters)+"/" 
    model_path=train_dir+"mu_net.ckpt"
    model = Mu(
            input_size, hidden_size, output_size
            )
    ckpt = torch.load(model_path,map_location=torch.device('cpu'))
    model.load_state_dict(ckpt['model'])
    dataset = Data_mu_p_train(
            file_name = file_name, 
            t_final = 1000,
            device = device,
            every=50,
            t_step = 0.01)
    plot(model,dataset)
