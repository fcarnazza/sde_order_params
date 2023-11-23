import matplotlib.pyplot as plt
import h5py
import argparse
from torch import optim
import torch
import numpy as np
from torch.utils.data import Dataset
import torch
from torch import nn
import tqdm
import os
from utils.datasets_traj import dataset_ctime as dataset
parser = argparse.ArgumentParser('mu cp')
parser.add_argument('--kappa', type=str, default="3.3559322033898304")
parser.add_argument('--N', type=int, default=100)
parser.add_argument('--dt', type=int, default=0.01)

args = parser.parse_args()
class Mu(nn.Module):
    """
    A fully connceted feed forward net which represents the 
    drift in the stochastic porcess for the contact process
    d rho = mu     (rho ) dt + sigma(rho  ) dW
         t    theta    t                t     t
    """
    def __init__(self):
        super(Mu, self).__init__()
        self.mu_net =  nn.Sequential(
            nn.Linear(1, 50),
            nn.ReLU(),
            nn.Linear(50,50),
            nn.ReLU(),
            nn.Linear(50, 1), 
        )   

    def forward(self,v):
           return self.mu_net(v) 

class AverageMu(nn.Module):
        """
        Module to compute a weighted average model for the Mu 
        the weighs are chosen in the training procedure, here we use the
        MSE loss between the prediction and the data.
        """
        def __init__(self,n_mu=10,weights=torch.ones(10),dt=args.dt):
                super(AverageMu, self).__init__()
                if len(weights) != n_mu:
                        raise ValueError("weights must have length n_mu")
                self.funcs = nn.ModuleList([Mu() for _ in range(n_mu)]) # the drifts
                self.weights = weights #the weights
                self.dt = dt #during training, we make use of a dt, the network need to be susequently rescaled accordingly
        def forward(self,x):
                y = torch.Tensor([0.])
                for w,f in zip(self.weights,self.funcs):
                        y = y + w*f(x) 
                return self.dt*y/self.weights.sum() 

def train_mu(model,
             dataset,
             train_dir,
             batch_size=200,
             input_size=1,
             hidden_size=64,
             output_size=1,
             num_iters=2000,
             stop_every= 100,
             training_diff=False,
             dt=args.dt,
             ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = optim.RMSprop(model.parameters(), lr=0.5e-3)


    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    infinite_train_dataloader = (elem for it in iter(lambda: train_dataloader, None) for elem in it)
    os.makedirs(os.path.dirname(train_dir),exist_ok=True)
    for global_step in tqdm.tqdm(range(1, num_iters+1)):
        x0, dx = next(iter(infinite_train_dataloader))
        model.zero_grad()
        loss = torch.abs(model(x0)*dt-dx).mean()
        loss.backward()
        optimizer.step()
        if global_step%stop_every==0:
            print(loss.item())
            name = train_dir+"/mu_diff.pdf"
            plot_results(model,x0,dx,name)

    return model

def train_avg_mu(avg_model,
             dataset,
             train_dir,
             batch_size=200,
             input_size=1,
             hidden_size=64,
             output_size=1,
             num_iters=2000,
             stop_every= 100,
             training_diff=False,
             dt=args.dt,
             ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    infinite_train_dataloader = (elem for it in iter(lambda: train_dataloader, None) for elem in it)
    os.makedirs(os.path.dirname(train_dir),exist_ok=True)
    for idx in range(len(avg_model.funcs)):
            model_idx = avg_model.funcs[idx]
            optimizer = optim.RMSprop( model_idx.parameters()  , lr=0.3e-3)
            print("Training for model %d"%idx)
            x0, dx = next(iter(infinite_train_dataloader))
            for global_step in tqdm.tqdm(range(1, num_iters+1)):
                model_idx.zero_grad()
                loss = torch.abs(model_idx(x0)-dx/dt).mean()
                loss.backward()
                optimizer.step()
                if global_step%stop_every==0:
                    print(loss.item())
                    name = train_dir+"/mu_diff_trai_"+str(idx)+".pdf"
                    plot_results(model_idx,x0,dx,name,dt)
            avg_model.weights[idx] = (  (((model_idx(x0)*dt-dx)**2).mean()) **(-1)).detach()
            avg_model.funcs[idx] = model_idx
    return avg_model
def plot_results(model,x,y,name = "./images/mu_diff.pdf",dt = args.dt):
    plt.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
            "font.size":11
         })
    fig, ax = plt.subplots(1, 1)
    with torch.no_grad():

        x, indices = torch.sort(x.flatten())
        y = y.flatten()[indices]
        mu_x_model = model(x.reshape(len(x),1)).detach().numpy().flatten()
        ax.plot(x,y,label = "Data")
        ax.plot(x,mu_x_model*dt,label = "Network")
        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel(r"$\mu(\rho)$")
        ax.legend()
        plt.savefig(name)
        plt.close()

if __name__ == "__main__":
    ##data_dir = "../../Autoencoder/HGN/nn_sde/c_time_con_proc/" 
    #data_dir = "./dataset/"
    #file_name  = data_dir + "N_"+str(args.N)+"_perc_1.0_nsteps_10000_inf_rate_"+str(args.kappa)+"_rec_rate_1_surv_prob_False_.h5" 
    #file_name = "./dataset/N_"+str(args.N)+"_t_final100_up0_ctime_cp_kappa_"+str(args.kappa)+"_gamma_1.0.h5"
    file_name = "./dataset/N_"+str(args.N)+"_t_final100000_up0_ctime_cp_kappa_"+str(args.kappa)+"_gamma_1.0.h5"
    model_dir = "./dump/N_"+str(args.N)+"_cp_mu_ctime_"+str(args.kappa)+"/" 
    data = dataset(file_name)
    # to train for just one model, w/o averaging over many (quite bad res.) decomment: 
    #model = Mu()
    #model = train_mu(model,data,model_dir)
    #model_name = 'mu_net.ckpt'
    #and comment following 3 lines:
    avg_model = AverageMu()
    avg_model = train_avg_mu( avg_model,data,model_dir  ) 
    model_name = 'mu_avg_net.ckpt'

    os.makedirs(os.path.dirname(model_dir),exist_ok=True)
    torch.save(
            {'model': avg_model.state_dict(),
             'weights': avg_model.weights,
             },
            os.path.join(model_dir, model_name)
            )
