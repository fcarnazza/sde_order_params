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
parser = argparse.ArgumentParser('inf generator for mu')
parser.add_argument('--T', type=float, default=2.2214)
parser.add_argument('--print_example', type=bool, default=False)
parser.add_argument('--pre_kappa', type=float, default=0.255)
parser.add_argument('--dt', type=float, default=0.001)
parser.add_argument('--N', type=int, default=128)

args = parser.parse_args()
class Mu(nn.Module):
    """
    This rather simple architecture is used to train the 
    drift mu.
    """
    def __init__(self):
        super(Mu, self).__init__()
        self.mu_net =  nn.Sequential(
            nn.Linear(1, 50),
            nn.ReLU(),
            nn.Linear(50,50),
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
        in the hope of better results
        """
        def __init__(self,n_mu=10,weights=torch.ones(10),dt=0.001):
                super(AverageMu, self).__init__()
                if len(weights) != n_mu:
                        raise ValueError("weights must have length n_mu")
                self.funcs = nn.ModuleList([Mu() for _ in range(n_mu)])
                self.weights = weights
                self.dt = dt
        def forward(self,x):
                y = torch.Tensor([0.])
                for w,f in zip(self.weights,self.funcs):
                        y = y + w*f(x) 
                return y/self.weights.sum() 


class dataset(Dataset):
    def __init__(self, file_name,thres=0.01,num_trajs=1000,t_max = 10000):
        self.ds = h5py.File(file_name,'r')
        # Z2 symm
        self.trajs = np.array([self.ds[key][:t_max] for key in self.ds.keys()])
        self.max = np.max(self.trajs)
        self.min = np.min(self.trajs)
        self.thres = thres
        self.num_trajs = num_trajs
        self.t_max = t_max
    def __getitem__(self,idx):
        x=self.trajs[:self.num_trajs,:self.t_max]

        x0=np.random.rand()*(self.max-self.min) + self.min
        idx = np.argwhere( np.abs(x[:,:-1]-x0)<self.thres  )
        x0s = x[idx[:,0],idx[:,1]]
        x_dt = x[:,1:][idx[:,0],idx[:,1]]
        #                   x                                
        #                 E   (Zt+dt)  - x                   
        #  mu(x) = lim   ------------------- (1)  ------->   E( Zt+1 - Zt) (2) 
        #          dt->0        dt                           
        #
        #mu_x = torch.mean( torch.from_numpy(x_dt))-x0     # (1)
        mu_x = torch.mean( torch.from_numpy(x_dt-x0s))   # (2)
        _x0s_ = torch.from_numpy(x0.reshape(1,))
        return _x0s_.reshape(1,).float(), mu_x.reshape(1,).float()
    def __len__(self):
        return self.trajs.shape[0]

def train_mu(model,
             dataset,
             train_dir,
             batch_size=100,
             input_size=1,
             hidden_size=64,
             output_size=1,
             num_iters=500,
             stop_every= 100,
             training_diff=False,
             dt=0.001,
             ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = optim.RMSprop(model.parameters(), lr=0.3e-3)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    infinite_train_dataloader = (elem for it in iter(lambda: train_dataloader, None) for elem in it)
    os.makedirs(os.path.dirname(train_dir),exist_ok=True)
    for global_step in tqdm.tqdm(range(1, num_iters+1)):
        x0, dx = next(iter(infinite_train_dataloader))
        model.zero_grad()
        loss = ((model(x0)-dx/dt)**2).mean()
        loss.backward()
        optimizer.step()
        if global_step%stop_every==0:
            #print(loss.item())
            name = train_dir+"/mu_diff.pdf"
            plot_results(model,x0,dx,name,dt)

    return model


def train_avg_mu(avg_model,
             dataset,
             train_dir,
             batch_size=100,
             input_size=1,
             hidden_size=64,
             output_size=1,
             num_iters=7000,
             stop_every= 100,
             training_diff=False,
             dt=0.001,#0.001,
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
                #    print(loss.item())
                    name = train_dir+"/mu_diff_trai_"+str(idx)+".pdf"
                    plot_results(model_idx,x0,dx,name,dt)
            avg_model.weights[idx] = (((model_idx(x0)*dt-dx)**2).mean()**(-1)).detach()
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
    fig.set_size_inches(w=4.7747/1.35, h=2.9841875*0.65)
    fig.tight_layout(pad=1.5)
    with torch.no_grad():

        x, indices = torch.sort(x.flatten())
        y = y.flatten()[indices]
        mu_x_model = model(x.reshape(len(x),1)).detach().numpy().flatten()
        ax.plot(x,y,label = "Data")
        ax.plot(x,mu_x_model*dt,label = "Network")
        ax.set_xlabel(r"$m$")
        ax.set_ylabel(r"$\mu(m)$")
        ax.ticklabel_format(style="sci",scilimits=(0,0))
        ax.legend()
        plt.savefig(name)
        plt.close()

if __name__ == "__main__":
    file_name =  "./dataset/N_%d_T_%.4f_num_traj_1000t_max_100000.h5"%(args.N,args.T)
    model_dir = "./dump/t_100000_ising_mu_%.4f_N_%d/"%(args.T,args.N)  
    data = dataset(file_name)
    
    # To train just a single instace of Mu de-comment this 3 lines:
    #model = Mu()
    #model = train_mu_avg(model,data,model_dir)
    #model_name = 'mu_net.ckpt'

    # and comment the following 3:
    model = AverageMu()
    model = train_avg_mu(model,data,model_dir)
    model_name = 'mu_avg_net_4.ckpt'

    os.makedirs(os.path.dirname(model_dir),exist_ok=True)
    torch.save(
            {'model': model.state_dict(),
             'weights':model.weights,
             },
            os.path.join(model_dir, model_name)
            )
