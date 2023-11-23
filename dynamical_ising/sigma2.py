import torch
import numpy as np
import h5py
from torch.utils.data import Dataset
import os
import argparse
import random
import matplotlib.pyplot as plt
from torch import optim
from torch import nn
import tqdm
from mu_diff import AverageMu
parser = argparse.ArgumentParser('sigma ising')
parser.add_argument('--T', type=float, default=2.269)
parser.add_argument('--pre_trained', type=bool, default=False)
parser.add_argument('--pre_T', type=float, default=0.98)
parser.add_argument('--dt', type=float, default=0.001)
parser.add_argument('--delta_t', type=int, default=500)
parser.add_argument('--N', type=int, default=128)
parser.add_argument('--n_steps', type=int, default=100000)
args = parser.parse_args()

def load_ising_dataset(t0,t1,file_name,device,steps):
        data_h5 = h5py.File(file_name,'r')
        keys = [key for key in data_h5.keys()]
        data = np.array(data_h5[keys[0]])
        for key in keys[1:]:
                data = np.concatenate((data,np.array(data_h5[key])))
        dta_len = int(len(data)/steps)
        ts = torch.linspace(0., float(steps), steps=steps, device=device)
        xs = torch.from_numpy(data.reshape(int(len(data)/steps),steps,1)).float()
        #xs = torch.abs(xs)
        return xs,ts

class Data_sigma_p_train(Dataset):
    def __init__(
                    self, 
                    file_name,device= "cpu",
                    delta_t=args.delta_t,
                    n_steps=args.n_steps,
    ):
        self.delta_t = delta_t
        self.xs, self.time = load_ising_dataset(0.,1.,file_name,device,steps = n_steps) 
        self.xs = self.xs
        dx2 = (self.xs.T[:,1:,:]-self.xs.T[:,:-1,:])**2
        qv_dx2 = torch.cumsum(dx2.T,1)
        coarse_qv_dx2 = qv_dx2.T[:,0::self.delta_t,:]
        self.qv = coarse_qv_dx2.T
        self.num = len(self.qv)
        self.coarse_xs = self.xs.T[:,0::self.delta_t,:].T
        # data is organized in tuples ( [X]_t, [X]_t+dt,X_t   )
        data = torch.cat(
                            (
                                    self.qv.T[:,:-1,:].T, # [X]_t
                                    self.qv.T[:,1:,:].T, # [X]_t+dt 
                                    self.coarse_xs.T[:,:-1,:].T, #X_t 
                            ),
                                dim = 2,
                                )
        self.dataset = data.reshape( (data.shape[0]*data.shape[1],3,1) )
        self.num = int(len(self.dataset))
    def __getitem__(self, idx):
        return self.dataset[(idx+1)% self.num].float()
    def __len__(self):
        return self.num

class Sigma2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Sigma2, self).__init__()
        self.sigma_net = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, output_size),
                    nn.Sigmoid(),
                    )

    def forward(self,v):
            sigma2 = self.sigma_net(v[2])**2
            return sigma2
class WeightedSigma(nn.Module):
        """
        A model for the diffusion which is a weighted average of other models
        """
        def __init__(   self,
                n_train = 10,
                input_size=1, 
                hidden_size=64,
                output_size=1,
                ):
                super(WeightedSigma, self).__init__()
                self.funcs = nn.ModuleList([
                Sigma2(
                        input_size, hidden_size, output_size
                        ) for _ in range(n_train)])
                self.weights = torch.ones(n_train,requires_grad=False)
                self.n_train = n_train

        def forward(self,x):
                y = torch.zeros_like(x)
                for idx in range(self.n_train):
                        y = y+ self.weights[idx].flatten()*self.funcs[idx].sigma_net.forward(x)
                return y/(self.weights.sum())

def train_sigma(      model,
                train_dir,
                dataset,
                batch_size=1000,
                input_size=1,
                hidden_size=64,
                output_size=1,
                num_iters=5000,
                stop_every= 400
                ): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = optim.RMSprop(model.parameters(), lr=1e-3)
    if args.pre_trained == True:
        pre_model_path  = "./dump/ising_sigma_%.4f_N_%d" %(args.pre_T,args.N)
        model_path = os.path.join(pre_model_path ,"sigma_net.ckpt")
        ckpt = torch.load(model_path,map_location=torch.device(device))
        func.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    infinite_train_dataloader = (elem for it in iter(lambda: train_dataloader, None) for elem in it)
    for global_step in tqdm.tqdm(range(1, num_iters+1)):
            xs = torch.transpose(next(iter(infinite_train_dataloader)),0,1)
            model.zero_grad()
            loss = torch.abs( (xs[1] -xs[0])/(args.delta_t) -  model(xs)).mean()
            if global_step%stop_every==0:
                    print(loss)
            loss.backward()
            optimizer.step()
    w = ((( (xs[1] -xs[0])/args.delta_t -  model(xs))**2).mean())**(-1)
    return model, w

def plots_sigma2(sigma,dataset):
    with torch.no_grad():
        x = torch.linspace(-1,1,100).reshape(100,1).cpu()
        plt.rcParams.update({
                "text.usetex": True,
                    "font.family": "Helvetica"
                    })
        plt.plot(x.flatten(),torch.abs(sigma(x)).flatten())
        plt.xlabel(r"\rho")
        plt.ylabel(r"$\sigma$")
        plt.savefig("./images/sigma2.pdf")
        plt.close()

if __name__ == "__main__":
    device="cpu"
    T = args.T 
    n_train = 5 
    t_final = 1000
    input_size = 1
    hidden_size = 64
    output_size = 1
    file_name =  "./dataset/N_%d_T_%.4f_num_traj_1000t_max_100000.h5"%(args.N,args.T)
    train_dir = "dump/ising_sigma_%.4f_N_%d"%(T,args.N)
    os.makedirs(train_dir, exist_ok=True)
    model = Sigma2(
            input_size, hidden_size, output_size
            )
    dataset = Data_sigma_p_train(
            file_name = file_name, 
            device = device,
            delta_t=args.delta_t,
            )
    model,_ = train_sigma(model,train_dir,dataset)
    x_min = dataset.xs.min()
    x_max = dataset.xs.max()
    x = torch.linspace(x_min,x_max,100).reshape(100,1)
    w_model = WeightedSigma(n_train=n_train)
    print("start %d trainings"%n_train)
    for idx in range(n_train):
            model = Sigma2(
                    input_size, hidden_size, output_size
                    )
            w_model.funcs[idx],w_model.weights[idx] = train_sigma(model,train_dir,dataset) 
    torch.save(
            {'model': w_model.state_dict(),
             'weights': w_model.weights,},
            os.path.join(train_dir, f'sigma_net.ckpt')
            )
    
    with torch.no_grad():
        mu = AverageMu()
        model_name = "mu_avg_net_4.ckpt"
        model_dir = "./dump/t_100000_ising_mu_%.4f_N_%d"%(args.T, args.N )
        model_path = os.path.join(model_dir,model_name)
        device = "cpu"
        ckpt = torch.load(model_path,map_location=torch.device('cpu'))
        mu.load_state_dict(ckpt['model'])
        mu.weights = ckpt["weights"]
        plt.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
            "font.size":9
         })
        plt.subplots_adjust(bottom=0.19)
        fig, (ax1,ax2) = plt.subplots(1, 2)
        fig.set_size_inches(w=4.7747/1.35, h=2.9841875*0.65)
        fig.tight_layout(pad =1.8)

       # box = ax2.get_position()
       # box.x0 = box.x0 - 0.1
       # box.x1 = box.x1 - 0.1
       # box.y0 = box.y0 - 0.1
       # box.y1 = box.y1 - 0.1
       # ax2.set_position(box)

       # box = ax1.get_position()
       # box.y0 = box.y0 - 0.1
       # box.y1 = box.y1 - 0.1
       # ax1.set_position(box)

        ax1.text(-0.15, 1.1, "(a)", transform=ax1.transAxes)
        ax1.ticklabel_format(style="sci",scilimits=(-0.2,0))
        ax2.text(-0.15, 1.1, "(b)", transform=ax2.transAxes)
        sigma_x = w_model(x).flatten()
        mu_x = args.dt*mu(x)
        ax1.plot(x.flatten(),sigma_x,c="tab:red",label = r"$\sigma_\theta$")
        ax1.plot(x.flatten(),mu_x,label =r"$\mu_\theta$",alpha = .5,c="tab:blue")
        box = ax1.get_position()
        ax1.set_position([box.x0-0.08, box.y0, box.width*1.3, box.height])
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.),frameon=False)
        #ax1.set_ylim(-0.5e-3,0.5e-3)
        #ax1.set_yticks([-0.5e-1,0.5e-1])


        ax1.set_xlabel(r"$m$")
        ax1.get_xaxis().get_offset_text().set_position((0.2,0.))
        ax2.plot( (dataset.qv.T[:,1:,:].T[0] -dataset.qv.T[:,:-1,:].T[0] )/args.delta_t,color = "dimgrey", 
                        label =r"$\Delta_t [m]_t$" )
        ax2.plot( (w_model(dataset.coarse_xs[0])**2),color="black", 
                        label = r"$\sigma_\theta^2(m_t)$")
        ax2.set_ylim([3e-5,8e-5])
        #ax2.legend(loc='center left', bbox_to_anchor=(-0.13, 0.7))
        box = ax2.get_position()
        ax2.set_position([box.x0-0.05, box.y0, box.width*1.3, box.height])
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.),frameon=False)
        ax2.set_xlabel(r"$t$")
        plt.savefig( os.path.join(train_dir,"sigma2_T_%.3f.pdf"%args.T) )
        plt.savefig( "./images/sigma2_T_%.3f.pdf"%args.T) 
