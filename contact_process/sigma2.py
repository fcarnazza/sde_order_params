import torch
import argparse
import random
import matplotlib.pyplot as plt
from torch import optim
from torch import nn
import tqdm
from utils.cp_loader import *

parser = argparse.ArgumentParser('ODE sigma cp')
parser.add_argument('--kappa', type=float, default=3.29)
parser.add_argument('--pre_trained', type=bool, default=False)
parser.add_argument('--pre_kappa', type=float, default=0.255)
parser.add_argument('--N', type=int, default=100)

args = parser.parse_args()

class Sigma2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Sigma2, self).__init__()
        self.sigma_net = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, output_size),
                    nn.Sigmoid()
                    )

    def forward(self,v):
            sigma2 = self.sigma_net( v[2])**2
            return sigma2 
class DummyNet(nn.Module):
    def __init__(self):
        super(DummyNet, self).__init__()
        self.sigma_net = None
def train_sigma(      model,
                file_name,
                train_dir,
                batch_size=1000,
                input_size=1,
                hidden_size=64,
                output_size=1,
                num_iters=5000,
                mult = 9.,
                stop_every= 400
                ): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Data(
            file_name = file_name, 
            t_final = 1000,
            device = device,
            #every=100,
            every=50,
            t_step = 0.01)
    optimizer = optim.RMSprop(model.parameters(), lr=1e-3)
    if args.pre_trained == True:
        pre_model_path  = "./dump/cp_mu_%.3f" %args.pre_kappa
        model_path = os.path.join(pre_model_path ,"sigma_net.ckpt")
        ckpt = torch.load(model_path,map_location=torch.device(device))
        func.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    infinite_train_dataloader = (elem for it in iter(lambda: train_dataloader, None) for elem in it)
    for global_step in tqdm.tqdm(range(1, num_iters+1)):
            xs = torch.transpose(next(iter(infinite_train_dataloader)),0,1)
            model.zero_grad()
            #loss = model(xs)
            loss = torch.abs( (xs[1] -xs[0])/xs[3] -  model(xs)).mean()
            if global_step%stop_every==0:
                    print(loss)
                    #torch.save(
                    #        {'model': model.state_dict(),
                    #         'optimizer': optimizer.state_dict()},
                    #        os.path.join(train_dir, f'sigma_net_step_{global_step}.ckpt')
                    #        )
            loss.backward()
            optimizer.step()
    torch.save(
            {'model': model.state_dict(),
             'optimizer': optimizer.state_dict()},
            os.path.join(train_dir, f'sigma_net.ckpt')
            )
    return model, dataset,loss.detach().item()

def plots_sigma2(sigma,dataset,kappa = args.kappa):
    with torch.no_grad():
        x = torch.linspace(0,1,100).reshape(100,1).cpu()
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(w=4.7747/1.5, h=2.9841875/2*1.7)
        plt.rcParams.update({
                "text.usetex": True,
                    "font.family": "Helvetica"
                    })
        ax.plot(x.flatten(),torch.abs(sigma(x)).flatten())
        ax.set_xlabel(r"\rho")
        ax.set_ylabel(r"$\sigma$")
        name_fig = "./images/sigma2_kappa_%.3f.pdf"%kappa
        plt.savefig(name_fig)
        plt.close()

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
                        y = y+ (1/self.weights)[idx].flatten()*self.funcs[idx].sigma_net.forward(x)
                return y/(1/self.weights).sum()

if __name__ == "__main__":
    device="cpu"
    kappa = args.kappa 
    t_final = 1000
    #mult =9.
    mult =1.
    every = 200
    t_step = 0.01
    input_size = 1
    hidden_size = 64
    output_size = 1
    n_train=10
    data_dir = "../../Autoencoder/HGN/nn_sde/c_time_con_proc/" 
    file_name  = data_dir + "N_"+str(args.N)+"_perc_1.0_nsteps_10000_inf_rate_"+str(args.kappa)+"_rec_rate_1_surv_prob_False_.h5" 
    train_dir = "dump/cp_mu_%.3f"%kappa
    train_dir = "dump/N_"+str(args.N)+"_cp_mu_"+str(args.kappa )+"/" 
    os.makedirs(train_dir, exist_ok=True)
    model = Sigma2(
            input_size, hidden_size, output_size
            )
    model,dataset ,_= train_sigma(model,file_name,train_dir)

    weight_model = WeightedSigma(n_train=n_train)
    for idx in range(n_train):
            weight_model.funcs[idx],_,weight_model.weights[idx] = train_sigma(model,file_name,train_dir)

    torch.save(
            {'model': weight_model.state_dict(),
             'weights':weight_model.weights,
             },
            os.path.join(train_dir, f'weighted_sigma_net.ckpt')
            )
    dummy = DummyNet()
    dummy.sigma_net = model.sigma_net
    x = torch.linspace(0,1,100).reshape(100,1)
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
        ax1.plot(x.flatten(),model.sigma_net(mult*x).flatten()/mult)
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
            pred.append((model.sigma_net(v[2])**2/mult**2).flatten())
            x.append(v[2])
        t = torch.cumsum(torch.tensor(t),0)
        ax2.plot(t,torch.tensor(pred).flatten(),color = "green",label= r"$\sigma_\theta^2(\rho_\textrm{True})$"  )
        ax2.plot(t,dt_qv,color = "red",label =r"$\partial_t [\rho_\textrm{True}]_t$")
        ax1.text(-0.1, 1.1, "(b)", transform=ax1.transAxes)
        ax2.text(-0.1, 1.1, "(a)", transform=ax2.transAxes)
        ax2.legend()
        ax2.set_xlabel(r"$t$")
        ax2.set_ylabel(r"$\sigma^2_t$")
        name_fig = "./images/sigma2_kappa_%.3f.pdf"%kappa
        plt.savefig(os.path.join(train_dir,"scatter.pdf")  )
        plt.savefig( name_fig)
        plt.close()
