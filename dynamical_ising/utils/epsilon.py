from torchmin import minimize
import h5py
import matplotlib.pyplot as plt
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
from mu_diff import AverageMu as Mu# this is to use the averaged model, otherwise just import Mu 

class Epsilon():
        """
        class to find the zeros of the drift mu, 
        when given as an argumnt mu( const * |T -Tc|**beta  )
        """
        def __init__(self,
                        funcs:nn.ModuleList,
                        Ts:torch.Tensor,
                        ):
                self.funcs = funcs
                self.Ts = Ts

        def eps(self,w_kc: torch.Tensor) -> torch.Tensor:
                """
                                {Ti}                   beta
                function sum |mu     ( const * |Ti -Tc|     )|
                          i
                for the Ti in self.Ts
                """
                beta = w_kc[0] 
                kc = w_kc[1]
                const = w_kc[2]
                x = torch.tensor([0.])
                for i, f in enumerate(self.funcs):
                        x = x + torch.abs(f((w_kc[2]*torch.abs( self.Ts[i]-w_kc[1])**w_kc[0]).reshape(1,1)).flatten())
                return x

def plot_beta(w_kc,Ts,x_mins,funcs,n_dx=200):
        """
        given a triple w_kc = (beta, Tc, const) and x_mins for the minima of the
        effective potential Mu (i.e. zeros of mu), this function makes a scatterpot of the minima
        agains the temperature T and the plot of x_min = const * |T-Tc|**beta
        """
        cmap = mpl.cm.get_cmap('rainbow')
        rgba = cmap(np.linspace(0, 1, len(Ts)))

        beta = w_kc[0] 
        Tc = w_kc[1]
        const = w_kc[2]
        x = torch.linspace(0,1.0,n_dx).reshape(n_dx,1)
        with torch.no_grad():
                fig, (ax1,ax2) = plt.subplots(1, 2)
                fig.set_size_inches(w=4.7747/1.35, h=2.9841875*0.65)
                fig.tight_layout(pad=2.5)
                ax2.scatter( Ts-Tc,x_mins ,s=5)
                ax2.plot(Ts[Ts<Tc]-Tc, const*torch.abs(Ts[Ts<Tc]-Tc)**beta,c="orange")
                #ax2.ticklabel_format(style="sci",scilimits=(0,0))
                ax2.xaxis.set_major_locator(plt.MaxNLocator(2))
                ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax2.get_xaxis().get_offset_text().set_position((0.,0))
                ax2.set_xlabel(r"$T-T_{\rm c}$")
                ax2.set_ylabel(r"$\bar m_{\rm stat}$")
                for idx in range(len(Ts))[0::3]:
                        mu = funcs[idx](x.to(device)).flatten().cpu()
                        ax1.plot(x,mu,c=rgba[idx])
                ax1.set_xlabel(r"$m$")
                ax1.set_ylabel(r"$\mu_\theta(m)$")
                norm = plt.Normalize(min(Ts),max(Ts))
                cbar1 = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
                                ax=ax1,
                                )
                cbar1.ax.set_title(r'$T$')
                ax1.text(-0.4, 1.1, "(a)", transform=ax1.transAxes)
                ax2.text(-0.4, 1.1, "(b)", transform=ax2.transAxes)
                plt.savefig("./images/beta_from_newton.pdf")
                plt.close()


def load_funcs(Ts,n_dx= 200):
        """
        function to load in a nn.ModuleList the trained drits mu.
        returns the ModuleList and its zeros along the postitive semi axis.
        """
        funcs = nn.ModuleList([Mu() for _ in range(len(Ts))])
        x_mins = torch.zeros(len(Ts))
        x = torch.linspace(0,1.0,n_dx).reshape(n_dx,1)
        for idx in range(len(Ts)):
                #for the single decommnet this and comment the following
                #model_name = "mu_net.ckpt"
                model_name = "mu_avg_net.ckpt"
                model_dir = "./dump/t_100000_ising_mu_%.4f_N_%d"%(Ts[idx], N )
                model_path = os.path.join(model_dir,model_name)
                device = "cpu"
                ckpt = torch.load(model_path,map_location=torch.device('cpu'))
                funcs[idx].load_state_dict(ckpt['model'])
                mu = funcs[idx](x.to(device)).flatten().cpu()
                y = torch.cumsum(-1/n_dx* mu ,dim=0)
                x_mins[idx] = x[torch.argmin(y   )]
        return funcs, x_mins

if __name__ == "__main__":
        plt.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
            "font.size":11
         })
        #with torch.no_grad():
        device = "cpu"
        n_dx = 200
        name_file = "./dataset/temperaures_from_2.2214_to_2.2966_N_28.h5"
        N = 128
        w_kc0 = torch.Tensor([0.125,2.269,1])
        w_kcs = torch.zeros(3)  
        Ts = torch.from_numpy(np.array(h5py.File(name_file,'r' )["temps"])).float()
        funcs, x_mins = load_funcs(Ts)
        for max_t in range(11,25):
                epsilon = Epsilon(  funcs[:max_t], Ts[:max_t])
                minim = minimize(epsilon.eps, w_kc0, method='bfgs')
                print(minim)
                print("T max = %.4f"%Ts[max_t].item())
                if minim.success == True:
                        w_kcs=minim.x 
                else:
                        break
        plot_beta(w_kcs,Ts,x_mins,funcs)
        [print(t,m) for t,m  in zip(Ts,x_mins)]

               
