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
                        kappas:torch.Tensor,
                        ):
                self.funcs = funcs
                self.kappas = kappas 

        def eps(self,w_kc: torch.Tensor) -> torch.Tensor:
                """
                                {Ti}                   beta
                function sum |mu     ( const * |Ti -Tc|     )|
                          i
                for the Ti in self.kappas
                """
                beta = w_kc[0] 
                kc = w_kc[1]
                const = w_kc[2]
                x = torch.tensor([0.])
                for i, f in enumerate(self.funcs):
                        x = x + (1/f.dt*f(((w_kc[2]**2)*torch.abs( self.kappas[i]-w_kc[1])**w_kc[0]).reshape(1,1)).flatten())**2
                return x
        def error_x(self,w_kc):
                s = torch.tensor([0.])
                for i, f in enumerate(self.funcs):
                        x = ((w_kc[2]**2)*torch.abs( self.kappas[i]-w_kc[1])**w_kc[0]).reshape(1,1)
                        si = f.weights.sum()**(-1)
                        s = s + si*4*(f(x).flatten())**2
                return s 

def error_beta(w_kc,kappas,x_mins):
        """
        Mean square error of the power law agains the zeros of the drift 
        """
        beta = w_kc[0] 
        Tc = w_kc[1]
        const = w_kc[2]** 2
        error = ((const*torch.abs(kappas[kappas>Tc]-Tc)**beta-x_mins[kappas>Tc])/x_mins[kappas>Tc]).sum()
        return error

def plot_beta(w_kc,kappas,x_mins,funcs,n_dx=200):
        """
        given a triple w_kc = (beta, Tc, const) and x_mins for the minima of the
        effective potential Mu (i.e. zeros of mu), this function makes a scatterpot of the minima
        agains the temperature T and the plot of x_min = const * |T-Tc|**beta
        """
        cmap = mpl.cm.get_cmap('binary')
        rgba = cmap(np.linspace(0, 1, len(kappas)))

        beta = w_kc[0] 
        Tc = w_kc[1]
        const = w_kc[2]**2
        x = torch.linspace(0,1.0,n_dx).reshape(n_dx,1)
        with torch.no_grad():
                fig, (ax1,ax2) = plt.subplots(1, 2)
                fig.set_size_inches(w=4.7747/1.35, h=2.9841875*0.65)
                fig.tight_layout(pad=1.5)
                ax1.ticklabel_format(style="sci",scilimits=(0,0))
                ax2.scatter( kappas-Tc,x_mins ,s=5, c="tab:blue")
                ax2.plot( kappas-Tc,x_mins ,c="tab:blue")
                Ts_to_Tc = torch.linspace(Tc,kappas.max(),100)
                ax2.plot(Ts_to_Tc-Tc, const*torch.abs(Ts_to_Tc-Tc)**beta,c="orange")

                #ax2.ticklabel_format(style="sci",scilimits=(0,0))
                ax2.xaxis.set_major_locator(plt.MaxNLocator(2))
                ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax2.get_xaxis().get_offset_text().set_position((0.,0))
                ax2.set_xlabel(r"$\kappa-\kappa_{\rm c}$")
                ax2.set_ylabel(r"$\bar \rho_{\rm stat}$")
                for idx in range(len(kappas))[0::5]:
                        mu = funcs[idx](x.to(device)).flatten().cpu()
                        y = torch.cumsum(-1/n_dx * mu ,dim=0)
                        if x[torch.argmin(y)] > 0.001:
                                ax1.plot(x,mu,c=rgba[idx])

                        else:
                                ax1.plot(x,mu,c=rgba[idx])
                        ax1.plot(x[torch.argmin(y):],mu[torch.argmin(y):],color=rgba[ idx  ])
                ax1.plot(x,torch.zeros_like(x),c="black", linestyle='dashed')
                ax1.set_xlabel(r"$\rho$")
                ax1.set_ylabel(r"$\mu^{\kappa_i}_\theta(\rho)$")
                ax1.set_ylim(-0.06)
                norm = plt.Normalize(min(kappas),max(kappas))
                cbar1 = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
                                ax=ax1,
                                )
                cbar1.ax.set_title(r'$\kappa$')
                ax1.text(-0.3, 0.9, "(a)", transform=ax1.transAxes)
                ax2.text(-0.3, 0.9, "(b)", transform=ax2.transAxes)
                plt.savefig("./images/beta_from_min_cp.pdf")
                plt.close()
                fig, ax1 = plt.subplots(1, 1)
                fig.set_size_inches(w=4.7747/1.35, h=2.9841875*0.65)
                #fig.tight_layout(pad=1.5)
                fig.tight_layout()
                for idx in range(len(kappas))[0::5]:
                        mu = funcs[idx](x.to(device)).flatten().cpu()
                        y = torch.cumsum(-1/n_dx * mu ,dim=0)
                        ax1.plot(x,y,c=rgba[idx])
                cbar1 = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
                                ax=ax1,
                                )
                cbar1.ax.set_title(r'$\kappa$')
                plt.savefig("./images/dynamical_function.pdf")


def load_funcs(kappas,N,n_dx= 200):
        """
        function to load in a nn.ModuleList the trained drits mu.
        returns the ModuleList and its zeros along the postitive semi axis.
        """
        funcs = nn.ModuleList([Mu() for _ in range(len(kappas))])
        x_mins = torch.zeros(len(kappas))
        x = torch.linspace(0,1.0,n_dx).reshape(n_dx,1)
        for idx in range(len(kappas)):
                print(kappas[idx])
                #for the single decommnet this and comment the following
                model_name = "mu_avg_net_1.ckpt"
                model_dir = "./dump/N_"+str(N)+"_cp_mu_"+kappas[idx].decode("utf-8")
                #model_dir = "./dump/N_"+str(N)+"_cp_mu_ctime_"+str(kappas[idx])
                #model_name = "mu_avg_net.ckpt"

                model_path = os.path.join(model_dir,model_name)
                device = "cpu"
                ckpt = torch.load(model_path,map_location=torch.device('cpu'))
                funcs[idx].load_state_dict(ckpt['model'])
                w = torch.ones(len(ckpt["weights"]))
                w = ckpt["weights"][ckpt["weights"]>0]
                funcs[idx].weights = w 
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
        #name_file1 = "./dataset/kappas_from_3.0000_to_3.3421_N_14.h5"
        #name_file = "./dataset/kappas_from_3.2900_to_4.5862_N_45.h5"
        #name_file = "./dataset/kappas_from_3.0000_to_4.5862_N_58.h5"
        #name_file = "./dataset/kappas_from_3.0000_to_4.5862_N_53.h5"
        name_file = "./dataset/kappas_from_2.0000_to_4.5862_N_63.h5"
        #name_file = "./dataset/kappas_from_2.0000_to_4.0000_N_45.h5"
        N = 100 
        w_kc0 = torch.Tensor([0.27,3.9,0.8])
        w_kcs = torch.zeros(3)  
        kappas = np.array(h5py.File(name_file,'r' )["kappas"])[:-10]
        funcs, x_mins = load_funcs(kappas,N)
        #kappas =torch.from_numpy(kappas)
        #kappas = torch.Tensor([float(k) for k in kappas])
        kappas = torch.Tensor([float(k.decode("utf-8")) for k in kappas])
        errors =[]
        values = []
        grads = []
        max_ts = []
        for max_t in range(20,len(kappas)):
                epsilon = Epsilon(funcs[-max_t:], kappas[-max_t:])
                minim = minimize(epsilon.eps, w_kc0, method='bfgs')
                error = error_beta(minim.x,kappas[-max_t:],x_mins[-max_t:]) 
                print( kappas[-max_t]  )
                print(minim)
                if minim.success == True:
                        if error > 0:
                                w_kcs=minim.x 
                                values.append([ w.item() for w in w_kcs])
                                errors.append(error.item())
                                grads.append([d.item() for d in minim.grad])
                                max_ts.append(max_t)
                else:
                        print("the minimization did not converge")
        min_err = torch.argmin(torch.tensor(errors))
        print(errors[min_err])
        print(errors)
        w_kcs=torch.tensor(values)[min_err]
        plot_beta(w_kcs,kappas,x_mins,funcs)
        max_t = max_ts[min_err]
        grad=torch.tensor(grads)[min_err]
        epsilon = Epsilon(  funcs[-max_t:],kappas[-max_t:])
        error_x = epsilon.error_x(w_kcs)
        print(error_x)
        error_beta = torch.sqrt(error_x/(grad[0]**2)/3.)
        error_Tc = torch.sqrt(error_x/(grad[1]**2)/3.)
        error_c = torch.sqrt(error_x/(grad[2]**2)/3.)

        print( "beta = %.4f +/- %.4f"  % (w_kcs[0].item(), error_beta.item()))
        print( "kc = %.4f +/- %.4f"    % (w_kcs[1].item(), error_Tc.item()))
        print( "const = %.4f +/- %.4f" % (w_kcs[2].item()**2, error_c.item()))






