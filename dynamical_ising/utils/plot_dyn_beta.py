import matplotlib.pyplot as plt
import argparse
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
from mu_ode_solve_ising import Data
from utils.weighted_model import AvgsODEFunc, ODEFunc 

parser = argparse.ArgumentParser('ODE ising')
parser.add_argument('--N', type=int, default=128)
args = parser.parse_args()

def plot_beta(Ts,n_train,dataet):
        with torch.no_grad():
                x_mins=torch.zeros(len(Ts))
                y_mins=torch.zeros(len(Ts))
                x_mins_all = torch.zeros(len(Ts),n_train)
                funcs = nn.ModuleList([AvgsODEFunc(n_train,torch.zeros(n_train)) for _ in range(len(Ts))])
                x_0=torch.zeros(len(Ts))
                fig,( ax1,ax2) = plt.subplots(1,2)
                fig.set_size_inches(w=4.774/1.35, h=2.9841875/2*1.5)
                fig.tight_layout(pad=1.6)
                cmap = mpl.cm.get_cmap('rainbow')
                rgba = cmap(np.linspace(0, 1, len(Ts)))
                n_dx = 600
                losses = torch.zeros(len(Ts),n_train)
                for idx in range(len(Ts)):
                        model_name = "weighted_mu_net.ckpt"
                        model_dir = "./dump/t_100000_ising_mu_%.3f_N_128"%Ts[idx]

                        model_path = os.path.join(model_dir,model_name)
                        device = "cpu"
                        ckpt = torch.load(model_path,map_location=torch.device('cpu'))
                        funcs[idx].load_state_dict(ckpt['model'])
                        funcs[idx].weights =ckpt['weights']
                        x = torch.linspace(0,1.0,n_dx).reshape(n_dx,1)
                        mu = funcs[idx].netto(x.to(device)).flatten().cpu()
                        y = torch.cumsum(-1/n_dx* mu ,dim=0)
                        x_0[idx] = x[torch.argmin(torch.abs(mu))]
                        x_mins[idx] = x[torch.argmin(y   )]
                        y_mins[idx] = y[torch.argmin(y)]
                        losses[idx,:]=1/ckpt['weights']
                        for jdx in range(n_train):
                                mu = funcs[idx].funcs[jdx].netto(x.to(device)).flatten().cpu()
                                y = torch.cumsum(-1/n_dx* mu ,dim=0)
                                x_mins_all[idx,jdx] = x[torch.argmin(y   )]
                #ax3.scatter(Ts,x_mins,s =5)
                yerr = torch.sqrt(torch.sum( losses**(-1) *  (   ( x_mins_all.T - x_mins )**2).T,dim =1)  / torch.sum( losses**(-1),dim =1  )  )
                #ax3.errorbar(
                #                    Ts,
                #                    x_mins,
                #                    linestyle='none',
                #                    yerr = yerr,
                #                    )
                l_max = losses.max()
                l_min = losses.min()
                
                norm = plt.Normalize(l_min,l_max)
                for i in range(len(Ts)):
                   for j in range(n_train):
                         lox = ((losses[i,j]-l_min)/l_max).item()
                         ax2.scatter(Ts[i],x_mins_all[i,j],s =5,c=cmap(lox))




                idx = 10

                file_name =  "./dataset/N_%d_T_%.4f_num_traj_1000t_max_100000.h5"%(args.N,Ts[idx])
                data = Data(file_name,device = device)
                j1 = 1 
                j2 = 5

                pred_y1 = odeint(funcs[idx].funcs[j1], data.x[0].to(device), data.time.to(device))
                pred_y2 = odeint(funcs[idx].funcs[j2], data.x[0].to(device), data.time.to(device))
                ax1.plot(data.time,data.x,label = "data")
                lox = ((losses[idx,j1]-l_min)/l_max).item()
                ax1.plot(data.time,pred_y1,label = "network", c = cmap(lox),linestyle="dashed")
                lox = ((losses[idx,j2]-l_min)/l_max).item()
                ax1.plot(data.time,pred_y2,c=cmap(lox),linestyle="dashed")
                ax1.set_ylabel(r"$\bar m$")
                ax1.set_xlabel(r"$t$")

                ax1.legend()


                # compute and plot the best estimate for beta
                kcs = np.linspace(3.2,3.4  ,100)
                Ts = [float(Ts[i]) for i in torch.nonzero(x_mins)]
                yerr = np.array( [yerr.clone()[i].item() for i in torch.nonzero(x_mins)])
                x_mins = np.array( [x_mins.clone()[i].item() for i in torch.nonzero(x_mins)])

                err = np.zeros(100)
                err_up = np.zeros(100)
                err_down = np.zeros(100)
                kcs = np.linspace(2.255,2.27  ,100)
                p_xs = np.zeros(100)
                intercepts = np.zeros(100)
                res_xs = np.zeros(100)
                kcs_up = np.linspace(2.255,2.27  ,100)
                kcs_down = np.linspace(2.255,2.27  ,100)
                for idx in range(100):
                        kc = kcs[idx]
                        p_x,res_x,_,_,_ = np.polyfit(np.log(kc-np.array(Ts)),np.log(x_mins),1,full=True)
                        p_xs[idx] = p_x[0]
                        intercepts[idx] = p_x[1] 
                        res_xs[idx] = res_x
                        p_x_up,res_x_up,_,_,_ = np.polyfit(np.log(np.abs(np.array(Ts)-kc)),np.log(x_mins+yerr),1,full=True)
                        p_x_down,res_x_down,_,_,_ = np.polyfit(np.log(np.abs(np.array(Ts)-kc)),np.log(x_mins-yerr),1,full=True)
                        diff = p_x[0] * np.log(np.abs(np.array(Ts)-kc)) + p_x[1] -np.log(x_mins)
                        diff_up = p_x_up[0] * np.log(np.abs(np.array(Ts)-kc)) + p_x_up[1] -np.log(x_mins)
                        diff_down = p_x_down[0] * np.log(np.abs(np.array(Ts)-kc)) + p_x_down[1] -np.log(x_mins)
                        err[idx] = (diff**2).sum()#np.sign( diff[0]  ) + np.sign(diff[-1])
                        err_up[idx] = (diff_up**2).sum()#np.sign( diff[0]  ) + np.sign(diff[-1])
                        err_down[idx] = (diff_down**2).sum()#np.sign( diff[0]  ) + np.sign(diff[-1])
                err_k = np.max(np.abs(np.array([kcs[np.argmin(err)] - kcs[np.argmin(err_up)], 
                        kcs[np.argmin(err)] - kcs[np.argmin(err_down)]])))
                kc = kcs[np.argmin(err)]
                beta = p_xs[np.argmin(err)]
                itcpt = intercepts[np.argmin(err)]
                ax2.plot(Ts,np.exp(itcpt)*(kc-np.array(Ts))**beta,c="orange")

                print("T_c = %f +/-  %f" %  (kcs[np.argmin(err)].item(0), err_k.item(0)  ))
                print("beta = %f +/- %f"%(p_xs[np.argmin(err)], res_xs[np.argmin(err)] ))


                ax2.set_xlabel(r"$T$")
                ax2.set_ylabel(r"$\bar m_{\rm stat}$")
                #ax2.xaxis.set_label_coords(0.5,-0.05)

                cbar2 = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
                                ax = ax2,
                                #orientation="horizontal",
                                )
                cbar2.ax.set_title(r'$L_{\rm stat}$')
                cbar2.ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

                #ax3.set_xlabel(r"$\kappa$")
                #ax3.set_ylabel(r"$\bar \rho_{\rm stat}$")

                ax1.text(-0.1, 1.1, "(a)", transform=ax1.transAxes)
                ax2.text(-0.1, 1.1, "(b)", transform=ax2.transAxes)
                #ax3.text(-0.1, 1.1, "(c)", transform=ax3.transAxes)
                plt.savefig("./images/test_beta.pdf")
                

if __name__ == "__main__":
    with torch.no_grad():
        device = "cpu"
        n_train = 37
        Ts = np.array([
        2.0,
        2.014157894736842,
        2.0283157894736843,
        2.042473684210526,
        2.0566315789473686,
        2.0707894736842105,
        2.084947368421053,
        2.099105263157895,
        2.1132631578947367,
        2.127421052631579,
        2.141578947368421,
        2.1557368421052634,
        2.1698947368421053,
        2.1840526315789477,
        2.1982105263157896,
        2.2123684210526315,
        2.226526315789474,
        2.240684210526316,
        2.254842105263158,
        #2.269,
                ])
        plt.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
            "font.size":11
         })
        dataset = "../../Autoencoder/HGN/nn_sde/c_time_con_proc/"
        plot_beta(Ts,n_train,dataset)

