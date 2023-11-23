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
from mu_ode_solve import Data
from utils.weighted_model import AvgsODEFunc, ODEFunc 
def plot_z(kappas,n_train,dataset,kc=3.31,fit = True):
        funcs = nn.ModuleList([AvgsODEFunc(n_train,torch.zeros(n_train)) for _ in range(len(kappas))])
        #funcs = nn.ModuleList([ODEFunc() for _ in range(len(kappas))])
        x_0=torch.zeros(len(kappas))
        x_mins=torch.zeros(len(kappas))

        y_mins=torch.zeros(len(kappas))
        fig,( ax,ax1) = plt.subplots(1,2)
        fig.set_size_inches(w=4.7747, h=2.9841875/2*1.7)
        fig.tight_layout(pad=3.)
        cmap = mpl.cm.get_cmap('rainbow')
        rgba = cmap(np.linspace(0, 1, len(kappas)))
        n_dx = 600
        for idx in range(len(kappas)):
                model_name = "weighted_mu_net.ckpt"
                model_dir = "./dump/N_100_cp_mu_"+str(kappas[idx])
                #model_name = "global_step_4.ckpt"
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
        for k,x in zip( kappas, x_0   ):
                print(k,x.item())
        kappas = [float(kappas[i]) for i in torch.nonzero(x_mins)]
        x_0 = [float(x_0[i]) for i in torch.nonzero(x_mins)]
        print("no zeros")
        for k,x in zip( kappas, x_0   ):
                print(k,x)
        times = torch.zeros(len(kappas))
        times_ = torch.zeros(len(kappas))
        for idx in range(len(kappas)):
                file_name  = dataset+"/N_100_perc_1.0_nsteps_1000_inf_rate_"+str(kappas[idx])+"_rec_rate_1_surv_prob_False_.h5"
                t_max = 1000
                #data = Data(file_name,device = device)
                data = Data(file_name,device = device)
                pred_y = odeint(funcs[idx], data.x[0].to(device), data.time.to(device))
                for t in range(t_max-1):
                        if abs(  data.x[t]-  x_mins[idx] )<0.015:
                                times[idx] = data.time[t]
                                break
                for t in range(t_max-1): 
                        if abs(  pred_y[t]- x_mins[idx] )<0.015:
                                times_[idx] = data.time[t]
                                break
                ax.plot( data.time[:t],data.x[:t], color=rgba[ idx  ] )
                #ax.scatter(times[idx], x_0[idx],color = "grey",s=5)
                ax.plot( data.time[:t],pred_y[:t], color=rgba[ idx  ] )
                ax.set_xlabel(r"$t$")
                ax.set_ylabel(r"$\rho(t)$")
                ax.scatter( times_[idx],x_0[idx]   ,color = rgba[ idx  ])
                print(x_0[idx],pred_y[t])
        ax1.scatter(kappas,times,s=5)
        ax1.set_xlabel(r"$\kappa$")
        ax1.set_ylabel(r"$\tau$")
        norm = plt.Normalize(min(kappas),max(kappas))
        cbar2 = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm))
        cbar2.ax.set_title(r'$T$')
        z,res_z,_,_,_ = np.polyfit(np.log(np.abs(kc-np.array(kappas))),np.log(times),1,full=True)
        print("from data z=%.3f x/- %.3f"%(z[0].item(0),res_z.item(0)))
        ax1.plot(kappas,np.exp(z[1] )* np.abs(kc-np.array(kappas))**z[0])


        ax1.scatter(kappas,times_,s=5)
        z,res_z,_,_,_ = np.polyfit(np.log(np.abs(kc-np.array(kappas))),np.log(times_),1,full=True)
        print("z=%.3f x/- %.3f"%(z[0].item(0),res_z.item(0)))
        ax1.plot(kappas,np.exp(z[1] )* np.abs(kc-np.array(kappas))**z[0])

        plt.savefig("./images/trajs.pdf")

if __name__ == "__main__":
    with torch.no_grad():
        device = "cpu"
        n_train = 10
        kappas = np.array([
       # 3.29,               
       # 3.3144827586206898, 
       # 3.3389655172413795 ,
       # 3.363448275862069  ,
       # 3.3879310344827585 ,
        3.412413793103448  ,
        3.436896551724138  ,
        3.4613793103448276 ,
        3.4858620689655173 ,
        3.510344827586207  ,
        3.5348275862068967 ,
        3.559310344827586  ,
        3.5837931034482757 ,
        3.6082758620689654 ,
        3.632758620689655  ,
        3.657241379310345  ,
        3.6817241379310346 ,
        3.7062068965517243 ,
        3.730689655172414  ,
        3.7551724137931033 ,
        3.779655172413793  ,
        3.8041379310344827 ,
        3.8286206896551724 ,
        3.853103448275862  ,
        3.8775862068965514,
        3.902068965517241,
        3.926551724137931,
        3.9510344827586206,
        3.9755172413793103,
        4.0,
                ])
        kappas_ = np.array([
                3.4858620689655173 ,
                3.8775862068965514,
                3.9510344827586206,
                3.9755172413793103])
        plt.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
            "font.size":11
         })
        dataset = "../../Autoencoder/HGN/nn_sde/c_time_con_proc/"
        plot_z(kappas,n_train,dataset,fit = False)

