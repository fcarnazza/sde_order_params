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
from mu_ode_solve import Data
from utils.weighted_model import AvgsODEFunc, ODEFunc 

def plot_beta(kappas,n_train,dataet):
        with torch.no_grad():
                x_mins=torch.zeros(len(kappas))
                y_mins=torch.zeros(len(kappas))
                x_mins_all = torch.zeros(len(kappas),n_train)
                funcs = nn.ModuleList([AvgsODEFunc(n_train,torch.zeros(n_train)) for _ in range(len(kappas))])
                x_0=torch.zeros(len(kappas))
                fig,( ax1,ax2) = plt.subplots(1,2)
                fig.set_size_inches(w=4.774/1.35, h=2.9841875/2*1.5)
                fig.tight_layout(pad=1.6)
                cmap = mpl.cm.get_cmap('rainbow')
                rgba = cmap(np.linspace(0, 1, len(kappas)))
                n_dx = 600
                losses = torch.zeros(len(kappas),n_train)
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
                        losses[idx,:]=1/ckpt['weights']
                        for jdx in range(n_train):
                                mu = funcs[idx].funcs[jdx].netto(x.to(device)).flatten().cpu()
                                y = torch.cumsum(-1/n_dx* mu ,dim=0)
                                x_mins_all[idx,jdx] = x[torch.argmin(y   )]
                #ax3.scatter(kappas,x_mins,s =5)
                yerr = torch.sqrt(torch.sum( losses**(-1) *  (   ( x_mins_all.T - x_mins )**2).T,dim =1)  / torch.sum( losses**(-1),dim =1  )  )
                #ax3.errorbar(
                #                    kappas,
                #                    x_mins,
                #                    linestyle='none',
                #                    yerr = yerr,
                #                    )
                l_max = losses.max()
                l_min = losses.min()
                
                norm = plt.Normalize(l_min,l_max)
                for i in range(len(kappas)):
                   for j in range(n_train):
                         lox = ((losses[i,j]-l_min)/l_max).item()
                         ax2.scatter(kappas[i],x_mins_all[i,j],s =5,c=cmap(lox))




                idx = 10

                file_name  = dataset+"/N_100_perc_1.0_nsteps_1000_inf_rate_"+str(kappas[idx])+"_rec_rate_1_surv_prob_False_.h5"
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
                ax1.set_ylabel(r"$\bar \rho$")
                ax1.set_xlabel(r"$t$")

                ax1.legend()


                # compute and plot the best estimate for beta
                kcs = np.linspace(3.2,3.4  ,100)
                kappas = [float(kappas[i]) for i in torch.nonzero(x_mins)]
                yerr = np.array( [yerr.clone()[i].item() for i in torch.nonzero(x_mins)])
                x_mins = np.array( [x_mins.clone()[i].item() for i in torch.nonzero(x_mins)])
                err = np.zeros(100)
                err_up = np.zeros(100)
                err_down = np.zeros(100)

                p_xs = np.zeros(100)
                itcps = np.zeros(100)
                res_xs = np.zeros(100)
                kcs_up = np.linspace(3.2,3.4  ,100)
                kcs_down = np.linspace(3.2,3.4  ,100)
                for idx in range(100):
                        kc = kcs[idx]
                        p_x,res_x,_,_,_ = np.polyfit(np.log(np.array(kappas)-kc),np.log(x_mins),1,full=True)
                        p_xs[idx] = p_x[0]
                        itcps[idx]=p_x[1]
                        res_xs[idx] = res_x
                        p_x_up,res_x_up,_,_,_ = np.polyfit(np.log(np.array(kappas)-kc),np.log(x_mins+yerr),1,full=True)
                        p_x_down,res_x_down,_,_,_ = np.polyfit(np.log(np.array(kappas)-kc),np.log(x_mins-yerr),1,full=True)
                        diff = p_x[0] * np.log(np.array(kappas)-kc) + p_x[1] -np.log(x_mins)
                        diff_up = p_x_up[0] * np.log(np.array(kappas)-kc) + p_x_up[1] -np.log(x_mins)
                        diff_down = p_x_down[0] * np.log(np.array(kappas)-kc) + p_x_down[1] -np.log(x_mins)
                        err[idx] = (diff**2).sum()#np.sign( diff[0]  ) + np.sign(diff[-1])
                        err_up[idx] = (diff_up**2).sum()#np.sign( diff[0]  ) + np.sign(diff[-1])
                        err_down[idx] = (diff_down**2).sum()#np.sign( diff[0]  ) + np.sign(diff[-1])
                err_k = np.max(np.abs(np.array([kcs[np.argmin(err)] - kcs[np.argmin(err_up)], 
                        kcs[np.argmin(err)] - kcs[np.argmin(err_down)]])))
                kc = kcs[np.argmin(err)]
                beta = p_xs[np.argmin(err)]
                itcpt = itcps[np.argmin(err)]
                ax2.plot(kappas,np.exp(itcpt)*(np.array(kappas)-kc)**beta,c="orange")

                print("kappa_c = %f +/-  %f" %  (kcs[np.argmin(err)].item(0), err_k.item(0)  ))
                print("beta = %f +/- %f"%(p_xs[np.argmin(err)], res_xs[np.argmin(err)] ))


                ax2.set_xlabel(r"$\kappa$")
                ax2.set_ylabel(r"$\bar \rho_{\rm stat}$")
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
        n_train = 10
        kappas = np.array([
        3.29,               
        3.3144827586206898, 
        3.3389655172413795 ,
        3.363448275862069  ,
        3.3879310344827585 ,
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
        plt.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
            "font.size":11
         })
        dataset = "../../Autoencoder/HGN/nn_sde/c_time_con_proc/"
        plot_beta(kappas,n_train,dataset)

