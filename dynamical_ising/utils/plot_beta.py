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
from mu_ode_solve_ising import Data
from utils.weighted_model import AvgsODEFunc, ODEFunc 


def loss_stat(model_path,file_name):
        """
        compuets the variance between the ground truth value of the trajectory at final time
        and the minimum of the model. return tis variance and the minimum of the model.
        """
        data = Data(file_name,device = device)
        func = ODEFunc()
        ckpt = torch.load(model_path,map_location=torch.device('cpu'))
        func.load_state_dict(ckpt['model'])
        x = torch.linspace(0,1.0,200).reshape(200,1)
        mu = func.netto(x.to(device)).flatten().cpu()
        y = torch.cumsum(-1/200* mu ,dim=0)
        true_y = data.x
        pred_y = odeint(func, data.x[0].to(device), data.time.to(device))
        return torch.mean((pred_y.flatten()[-1:] - x[torch.argmin(y)]    )**2), x[torch.argmin(y)]



def plot_beta(Ts,n_train,dataset):
        """
        This function provide for the estimate of T_c and beta 
        and gives the respective plots, and the reference models
        """
        cmap = mpl.cm.get_cmap('rainbow')
        rgba = cmap(np.linspace(0, 1, len(Ts)))
        funcs = nn.ModuleList([AvgsODEFunc(n_train,torch.zeros(n_train)) for _ in range(len(Ts))])
        x_mins_all = torch.zeros(len(Ts),n_train)
        x_mins = torch.zeros(len(Ts))
        y_mins = torch.zeros(len(Ts))
        losses = torch.zeros(len(Ts),n_train)
        for idx in range(len(Ts)):
                T = Ts[idx]
                file_name =  dataset+"/N_128_T_%.4f_num_traj_1000t_max_100000.h5"%T
                #for train in range(30,30+n_train):
                for train in range(n_train):
                        model_dir = "./dump/t_100000_ising_mu_%.3f_N_128"%Ts[idx]
                        model_name = "global_step_%d.ckpt"%train
                        model_path = os.path.join(model_dir,model_name)
                        loss, x_min = loss_stat(model_path,file_name)
                        #losses[idx,train-30] = loss
                        losses[idx,train] = loss
                        #x_mins_all[idx,train-30] = x_min
                        x_mins_all[idx,train] = x_min
                weights = losses[idx,:]**(-1)
                func = AvgsODEFunc(n_train,weights)
                #for train in range(30,30+n_train):
                model_dir = "./dump/t_100000_ising_mu_%.3f_N_128"%Ts[idx]
                for train in range(n_train):
                        model_name = "global_step_%d.ckpt"%train
                        #model_name = "trained_model_%d.ckpt"%train
                        model_path = os.path.join(model_dir,model_name)
                        device = "cpu"
                        ckpt = torch.load(model_path,map_location=torch.device('cpu'))
                        #func.funcs[train-30].load_state_dict(ckpt['model'])
                        func.funcs[train].load_state_dict(ckpt['model'])
                torch.save(
                    {'model': func.state_dict(),
                     'weights':weights,
                     },
                    os.path.join(model_dir, f'weighted_mu_net.ckpt')
                    )
                funcs[idx]=func
                x = torch.linspace(0,1.0,200).reshape(200,1)
                mu = funcs[idx].netto(x.to(device)).flatten().cpu()
                y = torch.cumsum(-1/200* mu ,dim=0)
                x_mins[idx] = x[torch.argmin(y)]
                y_mins[idx] = y[torch.argmin(y)]

        ### Plot for the minima ###

        Ts = [float(T) for T in Ts]
        fig, (ax2,ax1) = plt.subplots(1, 2)
        fig.set_size_inches(w=4.7747/1.5, h=2.9841875/2*1.5)
        fig.tight_layout(pad=2.)
        ax1.scatter(Ts,x_mins,s=5)
        yerr = torch.sqrt(torch.sum( losses**(-1) *  (   ( x_mins_all.T - x_mins )**2).T,dim =1)  / torch.sum( losses**(-1),dim =1  )  )
        ax1.errorbar(
                                    Ts,
                                    x_mins,
                                    yerr = yerr,
                                    linestyle='none',
                                    )
 
        for i in range(len(Ts)):
                print(Ts[i],x_mins[i].item(),y_mins[i].item())
        ax1.set_xlabel(r"$T$")
        ax1.set_ylabel(r"$\bar{m}_{\rm stat}$")
 
        
 
        lox_max = losses.max()
        lox_min = losses.min()
        norm = plt.Normalize(losses.min(),losses.max())
        for i in range(len(Ts)):
                for j in range(n_train):
                 lox = ((losses[i,j]-lox_min)/lox_max).item()
                 ax2.scatter(Ts[i],x_mins_all[i,j],s =5,c=cmap(lox))
        cbar2 = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm))
        cbar2.ax.set_title(r'$L_{\rm stat}$')
        ax2.set_xlabel(r"$T$")
        ax2.set_ylabel(r"$\bar{m}_{\rm stat}$")
        ax1.text(-0.1, 1.1, "(b)", transform=ax1.transAxes)
        ax2.text(-0.1, 1.1, "(a)", transform=ax2.transAxes)

        ### estimation of beta and T_c ###
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
        print("T_c = %f +/-  %f" %  (kcs[np.argmin(err)].item(0), err_k.item(0)  ))
        print("beta = %f +/- %f"%(p_xs[np.argmin(err)], res_xs[np.argmin(err)] ))
        T_c = kcs[np.argmin(err)]
        err_kc = err_k
        beta = p_xs[np.argmin(err)]
        intercept = intercepts[np.argmin(err)]
        err_beta = res_xs[np.argmin(err)]
        ax1.plot( Ts,np.exp(intercept)*np.abs(np.array(Ts)-T_c)**beta ,label = "Linear fit")
        plt.savefig("./images/ising_beta.pdf")
        plt.close()
        #print(kcs[np.argmin(err_up)],err[np.argmin(err_up)]/len(Ts))
        #print(kcs[np.argmin(err_down)],err[np.argmin(err_down)]/len(Ts))
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(w=4.7747, h=2.9841875/2*1.7)
        fig.tight_layout(pad=3.)
        ax.plot(kcs,err)
        #ax.plot(kcs_up,err_up)
        #ax.plot(kcs_down,err_down)
        ax.set_xlabel(r"$T_{\rm c}$")
        ax.set_ylabel(r"${d(T_{\rm c} )}$")
        plt.savefig("./images/eucl_err_cp.pdf")
 
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(w=4.7747, h=2.9841875/2*1.7)
        fig.tight_layout(pad=3.)
        Ts = torch.tensor(Ts) 
        x_mins = torch.from_numpy(x_mins)
        return funcs, T_c, err_kc, beta, err_beta, x_mins 

def plot_collapse(funcs,Ts,x_mins,T_min=2.259848):
        """
        This function provide for an estimate of the critical point and the leading rescaling
        expoenent w of the theory. Also an estimate for w at the critical point estimated via 
        beta is given.
        """
        if len(funcs) != len(Ts):
                raise ValueError("funcs and Ts must have same lengths")
        y_avgss = torch.zeros(len(Ts),200)
        rho_max = 1.5
        with torch.no_grad():
            for idx in range(len(Ts)):
                        x = torch.linspace(0,1,200).reshape(200,1)
                        mu = funcs[idx].netto(x.to(device) *x_mins[idx] ).flatten().cpu()
                        y = torch.cumsum(- x_mins[idx]/200* mu ,dim=0)
                        y_avgss[idx] = y

        dim =200
        dist_w_kc = torch.zeros(dim,dim)
        w_min = 0.2
        w_max = 0.5
        k_max = 2.35
        k_min = 2.3#T_min
        ws = np.linspace(w_min, w_max, dim)
        kapps_c = np.linspace(k_min, k_max, dim)
        extent = [k_min,k_max, w_max,w_min]
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(w=4.7747/1.5, h=2.9841875/2*1.5)
        fig.tight_layout(pad=2.)
        for idx1 in range(len(ws)):
          w = ws[idx1]
          for idx2 in range(len( kapps_c  )):
                kc = kapps_c[idx2]
                w_y = torch.zeros(200)
                w_y_av = torch.zeros(200)
                for i1 in range(len(Ts)):
                    y_min1 = abs(Ts[i1]-kc)**w
                    for i2 in range(i1,len(Ts)):
                                    y_min2 = abs(Ts[i2]-kc)**w
                                    av = 0.5*torch.abs(y_avgss[i1]/y_min1+y_avgss[i2]/y_min2)
                                    diff = torch.abs( y_avgss[i1]/y_min1-y_avgss[i2]/y_min2  )
                                    w_y = w_y + diff#torch.abs( y_avgss[i1]/y_min1-y_avgss[i2]/y_min2  )/av
                                    w_y_av = w_y_av +av
                dist_w_kc[idx1,idx2] = w_y.max()/len(Ts)
                if torch.isnan(dist_w_kc[idx1,idx2]):
                        dist_w_kc[idx1,idx2] = 100
        m = int(torch.argmin(dist_w_kc)/dim)
        n = int(torch.argmin(dist_w_kc)%dim)
        kapps_min = torch.zeros(dim)
        dist_kapps = torch.zeros(dim)
        print( [ dist_w_kc[i][torch.argmin(dist_w_kc[i])] for i in range(dim)])
        for i in range(dim):
               #ax.scatter(kapps_c[torch.argmin(dist_w_kc[i])], ws[i],color = "green") 
               kapps_min[i] =  kapps_c[torch.argmin(dist_w_kc[i])]
               dist_kapps[i] = dist_w_kc[i][torch.argmin(dist_w_kc[i])]
        pos = ax.imshow(dist_w_kc,extent = extent, aspect = "auto",interpolation=None)
        ax.set_ylabel(r"$w$")
        ax.set_xlabel(r"$T_c$")
        ax.scatter( kapps_c[n]  , ws[m],color="red",s=5)
        cbar2 = fig.colorbar(pos)
        cbar2.ax.set_title(r'$\epsilon$')
        plt.savefig("./images/imshow_Tc.pdf")

        print("Tc")
        print(kapps_c[n])
        print("w")
        print( ws[m])
        print("w estimated drom the beta-estimated T_c")
        print(   ws[torch.argmin(dist_w_kc[:,0])]   )
        plt.close()
        fig, ax = plt.subplots(1, 1)
        ax.scatter(kapps_min,dist_kapps)
        ax.set_xlabel(r"$T_{\rm min}$")
        ax.set_ylabel("r$\epsilon$")
        plt.savefig("./images/saddle.pdf")
        plt.close()

        return  kapps_c[n], ws[m], ws[torch.argmin(dist_w_kc[:,0])]



def plot_collapse_F(Ts,w,Tc,funcs,x_mins,name="F_collapse.pdf"):
        fig, (ax1,ax2,ax3) = plt.subplots(1, 3)
        fig.set_size_inches(w=4.7747, h=2.9841875/2*1.5)
        fig.tight_layout(pad=3.)
        y_avgss = torch.zeros(len(Ts),200)
        rho_max = 1.5
        cmap = mpl.cm.get_cmap('rainbow')
        norm = plt.Normalize(min(Ts),max(Ts))
        rgba = cmap(np.linspace(0, 1, len(Ts)))
        with torch.no_grad():
            for idx in range(len(Ts)):
                        T = Ts[idx]
                        x = torch.linspace(0,1,200).reshape(200,1)
                        mu = funcs[idx].netto(x.to(device)  ).flatten().cpu()
                        y = torch.cumsum(-1/200* mu ,dim=0)
                        x_sc = torch.linspace(0,1,200).reshape(200,1)

                        mu_sc = funcs[idx].netto(x_sc.to(device) *x_mins[idx] ).flatten().cpu()
                        y_sc = torch.cumsum(- x_mins[idx] /200* mu_sc ,dim=0)
                        print(T, y_sc[-1].item())
                        ax1.plot(x[:torch.argmin(y)],y[:torch.argmin(y)],color=rgba[ idx  ])
                        ax1.plot(x,y,color=rgba[ idx  ],alpha=0.5)
                        ax2.plot(x,y_sc,color=rgba[ idx  ])
                        ax3.plot(x,y_sc/abs(Ts[idx]-Tc)**w,color=rgba[ idx  ])
            ax1.set_xlabel(r"$\bar m$")
            ax2.set_xlabel(r"$\bar m'$")
            ax3.set_xlabel(r"$\bar m'$")
            ax1.set_ylabel(r"$F(\bar m)$")
            ax2.set_ylabel(r"$F(\bar m')$")
            ax3.set_ylabel(r"$F(\bar m') / |T-T_{\rm c}|^w$")
            ax1.text(-0.1, 1.1, "(a)", transform=ax1.transAxes)
            ax2.text(-0.1, 1.1, "(b)", transform=ax2.transAxes)
            ax3.text(-0.1, 1.1, "(c)", transform=ax3.transAxes)
            cbar2 = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm))
            cbar2.ax.set_title(r'$T$')
            name = "./images/"+name
            plt.savefig(name)





def plot_z(funcs,Ts,x_mins,Tc):
        """
        This function provide for an estimate of the dynamical expoenent z
        and gives the respective plot.
        """
        t_max = 50000
        device = "cpu"
        times = np.zeros(len(Ts))
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(w=4.7747/1.5, h=2.9841875/2*1.7)
        fig.tight_layout(pad=3.)
        cmap = mpl.cm.get_cmap('rainbow')
        rgba = cmap(np.linspace(0, 1, len(Ts)))

        for idx in range(len(Ts)):
                file_name =  "./dataset/N_128_T_%.4f_num_traj_1000t_max_100000.h5"%Ts[idx]
                data = Data(file_name,device = device,t_max = t_max,every=1)
                pred_y = odeint(funcs[idx], data.x[0].to(device), data.time.to(device))
                for t in range(t_max-1):
                        if abs(pred_y[t]-pred_y[t+1])<0.0001:
                                times[idx] = t
                                break
                ax.plot( data.time[:t],pred_y[:t], color=rgba[ idx  ] )
                
                z,res_z,_,_,_ = np.polyfit(np.log(Tc-np.array(Ts)),np.log(times),1,full=True)
        plt.savefig("./images/trajs.pdf")
        plt.close()
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(w=4.7747/1.5, h=2.9841875/2*1.7)
        fig.tight_layout(pad=3.)
        ax.scatter(Ts,times)
        ax.set_xlabel("$T$")
        ax.set_ylabel(r"$\tau$")
        plt.savefig("./images/times.pdf")
        plt.close()
        return z, res_z

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
        dataset = "./dataset"
        funcs, T_c, err_kc, beta, err_beta, x_mins = plot_beta(Ts,n_train,dataset)
        T_c_c,w,beta_w=plot_collapse(funcs,Ts,x_mins,T_min=T_c)
        plot_collapse_F(Ts,w,T_c_c,funcs,x_mins,name="F_collapse_Ising.pdf") 
        plot_collapse_F(Ts,beta_w,T_c,funcs,x_mins,name="F_from_beta.pdf") 
        print("funcs plotted")
        z,res_z = plot_z(funcs,Ts,x_mins,T_c)
        print("z = %f +/- %f" % (z.item(0),res_z.item(0)))
        exit()






