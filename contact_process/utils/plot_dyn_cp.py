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
def plot_dyn(kappa,train1,train2,device,dataset):
        """
        plot the true dynamics and the one retrieved by the network for two different training train_1 and rain2
        """
        plt.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
            "font.size":11
         })
        file_name  = dataset+"N_100_perc_1.0_nsteps_1000_inf_rate_"+str(kappa)+"_rec_rate_1_surv_prob_False_.h5" 
        data = Data(file_name,device = device)
        model_dir = "./dump/N_100_cp_mu_"+str(kappa)
        model_name = "global_step_%d.ckpt"%train1
        model_path = os.path.join(model_dir,model_name)
        mult = 1.0
        device = "cpu"
        func = ODEFunc()
        ckpt = torch.load(model_path,map_location=torch.device('cpu'))
        func.load_state_dict(ckpt['model'])
        x = torch.linspace(0,1.0,200).reshape(200,1)
        mu = func.netto(x.to(device)).flatten().cpu()
        y = torch.cumsum(-1/200* mu ,dim=0)
        true_y = data.x
        pred_y = odeint(func, data.x[0].to(device), data.time.to(device))
        fig, (ax1,ax2) = plt.subplots(1, 2)
        fig.set_size_inches(w=4.7747, h=2.9841875/2*1.7)
        fig.tight_layout(pad=3.)
        ax1.plot(data.time.to(device).flatten(),true_y.flatten(),label = "Data")
        ax1.plot(data.time.to(device).flatten(),pred_y.flatten()  ,label = "Network",linestyle="dashed")
        ax1.set_xlabel(r"$t$")
        ax1.set_ylabel(r"$\bar{\rho}$")
        ax1.legend()
        ax1.text(-0.1, 1.1, "(a)", transform=ax1.transAxes)

        model_name = "global_step_%d.ckpt"%train2
        model_path = os.path.join(model_dir,model_name)
        mult = 1.0
        device = "cpu"
        func = ODEFunc()
        ckpt = torch.load(model_path,map_location=torch.device('cpu'))
        func.load_state_dict(ckpt['model'])
        x = torch.linspace(0,1.0,200).reshape(200,1)
        mu = func.netto(x.to(device)).flatten().cpu()
        y = torch.cumsum(-1/200* mu ,dim=0)
        true_y = data.x
        pred_y = odeint(func, data.x[0].to(device), data.time.to(device))
        ax2.scatter(data.time.to(device).flatten()[-1],  x[torch.argmin(torch.abs(mu))])
        print(kappa ,x[torch.argmin(torch.abs(mu))],pred_y[-1]  )

        ax2.plot(data.time.to(device).flatten(),true_y.flatten(),label = "Data")
        ax2.plot(data.time.to(device).flatten(),pred_y.flatten()  ,label = "Network",linestyle="dashed")
        ax2.set_xlabel(r"$t$")
        ax2.set_ylabel(r"$\bar{\rho}$")
        ax2.legend()
        ax2.text(-0.1, 1.1, "(b)", transform=ax2.transAxes)

        plt.savefig("./images/dyn.pdf")

        fig, (ax) = plt.subplots(1, 1)
        fig.set_size_inches(w=4.7747, h=2.9841875/2*1.7)
        fig.tight_layout(pad=3.)
        x_dot = (-data.x.flatten()[:-1]+data.x.flatten()[1:])/0.01
        ax.plot(data.x[1:].flatten().flip((0) ),x_dot.flatten().flip((0)  ),label = "Data")
        x = torch.linspace( data.x.min(),data.x.max() ,1000 ).reshape(1000,1)
        mu = func.netto(x.to(device)).flatten().cpu()
        ax.plot(x,mu.flatten(),label = "Network",linestyle="dashed")
        ax.legend()
        ax.set_xlabel(r"$\rho(t)$")
        ax.set_ylabel(r"$\rho'(t)$")

        plt.savefig("./images/x_dot.pdf")


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
        return torch.mean((pred_y.flatten()[-10:] - x[torch.argmin(y)]    )**2), x[torch.argmin(y)]



def plot_beta(kappas,n_train,dataset,save_model=False):
        """
        This function provide for the estimate of kappa_c and beta 
        and gives the respective plots, and the reference models
        """
        cmap = mpl.cm.get_cmap('rainbow')
        rgba = cmap(np.linspace(0, 1, len(kappas)))
        funcs = nn.ModuleList([AvgsODEFunc(n_train,torch.zeros(n_train)) for _ in range(len(kappas))])
        x_mins_all = torch.zeros(len(kappas),n_train)
        x_mins = torch.zeros(len(kappas))
        losses = torch.zeros(len(kappas),n_train)
        for idx in range(len(kappas)):
                kappa = kappas[idx]
                file_name  = dataset+"/N_100_perc_1.0_nsteps_1000_inf_rate_"+str(kappas[idx])+"_rec_rate_1_surv_prob_False_.h5"
                for train in range(n_train):
                        model_dir = "./dump/N_100_cp_mu_"+str(kappas[idx])
                        model_name = "global_step_%d.ckpt"%train
                        model_path = os.path.join(model_dir,model_name)
                        loss, x_min = loss_stat(model_path,file_name)
                        losses[idx,train] = loss
                        x_mins_all[idx,train] = x_min
                weights = losses[idx,:]**(-1)
                func = AvgsODEFunc(n_train,weights)
                for train in range(n_train):
                        model_name = "global_step_%d.ckpt"%train
                        model_dir = "./dump/N_100_cp_mu_"+str(kappas[idx])
                        model_path = os.path.join(model_dir,model_name)
                        device = "cpu"
                        ckpt = torch.load(model_path,map_location=torch.device('cpu'))
                        func.funcs[train].load_state_dict(ckpt['model'])
                funcs[idx] = func
                x = torch.linspace(0,1.0,200).reshape(200,1)
                mu = func.netto(x.to(device)).flatten().cpu()
                y = torch.cumsum(-1/200* mu ,dim=0)
                x_mins[idx] = x[torch.argmin(y)]
                if save_model:
                        torch.save(
                            {'model': func.state_dict(),
                             'weights':weights,
                             },
                            os.path.join(model_dir, f'weighted_mu_net.ckpt')
                            )
        ### Plot for the minima ###

        kappas = [float(kappa) for kappa in kappas]
        fig, (ax2,ax1) = plt.subplots(1, 2)
        fig.set_size_inches(w=4.7747/1.5, h=2.9841875/2*1.7)
        fig.tight_layout(pad=2.)
        ax1.scatter(kappas,x_mins,s=5)
        yerr = torch.sqrt(torch.sum( losses**(-1) *  (   ( x_mins_all.T - x_mins )**2).T,dim =1)  / torch.sum( losses**(-1),dim =1  )  )
        ax1.errorbar(
                                    kappas,
                                    x_mins,
                                    linestyle='none',
                                    yerr = yerr,
                                    )
 
        for i in range(len(kappas)):
                print(kappas[i],x_mins[i])
        ax1.set_xlabel(r"$\kappa$")
        ax1.set_ylabel(r"$\bar{\rho}_{\rm stat}$")
 
        
 
        lox_max = losses.max()
        lox_min = losses.min()
        norm = plt.Normalize(losses.min(),losses.max())
        for i in range(len(kappas)):
                for j in range(n_train):
                 lox = ((losses[i,j]-lox_min)/lox_max).item()
                 ax2.scatter(kappas[i],x_mins_all[i,j],s =5,c=cmap(lox))
        cbar2 = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm))
        cbar2.ax.set_title(r'$L_{\rm stat}$')
        ax2.set_xlabel(r"$\kappa$")
        ax2.set_ylabel(r"$\bar{\rho}_{\rm stat}$")
        ax1.text(-0.1, 1.1, "(b)", transform=ax1.transAxes)
        ax2.text(-0.1, 1.1, "(a)", transform=ax2.transAxes)

        ### estimation of beta and kappa_c ###
        kappas = [float(kappas[i]) for i in torch.nonzero(x_mins)]
        yerr = np.array( [yerr.clone()[i].item() for i in torch.nonzero(x_mins)])
        x_mins = np.array( [x_mins.clone()[i].item() for i in torch.nonzero(x_mins)])

        err = np.zeros(100)
        err_up = np.zeros(100)
        err_down = np.zeros(100)
        kcs = np.linspace(3.2,3.4  ,100)
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
        ax1.plot(kappas,np.exp(itcpt)*(np.array(kappas)-kc)**beta)
        print("kappa_c = %f +/-  %f" %  (kcs[np.argmin(err)].item(0), err_k.item(0)  ))
        print("beta = %f +/- %f"%(p_xs[np.argmin(err)], res_xs[np.argmin(err)] ))
        plt.savefig("./images/scatter_x_mins_all.pdf")

        kappa_c = kcs[np.argmin(err)]
        err_kc = err_k
        beta = p_xs[np.argmin(err)]
        err_beta = res_xs[np.argmin(err)]

        #print(kcs[np.argmin(err_up)],err[np.argmin(err_up)]/len(kappas))
        #print(kcs[np.argmin(err_down)],err[np.argmin(err_down)]/len(kappas))
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(w=4.7747/1.5, h=2.9841875/2*1.7)
        fig.tight_layout(pad=3.)
        ax.plot(kcs,err)
        ax.plot(kcs_up,err_up)
        ax.plot(kcs_down,err_down)
        ax.set_xlabel(r"$\kappa_{\rm c}$")
        ax.set_ylabel(r"${d(\kappa_{\rm c} )}$")
        plt.savefig("./images/eucl_err_cp.pdf")
 
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(w=4.7747, h=2.9841875/2*1.7)
        fig.tight_layout(pad=3.)
        kappas = torch.tensor(kappas) 
        x_mins = torch.from_numpy(x_mins)
        return funcs, kappa_c, err_kc, beta, err_beta, x_mins 

def plot_collapse(funcs,kappas,x_mins,kappa_min=3.31):
        """
        This function provide for an estimate of the critical point and the leading rescaling
        expoenent w of the theory. Also an estimate for w at the critical point estimated via 
        beta is given.
        """
        if len(funcs) != len(kappas):
                raise ValueError("funcs and kappas must have same lengths")
        y_avgss = torch.zeros(len(kappas),200)
        rho_max = 1.5
        with torch.no_grad():
            for idx in range(len(kappas)):
                        x = torch.linspace(1,rho_max,200).reshape(200,1)
                        mu = funcs[idx].netto(x.to(device) *x_mins[idx] ).flatten().cpu()
                        y = torch.cumsum(-(rho_max-1)*x_mins[idx]/200* mu ,dim=0)
                        y_avgss[idx] = y

        dim =100
        dist_w_kc = torch.zeros(dim,dim)
        w_min = 1.5
        w_max = 2.0
        k_min = kappa_min
        k_max = 3.34
        ws = np.linspace(w_min, w_max, dim)
        kapps_c = np.linspace(k_min, k_max, dim)
        extent = [k_min,k_max, w_max,w_min]
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(w=4.7747/1.5, h=2.9841875/2*1.7)
        fig.tight_layout(pad=3.)
        for idx1 in range(len(ws)):
          w = ws[idx1]
          for idx2 in range(len( kapps_c  )):
                kc = kapps_c[idx2]
                w_y = torch.zeros(200)
                w_y_av = torch.zeros(200)
                for i1 in range(len(kappas)):
                    y_min1 = (kappas[i1]-kc)**w
                    for i2 in range(i1,len(kappas)):
                                    y_min2 = (kappas[i2]-kc)**w
                                    av = 0.5*torch.abs(y_avgss[i1]/y_min1+y_avgss[i2]/y_min2)
                                    diff = torch.abs( y_avgss[i1]/y_min1-y_avgss[i2]/y_min2  )
                                    w_y = w_y + diff#torch.abs( y_avgss[i1]/y_min1-y_avgss[i2]/y_min2  )/av
                                    w_y_av = w_y_av +av
                dist_w_kc[idx1,idx2] = w_y.max()/len(kappas)
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
        ax.set_xlabel(r"$\kappa_c$")
        ax.scatter( kapps_c[n]  , ws[m],color="red",s=5)
        cbar2 = fig.colorbar(pos)
        cbar2.ax.set_title(r'$\epsilon$')
        plt.savefig("./images/imshow_kc.pdf")

        print("kc")
        print(kapps_c[n])
        print("w")
        print( ws[m])
        print("w estimated drom the beta-estimated kappa_c")
        print(   ws[torch.argmin(dist_w_kc[0])]   )
        plt.close()
        fig, ax = plt.subplots(1, 1)
        ax.scatter(kapps_min,dist_kapps)
        ax.set_xlabel(r"$\kappa_{\rm min}$")
        ax.set_ylabel("r$\epsilon$")
        plt.savefig("./images/saddle.pdf")
        plt.close()
        return dist_w_kc, kapps_c[n], ws[m]

def plot_collapse_G(kappas,w,kc,funcs,x_mins,name="F_collapse.pdf"):
        y_avgss = torch.zeros(len(kappas),200)
        rho_max = 1.5
        kappas = [float(kappas[i]) for i in torch.nonzero(x_mins)]
        fig, (ax1,ax2,ax3) = plt.subplots(1, 3)
        fig.set_size_inches(w=4.7747, h=2.9841875/2*1.5)
        fig.tight_layout(pad=3.)
        y_avgss = torch.zeros(len(kappas),200)
        rho_max = 1.5
        cmap = mpl.cm.get_cmap('rainbow')
        norm = plt.Normalize(min(kappas),max(kappas))
        rgba = cmap(np.linspace(0, 1, len(kappas)))
        with torch.no_grad():
            for idx in range(len(kappas)):
                        kappa = kappas[idx]
                        x = torch.linspace(1,rho_max,200).reshape(200,1)
                        mu = funcs[idx].netto(x.to(device)  ).flatten().cpu()
                        y = torch.cumsum(-1/200* mu ,dim=0)
                        x_sc = torch.linspace(0,1,200).reshape(200,1)

                        mu = funcs[idx].netto(x.to(device) *x_mins[idx] ).flatten().cpu()
                        y_sc = torch.cumsum(-(rho_max-1)*x_mins[idx]/200* mu ,dim=0)
                        print(kappa, y_sc[-1].item())
                        ax1.plot(x[torch.argmin(y):],y[torch.argmin(y):],color=rgba[ idx  ])
                        ax1.plot(x,y,color=rgba[ idx  ],alpha=0.5)
                        ax2.plot(x,y_sc,color=rgba[ idx  ])
                        ax3.plot(x,y_sc/abs(kappas[idx]-kc)**w,color=rgba[ idx  ])
            ax1.set_xlabel(r"$\rho$")
            ax2.set_xlabel(r"$\rho'$")
            ax3.set_xlabel(r"$\rho'$")
            ax1.set_ylabel(r"$G(\rho)$")
            ax2.set_ylabel(r"$G(\rho')$")
            ax3.set_ylabel(r"$G(\rho') / |\kappa-\kappa_{\rm c}|^w$")
            ax1.text(-0.1, 1.1, "(a)", transform=ax1.transAxes)
            ax2.text(-0.1, 1.1, "(b)", transform=ax2.transAxes)
            ax3.text(-0.1, 1.1, "(c)", transform=ax3.transAxes)
            cbar2 = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm))
            cbar2.ax.set_title(r'$T$')
            name = "./images/"+name
            plt.savefig(name)



def plot_z(funcs,kappas,x_mins,kc,dataset):
        """
        This function provide for an estimate of the dynamical expoenent z
        and gives the respective plot.
        """
        t_max = 50000
        device = "cpu"
        times = np.zeros(len(kappas))
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(w=4.7747/1.5, h=2.9841875/2*1.7)
        fig.tight_layout(pad=3.)
        cmap = mpl.cm.get_cmap('rainbow')
        rgba = cmap(np.linspace(0, 1, len(kappas)))

        for idx in range(len(kappas)):
                file_name  = dataset+"/N_100_perc_1.0_nsteps_1000_inf_rate_"+str(kappas[idx])+"_rec_rate_1_surv_prob_False_.h5"
                data = Data(file_name,device = device,t_max = t_max,every=1)
                pred_y = odeint(funcs[idx], data.x[0].to(device), data.time.to(device))
                for t in range(t_max-1):
                        if abs(pred_y[t]-pred_y[t+1])<0.0001:
                                times[idx] = t
                                break
                ax.plot( data.time[:t],pred_y[:t], color=rgba[ idx  ] )
                
                z,res_z,_,_,_ = np.polyfit(np.log(kc-np.array(kappas)),np.log(times),1,full=True)
        plt.savefig("./images/trajs.pdf")
        plt.close()
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(w=4.7747/1.5, h=2.9841875/2*1.7)
        fig.tight_layout(pad=3.)
        ax.scatter(kappas,times)
        ax.set_xlabel("$T$")
        ax.set_ylabel(r"$\tau$")
        plt.savefig("./images/times.pdf")
        plt.close()
        return z, res_z

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
        kappa = kappas[np.random.randint(0,len(kappas))]
        print(kappa)
        train1  = np.random.randint(0,n_train)
        train2  = np.random.randint(0,n_train)
        print(train1,train2)
        kappa_bad = 3.730689655172414
        plot_dyn(kappa,train1,train2,device,dataset)
        funcs, kappa_c, err_kc, beta, err_beta, x_mins = plot_beta(kappas,n_train,dataset)

        dist_w_kci,kc,w = plot_collapse(funcs[-len(x_mins):],kappas[-len(x_mins):],x_mins,kappa_min =kappa_c)
        beta_from_kc, err_beta_from_kc,_,_,_ = np.polyfit(np.log(np.array(kappas[-len(x_mins):] -kc  )),np.log(x_mins),1,full=True)
        print("beta_from_kc = %4.f +/- %4.f"%(beta_from_kc.item(0), err_beta_from_kc.item(0)))
        plot_collapse_G(kappas,w,kc,funcs,x_mins,name="G_from_collapse.pdf") 
        #plot_z(funcs,kappas,x_mins,kc,dataset)
        exit()






