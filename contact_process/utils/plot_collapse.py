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

def plot_collapse_w(kappas,
                w_min = 1.5,
                w_max = 2.0,
                k_min=3.31,
                k_max = 3.34,
                ):
        """
        This function provide for an estimate of the critical point and the leading rescaling
        expoenent w of the theory. Also an estimate for w at the critical point estimated via 
        beta is given.
        """
        x_mins=torch.zeros(len(kappas))

        y_mins=torch.zeros(len(kappas))
        funcs = nn.ModuleList([AvgsODEFunc(n_train,torch.zeros(n_train)) for _ in range(len(kappas))])
        x_0=torch.zeros(len(kappas))
        fig,( ax1,ax2,ax3) = plt.subplots(1,3)
        fig.set_size_inches(w=4.7747/1.35, h=2.9841875/2*1.5)
        fig.tight_layout(pad=1.6)
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
        ws = np.linspace(w_min, w_max, dim)
        kapps_c = np.linspace(k_min, k_max, dim)
        extent = [k_min,k_max, w_max,w_min]
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(w=4.7747, h=2.9841875/2*1.7)
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

def plot_collapse(kappas,
                n_train,
                dataset,
                name,
                kc=3.323658810325477,
                w=1.8333333333333335,
                rho_max=1.5,
                w_min = 1.5,
                w_max = 2.0,
                k_min=3.31,
                k_max = 3.34,
                ):
        extent = [k_min,k_max, w_max,w_min]
        dist_w_kc,k_c_w,w_c = plot_collapse_w(kappas)

        funcs = nn.ModuleList([AvgsODEFunc(n_train,torch.zeros(n_train)) for _ in range(len(kappas))])
        x_0=torch.zeros(len(kappas))
        x_mins=torch.zeros(len(kappas))
 
        y_mins=torch.zeros(len(kappas))
        fig,( ax2,ax3,ax4) = plt.subplots(1,3)
        fig.set_size_inches(w=4.7747/1.35, h=2.9841875)
        fig.tight_layout(pad=1.6)
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
        kappas = [float(kappas[i]) for i in torch.nonzero(x_mins)]
        x_0 = [float(x_0[i]) for i in torch.nonzero(x_mins)]
        x_mins = x_0
        rho_max = 1/x_mins[np.argmax(x_mins  )]
        print(len(x_mins))
        for idx in range(len(kappas)):
                        kappa = kappas[idx]
                        x = torch.linspace(1,rho_max,200).reshape(200,1)
                        x_ = torch.linspace(0.4,0.65,200).reshape(200,1)

                        mu_ = funcs[idx].netto(x_.to(device)  ).flatten().cpu()
                        y = torch.cumsum(-1/200* mu ,dim=0)
                        y_ = torch.cumsum(-1/200* mu_ ,dim=0)
                        x_sc = torch.linspace(0,1,200).reshape(200,1)

                        mu = funcs[idx].netto(x.to(device) *x_mins[idx] ).flatten().cpu()
                        y_sc = torch.cumsum(-(rho_max-1)*x_mins[idx]/200* mu ,dim=0)
                        print("kappa %f, y_t_max %f, x_min %f"%(kappa, y_sc[-1].item(),x_mins[idx]))
                        #ax1.plot(x_[torch.argmin(y_):],y_[torch.argmin(y_):],color=rgba[ idx  ])
                        #ax1.plot(x_,y_,color=rgba[ idx  ],alpha=0.5)
                        ax2.plot(x,y_sc,color=rgba[ idx  ])
                        ax3.plot(x,y_sc/abs(kappas[idx]-kc)**w,color=rgba[ idx  ])
        #ax1.set_xlabel(r"$\bar \rho$")
        ax2.set_xlabel(r"$\bar \rho'$")
        ax3.set_xlabel(r"$\bar \rho'$")
        #ax1.set_ylabel(r"$G(\bar \rho)$")
        ax2.set_ylabel(r"$G(\bar \rho')$")
        ax3.set_ylabel(r"$G(\bar \rho') / |\kappa-\kappa_{\rm c}|^w$")
        #ax1.text(-0.1, 1.1, "(a)", transform=ax1.transAxes)
        ax2.text(-0.1, 1.1, "(a)", transform=ax2.transAxes)
        ax3.text(-0.1, 1.1, "(b)", transform=ax3.transAxes)
        norm = plt.Normalize(min(kappas),max(kappas))
        cbar2 = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
                        ax=(ax2,ax3),
                        orientation="horizontal",
                        pad=0.25,
                        aspect = 20
                        )
        cbar2.ax.set_xlabel(r'$\kappa$')
        ax4.ticklabel_format(style="sci")
        pos = ax4.imshow(dist_w_kc,extent = extent, aspect = "auto",interpolation="nearest")
        ax4.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax4.set_xticks( np.arange(k_min,k_max,0.019)  )

        ax4.set_ylabel(r"$w$")
        ax4.set_xlabel(r"$\kappa_{\rm c}$")
        ax4.scatter( k_c_w  , w_c,color="red",s=5)
        cbar1 = fig.colorbar(pos,ax = ax4,orientation="horizontal",aspect=8,pad = 0.25)
        cbar1.ax.set_xlabel(r'$\epsilon$')
        cbar1.ax.xaxis.set_ticks_position("bottom")
        cbar1.ax.xaxis.set_label_coords(0.5,-1.)
        cbar2.ax.xaxis.set_ticks_position("bottom")
        ax4.text(-0.1, 1.1, "(c)", transform=ax4.transAxes)
        name = "./images/"+name

        plt.savefig(name)

if __name__ == "__main__":
    with torch.no_grad():
        device = "cpu"
        n_train = 10
        kappas = np.array([
        #3.29,               
        #3.3144827586206898, 
        #3.3389655172413795 ,
        #3.363448275862069  ,
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
        plot_collapse(kappas,n_train,dataset,name="G_collapse_cp.pdf")

