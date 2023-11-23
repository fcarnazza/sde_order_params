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
from utils.weighted_model import AvgsODEFunc, ODEFunc 

def plot_collapse_w(Ts,
                w_min = 0.2,
                w_max = 0.5,
                k_min=2.3,
                k_max = 2.345,
                ):
        """
        This function provide for an estimate of the critical point and the leading rescaling
        expoenent w of the theory extracted from the dynamical function.
        Also an estimate for w at the critical point estimated via 
        beta is given.
        """
        x_mins=torch.zeros(len(Ts))

        y_mins=torch.zeros(len(Ts))
        funcs = nn.ModuleList([AvgsODEFunc(n_train,torch.zeros(n_train)) for _ in range(len(Ts))])
        x_0=torch.zeros(len(Ts))
        fig,( ax1,ax2,ax3) = plt.subplots(1,3)
        fig.set_size_inches(w=4.7747/1.35, h=2.9841875/2*1.5)
        fig.tight_layout(pad=1.6)
        cmap = mpl.cm.get_cmap('rainbow')
        rgba = cmap(np.linspace(0, 1, len(Ts)))
        n_dx = 600
        for idx in range(len(Ts)):
                model_name = "weighted_mu_net.ckpt"
                model_dir = "./dump/t_100000_ising_mu_%.3f_N_128"%Ts[idx]
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
        y_avgss = torch.zeros(len(Ts),200)
        rho_max = 1.5
        with torch.no_grad():
            for idx in range(len(Ts)):
                        x = torch.linspace(0,1,200).reshape(200,1)
                        mu = funcs[idx].netto(x.to(device) *x_mins[idx] ).flatten().cpu()
                        y = torch.cumsum(- x_mins[idx]/200* mu ,dim=0)
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

def plot_collapse(Ts,
                n_train,
                dataset,
                name,
                kc=3.323658810325477,
                w=1.8333333333333335,
                rho_max=1.5,
                w_min = 0.2,
                w_max = 0.5,
                k_min=2.3,
                k_max = 2.345,
                ):
        extent = [k_min,k_max, w_max,w_min]
        dist_w_kc,kc,w = plot_collapse_w(Ts)

        funcs = nn.ModuleList([AvgsODEFunc(n_train,torch.zeros(n_train)) for _ in range(len(Ts))])
        x_0=torch.zeros(len(Ts))
        x_mins=torch.zeros(len(Ts))
 
        y_mins=torch.zeros(len(Ts))
        fig,( ax2,ax3,ax4) = plt.subplots(1,3)
        fig.set_size_inches(w=4.7747/1.35, h=2.9841875)
        fig.tight_layout(pad=1.6)
        cmap = mpl.cm.get_cmap('rainbow')
        rgba = cmap(np.linspace(0, 1, len(Ts)))
        n_dx = 600
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

        Ts = [float(Ts[i]) for i in torch.nonzero(x_mins)]
        x_0 = [float(x_0[i]) for i in torch.nonzero(x_mins)]
        x_mins = x_0
        rho_max = 1/x_mins[np.argmax(x_mins  )]
        print(len(x_mins))
        for idx in range(len(Ts)):
                        kappa = Ts[idx]
                        x = torch.linspace(0,1.,200).reshape(200,1)
                        x_ = torch.linspace(0.,1.,200).reshape(200,1)

                        mu_ = funcs[idx].netto(x_.to(device)  ).flatten().cpu()
                        y = torch.cumsum(-1/200* mu ,dim=0)
                        y_ = torch.cumsum(-1/200* mu_ ,dim=0)
                        x_sc = torch.linspace(0,1,200).reshape(200,1)

                        mu = funcs[idx].netto(x.to(device) *x_mins[idx] ).flatten().cpu()
                        y_sc = torch.cumsum(-x_mins[idx]/200* mu ,dim=0)
                        print("kappa %f, y_t_max %f, x_min %f"%(kappa, y_sc[-1].item(),x_mins[idx]))
                        ax2.plot(x,y_sc,color=rgba[ idx  ])
                        ax3.plot(x,y_sc/abs(Ts[idx]-kc)**w,color=rgba[ idx  ])
        ax2.ticklabel_format(style="sci",scilimits=(0,0))
        ax3.ticklabel_format(style="sci",scilimits=(0,0))
        ax2.set_xlabel(r"$\bar m'$")
        ax3.set_xlabel(r"$\bar m'$")
        ax2.set_ylabel(r"$F(\bar m')$")
        ax3.set_ylabel(r"$F(\bar m') / |T-T_{\rm c}|^w$")
        ax2.text(-0.28, 1.1, "(a)", transform=ax2.transAxes)
        ax3.text(-0.28, 1.1, "(b)", transform=ax3.transAxes)
        norm = plt.Normalize(min(Ts),max(Ts))
        cbar2 = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
                        ax=(ax2,ax3),
                        orientation="horizontal",
                        pad=0.25,
                        aspect = 20
                        )
        #cbar2 = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),ax=ax3, orientation="horizontal", pad=0.2)
        cbar2.ax.xaxis.set_ticks_position("bottom")
        cbar2.ax.set_xlabel(r'$T$',labelpad=1.5)

        #ax4.ticklabel_format(style="sci")
        pos = ax4.imshow(dist_w_kc,extent = extent, aspect = "auto",interpolation="nearest")
        ax4.set_xticks( np.arange(k_min,k_max,0.03)  )

        ax4.set_ylabel(r"$w$")
        ax4.set_xlabel(r"$T_{\rm c}$")
        ax4.scatter( kc  , w,color="red",s=5)
        cbar1 = fig.colorbar(pos,ax = ax4,orientation="horizontal",aspect=6,pad = 0.25)
        cbar1.ax.xaxis.set_ticks_position("bottom")
        cbar1.ax.xaxis.set_label_coords(0.5,-0.5)
        cbar2.ax.xaxis.set_label_coords(0.5,-0.5)

        cbar1.ax.set_xlabel(r'$\epsilon$',labelpad=1.5)
        cbar1.formatter.set_powerlimits((0, 0))

        ax4.text(-0.28, 1.1, "(c)", transform=ax4.transAxes)
        name = "./images/"+name

        plt.savefig(name)

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
        plot_collapse(Ts,n_train,dataset,name="F_collapse_ising.pdf")

