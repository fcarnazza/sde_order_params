import matplotlib.pyplot as plt
import h5py
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
from mu_diff import Mu 
def zeros(x):
        z = []
        for i in range(len(x)):
                if x[i] == 0:
                        z.append(i)
        return z


def plot_collapse(Ts,
                name_of_fig,
                fit = False,
                n_dx = 200, # size on which to integrate mu to get M, the potenetial
                w_min = 0.1,
                w_max = 2., 
                k_min=2.26900001, # min temp in the collapse
                k_max = 2.27, # max temp in the collapse
                N=128, # lattice size = N x N
                every=1, #every how many curves of mu to plot, not to have too clogged plots
                ):
        """
        Plot the profiles of the learned drift at different temperatures,
        1. Use its minima to find the best Tc and beta,
        2. Find the best Tc and w collapsing them, and then use 
           Tc to compute beta yet again. It would be nice to interpret w 
           in terms of other critical exponents.
        Arguments:
        Ts: list of temperatures
        name_of_fig: how to name the produces figure
        """

        #load the functions and compute the minima

        funcs = nn.ModuleList([Mu() for _ in range(len(Ts))])
        x_0=torch.zeros(len(Ts))
        x_mins=torch.zeros(len(Ts))
        y_mins=torch.zeros(len(Ts))
        y_avgss = torch.zeros(len(Ts),n_dx)
        # This is for the colorz 
        cmap = mpl.cm.get_cmap('rainbow')
        rgba = cmap(np.linspace(0, 1, len(Ts)))
        for idx in range(len(Ts)):
                model_name = "mu_net.ckpt"
                model_dir = "./dump/t_100000_ising_mu_%.3f_N_%d"%(Ts[idx], N )
                model_path = os.path.join(model_dir,model_name)
                device = "cpu"
                ckpt = torch.load(model_path,map_location=torch.device('cpu'))
                funcs[idx].load_state_dict(ckpt['model'])
                x = torch.linspace(0,1.0,n_dx).reshape(n_dx,1)
                mu = funcs[idx](x.to(device)).flatten().cpu()
                y = torch.cumsum(-1/n_dx* mu ,dim=0)
                x_0[idx] = x[torch.argmin(torch.abs(mu))]
                x_mins[idx] = x[torch.argmin(y   )]
                y_mins[idx] = y[torch.argmin(y)]

        # collapse the functions
        nonzero_idx = torch.nonzero(x_mins)
        zero_idx = zeros(x_mins)

        Ts0 = [float(Ts[i]) for i in zero_idx]
        Ts1 = [float(Ts[i]) for i in nonzero_idx]


        y_avgss = torch.zeros(len(Ts),n_dx)
        with torch.no_grad():
                for idx in nonzero_idx:#range(len(Ts)):
                        x = torch.linspace(0,1, n_dx  ).reshape( n_dx  ,1)
                        mu = funcs[idx](x.to(device) *x_mins[idx] ).flatten().cpu()
                        y = torch.cumsum(- x_mins[idx]/n_dx * mu ,dim=0)
                        y_avgss[idx] = y
        
        norm = plt.Normalize(min(Ts),max(Ts))
        x_mins0 = np.array( [x_mins.clone()[i].item() for i in zero_idx ])
        x_mins = np.array( [x_mins.clone()[i].item() for i in nonzero_idx ])



        dim =100
        dist_w_kc = torch.zeros(dim,dim)
        ws = np.linspace(w_min, w_max, dim)
        kapps_c = np.linspace(k_min, k_max, dim)
        fig, (ax0,ax) = plt.subplots(1, 2)
        fig.set_size_inches(w=4.7747/1.35, h=2.9841875*0.65)
        fig.tight_layout(pad=2.5)
        for idx1 in range(len(ws)):
          w = ws[idx1]
          for idx2 in range(len( kapps_c  )):
                kc = kapps_c[idx2]
                w_y = torch.zeros(n_dx)
                w_y_av = torch.zeros(n_dx)
                count = 0
                for i1 in nonzero_idx:#range(len(Ts)):
                    y_min1 = abs(Ts[i1]-kc)**w
                    for i2 in nonzero_idx[nonzero_idx>i1]:    #range(i1+1,len(Ts)):
                                    y_min2 = abs(Ts[i2]-kc)**w
                                    a = y_avgss[i1]/y_min1
                                    b = y_avgss[i2]/y_min2
                                    diff = torch.abs( (a-b)/torch.mean(a+b) )
                                    w_y = w_y + diff 
                dist_w_kc[idx1,idx2] = w_y.max()
        m = int(torch.argmin(dist_w_kc)/dim)
        n = int(torch.argmin(dist_w_kc)%dim)
        kc = kapps_c[n]
        w = ws[m]
        for i1 in nonzero_idx:# range(len(Ts)):
                    ax0.plot(x.flatten(),y_avgss[i1].flatten()/(np.abs(Ts[i1]-kc)**w), color=rgba[ i1  ])
        kapps_min = torch.zeros(dim)
        dist_kapps = torch.zeros(dim)
        print( [ dist_w_kc[i][torch.argmin(dist_w_kc[i])] for i in range(dim)])
        for i in range(dim):
               #ax.scatter(kapps_c[torch.argmin(dist_w_kc[i])], ws[i],color = "green") 
               kapps_min[i] =  kapps_c[torch.argmin(dist_w_kc[i])]
               dist_kapps[i] = dist_w_kc[i][torch.argmin(dist_w_kc[i])]
        extent = [k_min-kc,k_max-kc, w_max,w_min]
        pos = ax.imshow(dist_w_kc,extent = extent, aspect = "auto",interpolation=None)

        ax0.ticklabel_format(style="sci",scilimits=(0,0))
        ax.ticklabel_format(style="sci",scilimits=(0,0))

        ax.set_ylabel(r"$w$")
        ax.set_xlabel(r"$T_{\rm c}^* - T_{\rm c}$")
        #ax0.set_ylabel(r"$M^{T_i}_\theta( x \bar m_{\rm stat})/|T-T_{\rm c}|^w$")
        ax0.set_ylabel(r"$W_i^{w,T_{\rm c}}( x )$")
        ax0.set_xlabel(r"$x$")
        ax.scatter( kapps_c[n]-kc  , ws[m],color="red",s=5)
        cbar2 = fig.colorbar(pos)
        cbar2.ax.set_title(r'$\epsilon$')
        cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
                        ax=ax0,
                        )

        cbar.ax.set_title(r'$T$')
        print("Tc= %f"%kapps_c[n].item())
        T_sub = np.array(Ts1)[np.array(Ts1)<kc]
        T_sup = np.array(Ts1)[np.array(Ts1)>kc]
        x_mins_sub = np.array(x_mins)[np.array(Ts1)<kc]
        x_mins_sup = np.array(x_mins)[np.array(Ts1)>kc]
        p_x,res_x,_,_,_ = np.polyfit( np.log(np.abs(np.array( T_sub   )-kc)),np.log(x_mins_sub),1,full=True)
        print("beta= %f"%p_x.item(0))
        print("w= %f"%w.item())
        ax.set_ylabel(r"$w$")
        ax.set_xlabel(r"$\tilde T_{\rm c} - T_{\rm c}$")
        ax0.text(-0.4, 1.1, "(a)", transform=ax0.transAxes)
        ax.text(-0.4, 1.1, "(b)", transform=ax.transAxes)
        ax.get_xaxis().get_offset_text().set_position((0.,0))

        plt.savefig("./images/imshow_Tc_"+str(N)+".pdf")
        plt.close()

        # plot the scatter of the minima with the beta interpolating them 

        fig,( ax,ax1) = plt.subplots(1,2)
        fig.set_size_inches(w=4.7747/1.35, h=2.9841875*0.65)
        fig.tight_layout(pad=2.5)
        kcs = np.linspace(2.2691,2.275  ,100)
        #Ts = [float(Ts[i]) for i in torch.nonzero(x_mins)]
        #x_mins = np.array( [x_mins.clone()[i].item() for i in torch.nonzero(x_mins)])
        err = np.zeros(100)
        for idx in nonzero_idx[0::every]:#range(len(Ts)):
                x = torch.linspace(0,1.0,n_dx).reshape(n_dx,1)
                mu = funcs[idx](x.to(device)).flatten().cpu()
                y = torch.cumsum(-1/n_dx* mu ,dim=0)
                x_0[idx] = x[torch.argmin(torch.abs(mu))]
                ax.plot(x,y,color=rgba[ idx  ])
        for idx in zero_idx[0::every]:#range(len(Ts)):
                x = torch.linspace(0,1.0,n_dx).reshape(n_dx,1)
                mu = funcs[idx](x.to(device)).flatten().cpu()
                y = torch.cumsum(-1/n_dx* mu ,dim=0)
                x_0[idx] = x[torch.argmin(torch.abs(mu))]
                ax.plot(x,y,color=rgba[ idx  ])

        p_xs = np.zeros(100)
        itcps = np.zeros(100)
        res_xs = np.zeros(100)
        if fit:
                for idx in range(100):
                        kc = kcs[idx]
                        p_x,res_x,_,_,_ = np.polyfit( np.log(np.abs(np.array(Ts)-kc)),np.log(x_mins),1,full=True)
                        p_xs[idx] = p_x[0]
                        itcps[idx]=p_x[1]
                        res_xs[idx] = res_x
                        diff = p_x[0] * np.log(np.abs(np.array(Ts)-kc)) + p_x[1] -np.log(x_mins)
                        err[idx] = (diff**2).sum()#np.sign( diff[0]  ) + np.sign(diff[-1])
                kc = kcs[np.argmin(err)]
                beta = p_xs[np.argmin(err)]
                itcpt = itcps[np.argmin(err)]
                print("T_c = %f +/-  " %  (kcs[np.argmin(err)].item(0), ))
                print("beta = %f +/- %f"%(p_xs[np.argmin(err)], res_xs[np.argmin(err)] ))
        ax1.plot(np.abs(np.array(T_sub)-kc ) ,np.exp(p_x[1])*(np.abs(np.array(T_sub)-kc))**p_x[0],c="orange")

        cbar2 = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
                        ax=ax,
                        )

        cbar2.ax.set_title(r'$T$')

        ax1.ticklabel_format(style="sci",scilimits=(0,0))
        ax1.get_xaxis().get_offset_text().set_position((0.,0))
        ax1.scatter( np.abs(np.array(T_sub)-kc)   ,x_mins_sub,s=5,c= "blue")
        ax1.scatter( -np.abs(np.array(T_sup)-kc)   ,x_mins_sup,s=5,c="blue")
        ax1.scatter( -np.abs(np.array(Ts0)-kc)   ,x_mins0,s=5,c="blue")
        ax.set_xlabel(r"$m$")
        ax.set_ylabel(r"$M^{T_i}_\theta(m)$")
        ax1.set_xlabel(r"$|T-T_{\rm c}|$")
        ax1.set_ylabel(r"$\bar m_{\rm stat}$")

        ax.text(-0.4, 1.1, "(a)", transform=ax.transAxes)
        ax1.text(-0.4, 1.1, "(b)", transform=ax1.transAxes)
        plt.savefig("./images/"+name_of_fig)
        plt.close()

        print("something has been plotted")


if __name__ == "__main__":
    with torch.no_grad():
        device = "cpu"
        Ts = np.sort(np.array([
        2.2214137931034483,
        2.223793103448276,
        2.226172413793104,
        2.2285517241379313,
        2.230931034482759,
        2.2333103448275864,
        2.235689655172414,
        2.2380689655172414,
        2.240448275862069,
        2.2428275862068965,
        2.2452068965517245,
        2.247586206896552,
        2.252344827586207,
        2.2547241379310345,
        2.257103448275862,
        2.2594827586206896,
        2.2618620689655176,
        2.264241379310345,
        2.2666206896551726,
        2.269,
        2.279333333333333,
        2.2827777777777776,
        2.2862222222222224,
        2.2896666666666667,
        2.293111111111111,
        2.2965555555555555,
                ]))
        plt.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
            "font.size":11
         })

        name_file = "./dataset/temperaures_from_2.2024_to_2.3000_N_20.h5" 
        N = 256
        Ts = np.array(h5py.File(name_file,'r' )["temps"])[:-1]
        
        name_of_fig = "G_collapse_ising_mu_N_"+str(N)+".pdf"
        plot_collapse(Ts, name_of_fig = name_of_fig  ,N=256)

