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


from scipy.interpolate import interp1d 

def collapse_inversion(kappas,dataset = "./dataset/"):
    with torch.no_grad():
        
        fig, (ax1,ax2) = plt.subplots(1, 2)
        fig.set_size_inches(w=4.7747, h=2.9841875/2*1.7)
        fig.tight_layout(pad=3.)
        cmap = mpl.cm.get_cmap('rainbow')
        norm = plt.Normalize(min(kappas),max(kappas))
        rgba = cmap(np.linspace(0, 1, len(kappas)))
        y_avgss = torch.zeros(len(kappas),999)
        x_avgss = torch.zeros(len(kappas),999)
        for idx in range(len(kappas)):
                kappa = kappas[idx]
                file_name  = dataset+"N_100_perc_1.0_nsteps_1000_inf_rate_"+str(kappa)+"_rec_rate_1_surv_prob_False_.h5" 
                data = Data(file_name,device = device)
                x_dot = (-data.x.flatten()[:-1]+data.x.flatten()[1:])
                ax1.plot(data.x[1:].flatten().flip((0) )/data.x[-1],x_dot.flatten().flip((0)  ) ,color=rgba[ idx  ])
                kappa = kappas[idx]
                file_name  = dataset+"N_100_perc_1.0_nsteps_1000_inf_rate_"+str(kappa)+"_rec_rate_1_surv_prob_False_.h5" 
                data = Data(file_name,device = device)
                #ax2.plot(data.x[1:].flatten().flip((0) )/data.x[-1],x_dot.flatten().flip((0))/(kappa-3.29)**2.23   ,color=rgba[ idx  ])
                y_avgss[idx,:]=x_dot.flatten().flip((0))
                x_avgss[idx,:]=data.x[1:].flatten().flip((0) )/data.x[-1]
        #interpolation to get the mus on the same domain
        max_y = y_avgss[len(kappas)-1,-1]
        max_x = x_avgss[len(kappas)-1,-1]
        for idx in range(len(kappas)-1):
                difference_array = np.absolute(y_avgss[idx,:]-max_y)
                index = difference_array.argmin()
                f = interp1d( x_avgss[idx,:],y_avgss[idx,:]  )
                domain = np.linspace(1,max_x,999)
                y_avgss[idx,:] = torch.from_numpy(f(domain))/((kappas[idx]-3.29)**2.23)#this is husta a check, the values are roughly the xpc'd ones
                ax2.plot(domain, y_avgss[idx,:],color=rgba[ idx  ])

        cbar2 = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm))
        cbar2.ax.set_title(r'$\kappa$')
        ax1.set_xlabel(r"$\rho(t)/\rho_{\rm stat}$")
        ax1.set_ylabel(r"$\rho'(t)$")
        ax2.set_xlabel(r"$\rho(t)/\rho_{\rm stat}$")
        ax2.set_ylabel(r"$\rho'(t)\ |\kappa-\kappa_c|^w$")
        print("mu plotted")


        plt.savefig("./images/collapse_inversion.pdf")

        dim =100
        dist_w_kc = torch.zeros(dim,dim)
        w_min = 2.0
        w_max = 2.3
        k_min = 3.29 
        k_max = 3.3
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
                w_y = torch.zeros(999)
                w_y_av = torch.zeros(999)
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
               ax.scatter(kapps_c[torch.argmin(dist_w_kc[i])], ws[i],color = "green") 
               kapps_min[i] =  kapps_c[torch.argmin(dist_w_kc[i])]
               dist_kapps[i] = dist_w_kc[i][torch.argmin(dist_w_kc[i])]
        pos = ax.imshow(dist_w_kc,extent = extent, aspect = "auto",interpolation=None)
        ax.set_ylabel(r"$w$")
        ax.set_xlabel(r"$\kappa_c$")
        ax.scatter( kapps_c[n]  , ws[m],color="red")
        print("kc")
        print(kapps_c[n])
        cbar2 = fig.colorbar(pos)
        cbar2.ax.set_title(r'$\epsilon$')
        plt.savefig("./images/imshow_kc_inversion.pdf")


if __name__ == "__main__":
    with torch.no_grad():
        device = "cpu"
        n_train = 10
        kappas = np.array([
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
        collapse_inversion(kappas,dataset)
