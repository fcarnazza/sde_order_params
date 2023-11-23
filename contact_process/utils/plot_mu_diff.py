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
from mu_diff import Mu 


def plot_collapse(
                kappas,
                dataset,
                name,
                w_min = 1.,
                w_max = 1.5,
                k_min=3.,
                k_max = 3.2,
                n_dx = 200,
                ):

        funcs = nn.ModuleList([Mu() for _ in range(len(kappas))])
        x_0=torch.zeros(len(kappas))
        x_mins=torch.zeros(len(kappas))
 
        y_mins=torch.zeros(len(kappas))
        fig,( ax,ax1) = plt.subplots(1,2)
        fig.set_size_inches(w=4.7747/1.35, h=2.9841875*0.65)
        fig.tight_layout(pad=2)
        cmap = mpl.cm.get_cmap('rainbow')
        rgba = cmap(np.linspace(0, 1, len(kappas)))
        for idx in range(len(kappas)):
                model_name = "mu_net.ckpt"
                model_dir = "./dump/N_100_cp_mu_"+str(kappas[idx])
                #model_name = "global_step_4.ckpt"
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
                ax.plot(x,y,color=rgba[ idx  ])

        # compute and plot the best estimate for beta

        kcs = np.linspace(3.1,3.15  ,100)
        kappas = np.array( [float(kappas[i]) for i in torch.nonzero(x_mins)])
        x_mins = np.array( [x_mins.clone()[i].item() for i in torch.nonzero(x_mins)])
        err = np.zeros(100)

        p_xs = np.zeros(100)
        itcps = np.zeros(100)
        res_xs = np.zeros(100)
        for idx in range(100):
                kc = kcs[idx]
                p_x,res_x,_,_,_ = np.polyfit(  np.log(np.array(kappas)-kc),np.log(x_mins),1,full=True)
                p_xs[idx] = p_x[0]
                itcps[idx]=p_x[1]
                res_xs[idx] = res_x
                diff = p_x[0] * np.log(np.array(kappas)-kc) + p_x[1] -np.log(x_mins)
                err[idx] = (diff**2).sum()#np.sign( diff[0]  ) + np.sign(diff[-1])
        kc = kcs[np.argmin(err)]
        beta = p_xs[np.argmin(err)]
        itcpt = itcps[np.argmin(err)]
        ax1.plot(kappas,np.exp(itcpt)*(np.array(kappas)-kc)**beta,c="orange")

        ax.text(-0.4, 1.1, "(a)", transform=ax.transAxes)
        ax1.text(-0.4, 1.1, "(b)", transform=ax1.transAxes)

        print("kappa_c = %f +/-  " %  (kcs[np.argmin(err)].item(0), ))
        print("beta = %f +/- %f"%(p_xs[np.argmin(err)], res_xs[np.argmin(err)] ))
        norm = plt.Normalize(min(kappas),max(kappas))
        cbar2 = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
                        ax=ax,
                        )

        cbar2.ax.set_title(r'$\kappa$')
        ax1.scatter(kappas,x_mins,s=5)
        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel(r"$M_\theta^{\kappa_i}(\rho)$")
        ax1.set_xlabel(r"$\kappa$")
        ax1.set_ylabel(r"$\bar\rho_{\rm stat}$")
        plt.savefig("./images/"+name)
        plt.close()
        fig,( ax) = plt.subplots(1,1)
        fig.set_size_inches(w=4.7747/1.35, h=2.9841875*0.65)
        fig.tight_layout(pad=2.)
        ax.plot(kcs,err)
        ax.set_xlabel(r"$\kappa_{\rm c}$")
        ax.set_ylabel(r"${\rm error}$")
        ax.text(-0.4, 1.1, "(a)", transform=ax.transAxes)
        ax1.text(-0.4, 1.1, "(b)", transform=ax1.transAxes)
        plt.savefig("./images/error_k_mu.pdf")
        print("some stuff has been plotted")
        plt.close()
        y_avgss = torch.zeros(len(kappas),n_dx)
        rho_max = 1/x_mins[np.argmax(x_mins  )]
        for idx in range(len(kappas)):
                        kappa = kappas[idx]
                        x = torch.linspace(0,rho_max,200).reshape(200,1)
                        x_ = torch.linspace(0.4,0.65,200).reshape(200,1)

                        mu_ = funcs[idx](x_.to(device)  ).flatten().cpu()
                        y = torch.cumsum(-1/200* mu ,dim=0)
                        y_ = torch.cumsum(-1/200* mu_ ,dim=0)
                        x_sc = torch.linspace(0,1,200).reshape(200,1)

                        mu = funcs[idx](x.to(device) *x_mins[idx] ).flatten().cpu()
                        y_avgss[idx] = torch.cumsum(-(rho_max-1)*x_mins[idx]/200* mu ,dim=0)

        dim =100
        dist_w_kc = torch.zeros(dim,dim)
        ws = np.linspace(w_min, w_max, dim)
        kapps_c = np.linspace(k_min, k_max, dim)
        fig, (ax0,ax) = plt.subplots(1, 2)
        fig.set_size_inches(w=4.7747/1.35, h=2.9841875*0.65)
        fig.tight_layout(pad=2.)
        for idx1 in range(len(ws)):
          w = ws[idx1]
          for idx2 in range(len( kapps_c  )):
                kc = kapps_c[idx2]
                w_y = torch.zeros(n_dx)
                w_y_av = torch.zeros(n_dx)
                for i1 in range(len(kappas)):
                    y_min1 = abs(kappas[i1]-kc)**w
                    for i2 in range(i1,len(kappas)):
                                    y_min2 = abs(kappas[i2]-kc)**w
                                    av = 0.5*torch.abs(y_avgss[i1]/y_min1+y_avgss[i2]/y_min2)
                                    diff = torch.abs( y_avgss[i1]/y_min1-y_avgss[i2]/y_min2  )
                                    w_y_av = w_y_av +av

                                    a = y_avgss[i1]/y_min1
                                    b = y_avgss[i2]/y_min2
                                    diff = torch.abs( (a-b)/torch.mean(a+b) )
                                    w_y = w_y + diff
                dist_w_kc[idx1,idx2] = w_y.max()/len(kappas)
        m = int(torch.argmin(dist_w_kc)/dim)
        n = int(torch.argmin(dist_w_kc)%dim)
        kc = kapps_c[n]
        w = ws[m]
        print(kc,w)
 
        ax0.ticklabel_format(style="sci",scilimits=(0,0))
        ax1.get_xaxis().get_offset_text().set_position((0,0))
        for i1 in range(len(kappas)):
                    ax0.plot(x,y_avgss[i1]/(np.abs(kappas[i1]-kc)**w), color=rgba[ i1  ])
        extent = [k_min-kc,k_max-kc, w_max,w_min]
        pos = ax.imshow(dist_w_kc,extent = extent, aspect = "auto",interpolation=None)
        ax.set_ylabel(r"$w$")
        ax.set_xlabel(r"$\kappa_{\rm c}-\tilde \kappa_{\rm c}$")
        ax.scatter( kc-kapps_c[n]  , ws[m],color="red",s=5)
        cbar2 = fig.colorbar(pos)
        cbar2.ax.set_title(r'$\epsilon$')
        cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
                        ax=ax0,
                        )

        cbar.ax.set_title(r'$\kappa$')
        ax0.text(-0.4, 1.1, "(a)", transform=ax0.transAxes)
        ax.text(-0.4, 1.1, "(b)", transform=ax.transAxes)
        ax0.set_xlabel(r"$x$")
        ax0.ticklabel_format(style="sci",scilimits=(0,0))
        ax0.set_ylabel(r"$W_i^{w,\kappa_{\rm c}}(x)$")

        plt.savefig("./images/imshow_kc.pdf")
        plt.close()
        fig,( ax,ax1) = plt.subplots(1,2)
        fig.set_size_inches(w=4.7747/1.35, h=2.9841875*0.65)
        fig.tight_layout(pad=2.)
        cmap = mpl.cm.get_cmap('rainbow')
        rgba = cmap(np.linspace(0, 1, len(kappas)))
        for idx in range(len(kappas)):
                x = torch.linspace(0,1.0,n_dx).reshape(n_dx,1)
                mu = funcs[idx](x.to(device)).flatten().cpu()
                y = torch.cumsum(-1/n_dx* mu ,dim=0)
                ax.plot(x,y,color=rgba[ idx  ])

        # plot beta computed from the kc of the collpse
        ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        p_x,res_x,_,_,_ = np.polyfit( np.log(np.abs(np.array( kappas[kappas> kc]   )-kc)),np.log(x_mins[kappas>kc]),1,full=True)

        ax1.plot(kappas[kappas>kc]-kc,np.exp(p_x[1])*(np.array(kappas[kappas>kc])-kc)**p_x[0],c="orange")
        print(p_x[0])

        ax.text(-0.4, 1.1, "(a)", transform=ax.transAxes)
        ax1.text(-0.4, 1.1, "(b)", transform=ax1.transAxes)

        print("beta = %f +/- %f"%(p_x[0], res_x ))
        norm = plt.Normalize(min(kappas),max(kappas))
        cbar2 = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
                        ax=ax,
                        )

        cbar2.ax.set_title(r'$\kappa$')
        ax1.scatter(kappas-kc,x_mins,s=5)
        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel(r"$M_\theta^{\kappa_i}(\rho)$")
        ax1.set_xlabel(r"$\kappa-\kappa_{\rm c}$")
        ax1.set_ylabel(r"$\bar\rho_{\rm stat}$")
        plt.savefig("./images/"+name)
        plt.close()

        fig,( ax) = plt.subplots(1,1)
        fig.set_size_inches(w=4.7747/1.35, h=2.9841875*0.65)
        fig.tight_layout(pad=1.6)
        ax.plot(kcs,err)
        ax.set_xlabel(r"$\kappa_{\rm c}$")
        ax.set_ylabel(r"${\rm error}$")
        ax.text(-0.4, 1.1, "(a)", transform=ax.transAxes)
        ax1.text(-0.4, 1.1, "(b)", transform=ax1.transAxes)
        plt.savefig("./images/error_k_mu.pdf")


def plot_w(kappas,
           dataset,
           name,
           w_min = 1.,
           w_max = 2.5,
           k_min=3.001,
           k_max = 3.3,
           n_dx = 200,
                ):
        """
        load the traind models for mu, compute the effective potentials, 
        compute collapse function e and minimize it.
        """
        # mu fucntions to be loaded
        funcs = nn.ModuleList([Mu() for _ in range(len(kappas))])
        x_0=torch.zeros(len(kappas))
        # minima of the potentatials 
        x_mins=torch.zeros(len(kappas))
        y_mins=torch.zeros(len(kappas))
        fig,( ax,ax1) = plt.subplots(1,2)
        fig.set_size_inches(w=4.7747/1.35, h=2.9841875*0.65)
        fig.tight_layout(pad=2.)
        cmap = mpl.cm.get_cmap('rainbow')
        rgba = cmap(np.linspace(0, 1, len(kappas)))
        for idx in range(len(kappas)):
                model_name = "mu_net.ckpt"
                model_dir = "./dump/N_100_cp_mu_"+str(kappas[idx])
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
        y_avgss = torch.zeros(len(kappas),n_dx)
        # extrema of where to minimze to ensure proper domain
        rho_max = 1/x_mins[np.argmax(x_mins  )]
        for idx in range(len(kappas)):
                        kappa = kappas[idx]
                        x = torch.linspace(0,rho_max,200).reshape(200,1)
                        x_ = torch.linspace(0.4,0.65,200).reshape(200,1)

                        mu_ = funcs[idx](x_.to(device)  ).flatten().cpu()
                        y = torch.cumsum(-1/200* mu ,dim=0)
                        y_ = torch.cumsum(-1/200* mu_ ,dim=0)
                        x_sc = torch.linspace(0,1,200).reshape(200,1)

                        mu = funcs[idx](x.to(device) *x_mins[idx] ).flatten().cpu()
                        y_avgss[idx] = torch.cumsum(-(rho_max-1)*x_mins[idx]/200* mu ,dim=0)

        dim =100
        dist_w_kc = torch.zeros(dim,dim)
        ws = np.linspace(w_min, w_max, dim)
        kapps_c = np.linspace(k_min, k_max, dim)
        fig, (ax0,ax) = plt.subplots(1, 2)
        fig.set_size_inches(w=4.7747/1.35, h=2.9841875*0.65)
        fig.tight_layout(pad=2.)
        for idx1 in range(len(ws)):
          w = ws[idx1]
          for idx2 in range(len( kapps_c  )):
                kc = kapps_c[idx2]
                w_y = torch.zeros(n_dx)
                w_y_av = torch.zeros(n_dx)
                for i1 in range(len(kappas)):
                    y_min1 = abs(kappas[i1]-kc)**w
                    for i2 in range(i1,len(kappas)):
                                    y_min2 = abs(kappas[i2]-kc)**w
                                    av = 0.5*torch.abs(y_avgss[i1]/y_min1+y_avgss[i2]/y_min2)
                                    diff = torch.abs( y_avgss[i1]/y_min1-y_avgss[i2]/y_min2  )
                                    w_y_av = w_y_av +av

                                    a = y_avgss[i1]/y_min1
                                    b = y_avgss[i2]/y_min2
                                    diff = torch.abs( (a-b)/torch.mean(a+b) )
                                    w_y = w_y + diff
                dist_w_kc[idx1,idx2] = w_y.max()/len(kappas)

        m = int(torch.argmin(dist_w_kc)/dim)
        n = int(torch.argmin(dist_w_kc)%dim)
        kc = kapps_c[n]
        w = ws[m]
        print(kc,w)
 
        ax0.ticklabel_format(style="sci",scilimits=(0,0))
        ax1.get_xaxis().get_offset_text().set_position((0.,0.))
        for i1 in range(len(kappas)):
                    ax0.plot(x,y_avgss[i1]/(np.abs(kappas[i1]-kc)**w), color=rgba[ i1  ])
        extent = [k_min-kc,k_max-kc, w_max,w_min]
        pos = ax.imshow(dist_w_kc,extent = extent, aspect = "auto",interpolation=None)
        ax.set_ylabel(r"$w$")
        ax.set_xlabel(r"$\kappa_{\rm c}-\tilde \kappa_{\rm c}$")
        ax.scatter( kc-kapps_c[n]  , ws[m],color="red",s=5)
        cbar2 = fig.colorbar(pos)
        cbar2.ax.set_title(r'$\epsilon$')
        norm = plt.Normalize(min(kappas),max(kappas))
        cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
                        ax=ax0,
                        )

        cbar.ax.set_title(r'$\kappa$')
        ax0.text(-0.4, 1.1, "(a)", transform=ax0.transAxes)
        ax.text(-0.4, 1.1, "(b)", transform=ax.transAxes)
        ax0.set_xlabel(r"$x$")
        ax0.ticklabel_format(style="sci",scilimits=(0,0))
        ax0.set_ylabel(r"$W_i^{w,\kappa_{\rm c}}(x)$")

        plt.savefig("./images/imshow_kc.pdf")
        plt.close()
        fig,( ax,ax1) = plt.subplots(1,2)
        fig.set_size_inches(w=4.7747/1.35, h=2.9841875*0.65)
        fig.tight_layout(pad=2)
        cmap = mpl.cm.get_cmap('rainbow')
        rgba = cmap(np.linspace(0, 1, len(kappas)))
        for idx in range(len(kappas))[0::4]:
                x = torch.linspace(0,1.0,n_dx).reshape(n_dx,1)
                mu = funcs[idx](x.to(device)).flatten().cpu()
                y = torch.cumsum(-1/n_dx* mu ,dim=0)
                ax.plot(x,y,color=rgba[ idx  ])

        # compute and plot the best estimate for beta

        kcs = np.linspace(3.1,3.15  ,100)

        p_x,res_x,_,_,_ = np.polyfit( np.log(np.array(kappas[kappas> kc])-kc),np.log(x_mins[kappas> kc]),1,full=True)
        ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        ax1.plot(kappas[kappas>kc]-kc,np.exp(p_x[1])*(kappas[kappas> kc]-kc)**p_x[0],c="orange")
        print(p_x[0])

        ax.text(-0.4, 1.1, "(a)", transform=ax.transAxes)
        ax1.text(-0.4, 1.1, "(b)", transform=ax1.transAxes)

        print("beta = %f +/- %f"%(p_x[0], res_x ))
        cbar2 = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
                        ax=ax,
                        )

        cbar2.ax.set_title(r'$\kappa$')
        ax1.scatter(kappas-kc,x_mins,s=5)
        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel(r"$M_\theta^{\kappa_i}(\rho)$")
        ax1.set_xlabel(r"$\kappa-\kappa_{\rm c}$")
        ax1.set_ylabel(r"$\bar\rho_{\rm stat}$")
        plt.savefig("./images/"+name)
        plt.close()
        print("plot for bet plotted")
        return p_x, res_x


if __name__ == "__main__":
    with torch.no_grad():
        device = "cpu"
        n_train = 10
        kappas = np.sort(np.array([
        3.0,
        3.0332222222222223,
        3.0664444444444445,
        3.099666666666667,
        3.132888888888889,
        3.166111111111111,
        3.199333333333333,
        3.2325555555555554,
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
        # I didn't trained on this, but the data is there:
        #3.9510344827586206,
        #3.9755172413793103,
        #4.0,
                ]))
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
        p_x,res_x = plot_w(kappas,dataset,name="G_collapse_cp_mu.pdf")

