import os 
from torch.utils.data import Dataset
import numpy as np
import h5py
import sys
import numpy
import torch
#from latent_sde_plot import plot
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Data_p_train(Dataset):

    def __init__(self, file_name,t_final,device,mult = 1.,every = 10,t_step=0.01):
        self.t_step = t_step
        ds = h5py.File(file_name,'r')
        self.t_final = t_final
        self.ds = h5py.File(file_name,'r')
        self.length = int(len(self.ds)/3)
        self.num = int(self.length)
        self.t_final = t_final
        self.device= device
        self.mult = mult
        self.every = every
        self.z_len = int(self.t_final/self.t_step)
        #this method returns a single datapoint and will later on be used to sample full batches
    def __getitem__(self, idx):
        data_point = self.ds["mag_"+str(idx % self.num + 1)]
        data_time = self.ds["T_"+str(idx % self.num + 1)]
        l_t = len(data_time)
        l_p = len(data_point)
        if  l_p == 1:
                z = torch.zeros(self.z_len,device = self.device)
                return self.mult * z.reshape(self.z_len,1)[0::self.every].float()
        else:
                set_interp = interp1d(data_time[:l_p], data_point[:l_t], kind='linear')
                time = [t for t in np.arange(0,data_time[:l_p][-1] ,self.t_step)] 
                interp_data_point = set_interp(time)
                v = torch.tensor(interp_data_point,device = self.device)
                z = torch.zeros(self.z_len,device = self.device)
                z[:len(v)] = v[:self.z_len]
                return self.mult * z.reshape(self.z_len,1)[0::self.every].float()
    
    def __len__(self):
        return self.num

class Data_sigma_p_train(Dataset):

    def __init__(self, file_name,t_final,device,mult = 1.,every = 1,t_step=0.01,n_trajs=30):
        self.t_step = t_step
        ds = h5py.File(file_name,'r')
        self.t_final = t_final
        self.ds = h5py.File(file_name,'r')
        self.length = int(len(self.ds)/3)
        self.num = int(self.length)
        self.t_final = t_final
        self.device= device
        self.mult = mult
        self.every = every
        self.n_trajs = n_trajs
        self.z_len = int(self.t_final/self.t_step)
        #this method returns a single datapoint and will later on be used to sample full batches
        self.dataset = torch.tensor([],device=device)
        # data are organised in tuples [mag(time), mag(time + delta_t), time, time+delta_t]
        for i in range(1, self.n_trajs +1 ):
            l_traj = len(torch.tensor(self.ds["mag_"+str(i)]))
            l_t = len(torch.tensor(self.ds["T_"+str(i)]))
            if l_t > 3:
                m = self.mult*torch.tensor(self.ds["mag_"+str(i)][:l_t],device = device)
                t = torch.tensor(self.ds["T_"+str(i)][:l_traj],device = device)[0::self.every]
                qv = torch.cumsum((m[:-1]-m[1:])**2,0)[0::self.every]
                m = m[0::self.every]
                dt = t[1:]-t[:-1]
                len_dt = len(dt)
                len_qv=len(qv)
                new = torch.cat(
                            (
                                qv[:-1].reshape(len_qv-1,1), # [X]_t time t 
                                qv[1:].reshape(len_qv-1,1), # [X]_t+dt time t + dt
                                m[:(len_qv-1)].reshape(len_qv-1,1), #X_t time t
                                dt[:(len_qv-1)].reshape(len_qv-1,1)),
                                dim = 1
                                )
                self.dataset = torch.cat((self.dataset,new))
        self.data_length = len(self.dataset)
        self.dataset = self.dataset.reshape(self.data_length,4,1)
    def __getitem__(self, idx):
        return self.dataset[(idx+1)% self.data_length].float()
    def __len__(self):
        return self.num




class Data_mu_p_train(Dataset):

    def __init__(self, file_name,t_final,device,mult = 1.,every = 1,t_step=0.01,n_trajs=30):
        self.t_step = t_step
        ds = h5py.File(file_name,'r')
        self.t_final = t_final
        self.ds = h5py.File(file_name,'r')
        self.length = int(len(self.ds)/3)
        self.num = int(self.length)
        self.t_final = t_final
        self.device= device
        self.mult = mult
        self.every = every
        self.n_trajs = n_trajs
        self.z_len = int(self.t_final/self.t_step)
        self.x = np.array(self.ds["sim"]) 
        self.x = torch.from_numpy(np.mean(self.x,axis=0))
        
        self.data = np.array(self.ds["sim"]) 
         
    def __getitem__(self, idx):
            stop_time = torch.randint(len(self.x),[1,])
            #x = self.data[np.random.choice(self.data.shape[0], self.n_trajs, replace=False), :stop_time.item()]
            return torch.from_numpy(self.data[idx]).reshape(len(self.data[idx]),1).float() # torch.from_numpy(x).float()
    def __len__(self):
        return len(self.data) 

class Data_ctime(Dataset):
    def __init__(self, 
                    file_name,
                    device,
                    mult = 1.,
                    every = 10,
                    t_step=1,
                    n_trajs=100,
                    t_max = 100000, 
                    N=100,
                    ):

        t_max = int(t_max/N)
        self.ds = h5py.File(file_name,'r')
        print("t_max")
        print(t_max)
        n_trajs = n_trajs+10
        self.x = np.zeros(n_trajs*t_max).reshape( n_trajs,t_max  )
        self.times = np.arange(n_trajs*t_max).reshape( n_trajs,t_max  )
        for j in range(1,n_trajs):
                m = np.array(self.ds["mag_"+str(j)])[0::t_step]
                t = np.array(self.ds["T_"+str(j)])[0::t_step]
                self.times[j-1,:][:len(t)]=t[:len(m)]
                self.x[j-1,:][:len(m)]=m[:len(m)]
        n_trajs = n_trajs-10
        self.t_step = t_step
        self.x = torch.from_numpy(self.x.reshape(self.x.shape[0],self.x.shape[1],1)).float() 
        self.times = torch.from_numpy(self.times.reshape(self.x.shape[0],self.x.shape[1],1)).float() 
        #self.time = torch.tensor([t for t in np.arange(0,len(self.x) ,t_step)]) 
        #self.x = self.x[:len(self.time)].to(device)
        #self.time = self.time[0::every][:len(self.x)].to(device)
        self.num = self.x.shape[0]
        self.dataset = torch.tensor([],device=device)
        # data are organised in tuples [mag(time), mag(time + delta_t), time, time+delta_t]
        for i in range(1, n_trajs +1 ):
                m = self.x[i]
                t = self.times[i]
                dt = t[:-1]-t[1:]
                dt[dt==0]=1
                qv = torch.cumsum(( (m[:-1]-m[1:]) )**2,0)[0::every]
                dt = torch.ones_like(qv)*every*t_step
                m = m[0::every]
                len_qv = qv.shape[0]
                new = torch.cat(
                            (
                                qv[:-1].reshape(len_qv-1,1), # [X]_t at time t 
                                qv[1:].reshape(len_qv-1,1), # [X]_t+dt at time t + dt
                                m[:(len_qv-1)].reshape(len_qv-1,1), #X_t at time t
                                dt[:(len_qv-1)].reshape(len_qv-1,1)),
                                dim = 1
                                )
                self.dataset = torch.cat((self.dataset,new))
        self.data_length = len(self.dataset)
        self.dataset = self.dataset.reshape(self.data_length,4,1)
    def __getitem__(self, idx):
        return self.dataset[(idx+1)% self.data_length].float()
    def __len__(self):
        return self.num

class Data(Dataset):
    def __init__(self, file_name,t_final,device,mult = 1.,every = 1,t_step=0.01,n_trajs=100):
        self.ds = h5py.File(file_name,'r')
        self.x = torch.from_numpy(np.array(self.ds["sim"])) 
        self.t_step = t_step
        self.x = self.x.reshape(self.x.shape[0],self.x.shape[1],1).float() 
        self.time = torch.tensor([t for t in np.arange(0,len(self.x) ,t_step)]) 
        self.x = self.x[:len(self.time)].to(device)
        self.time = self.time[0::every][:len(self.x)].to(device)
        self.num = self.x.shape[0]
        self.dataset = torch.tensor([],device=device)
        # data are organised in tuples [mag(time), mag(time + delta_t), time, time+delta_t]
        for i in range(1, n_trajs +1 ):
                m = self.x[i]
                qv = torch.cumsum((m[:-1]-m[1:])**2,0)[0::every]
                dt = torch.ones_like(qv)*every*t_step
                m = m[0::every]
                len_qv = qv.shape[0]
                new = torch.cat(
                            (
                                qv[:-1].reshape(len_qv-1,1), # [X]_t time t 
                                qv[1:].reshape(len_qv-1,1), # [X]_t+dt time t + dt
                                m[:(len_qv-1)].reshape(len_qv-1,1), #X_t time t
                                dt[:(len_qv-1)].reshape(len_qv-1,1)),
                                dim = 1
                                )
                self.dataset = torch.cat((self.dataset,new))
        self.data_length = len(self.dataset)
        self.dataset = self.dataset.reshape(self.data_length,4,1)
    def __getitem__(self, idx):
        return self.dataset[(idx+1)% self.data_length].float()
    def __len__(self):
        return self.num







def cp_loader(t0,t1,t_final,file_name,device,mult=1):
        ds = h5py.File(file_name,'r')
        z = torch.zeros(int(len(ds)/2),t_final).to(device)
        ts = torch.linspace(t0, t1, steps=t_final, device=device)
        for k in range(int(len(ds)/2)):
                z[k,:len(ds["mag_"+str(k+1)])] = torch.tensor(ds["mag_"+str(k+1)])
        return mult*z.reshape(int(len(ds)/2),t_final,1), ts


def cp_loader_t(t0,t1,t_final,file_name,device,mult=1.):
        ds = h5py.File(file_name,'r')
        z = torch.zeros(int(len(ds)/2),t_final).to(device)
        ts = torch.linspace(t0, t1, steps=t_final, device=device)
        for k in range(int(len(ds)/2)):
                if len(ds["mag_"+str(k+1)]) < t_final:
                        z[k,:len(ds["mag_"+str(k+1)])] = torch.tensor(ds["mag_"+str(k+1)])
                else:
                        z[k,:] = torch.tensor(ds["mag_"+str(k+1)])[:t_final]
        return mult*z.reshape(int(len(ds)/2),t_final,1), ts


def c_time_cp_loader_t(t0,t1,t_final,t_step,file_name,device,mult=1.,every=10):
        #for this datasets, one has time, configurations and magnetization
        # so one must divide the length of the dataset by three
        y_len = int(t_final/self.t_step)
        ds = h5py.File(file_name,'r')
        n_steps = int(t_final/(t_step*every))
        z = torch.zeros(int(len(ds)/3),n_steps).to(device)
        ts = torch.linspace(t0, t1, steps=n_steps, device=device)
        for k in range(int(len(ds)/3)):
                data_point = self.ds["mag_"+str(k + 1)]
                data_time = self.ds["T_"+str(k + 1)]
                l_t = len(data_time)
                l_p = len(data_point)
                if  l_p > 1:
                        set_interp = interp1d(data_time[:l_p], data_point[:l_t], kind='linear')
                        time = [t for t in np.arange(0,data_time[:l_p][-1] ,t_step)] 
                        interp_data_point = set_interp(time)
                        y = torch.zeros(self.z_len,device = self.device)
                        v = torch.tensor(interp_data_point,device = device)
                        y[:len(v)] = v[:z_len]
                        z[k,:] =  y[0::self.every].float()
        return mult*z.reshape(int(len(ds)/3),n_steps,1), ts





if __name__ == "__main__":
        plt.rcParams.update({
                    "text.usetex": True,
                        })
        with torch.no_grad():
                device = "cpu"
                t0 = 0.
                t1 = 2.
                t_final = 1000
                every = 10
                t_step = 0.01
                n_steps = int(t_final/(every*t_step))
                sample_size = 1
                ts = torch.linspace(t0,t1,n_steps)
                kappas = [0.56]
                xs_avgs = []
                for kappa in kappas:  #np.arange(1.1,1.9,0.05):
                        file_name  = "c_time_con_proc/t_final1000_rand0_ctime_cp_kappa_%.3f_gamma_0.1.h5" % (kappa)
                        data = Data_p_train(file_name = file_name, t_final = t_final, device = device,every = every) 
                        xs = torch.empty(sample_size,n_steps)
                        for i in range(sample_size):
                                xs[i] = data[i].flatten()
                                plt.plot(ts,xs[i])
                        plt.savefig("./images/sample_data.pdf")
                        exit()
                        xs_mean = xs.mean(dim = 0).flatten()
                        xs_avgs.append(xs_mean[-1])
                        kappas.append(kappa)
                        log_mean = torch.log(xs_mean)
                        plt.plot(torch.log(ts)[10:], -log_mean[10]+log_mean[10:])
                        plt.plot( torch.log(ts)[10:], 0.16*torch.log(ts)[10]-0.16*torch.log(ts)[10:]  )
                        plt.legend(["data","accepted behaviour"])
                        plt.xlabel(r"$\log(t)$")
                        plt.ylabel(r"$\log(\rho)$")
                        plt.savefig("./images/log_log_ctime_cp_kappa_%.3f.pdf"%kappa)
                        plt.close()
                print(kappas)
                print(xs_avgs)
                plt.scatter(kappas, xs_avgs,marker='^',)
                plt.xlabel(r'$\kappa$')
                plt.ylabel(r'$\rho(t_\textrm{max})$')
                plt.savefig("./images/cp_critical.pdf")
                plt.close()
                exit()
                name_file =  "/mnt/qb/datasets/STAGING/lesanovsky/fcarnazza/con_proc/cp_kappa_0.1_gamma_0.1.h5"
                file_name =  "c_time_con_proc/rand0_ctime_cp_kappa_0.55_gamma_0.1.h5"  
                device = "cpu"
                data = Data_p_train(file_name = file_name, t_final = 100,device = device) 
                for i in range(1000):
                        print(len((data[i])))
                #for kappa in np.arange(0.75,0.95,0.05):
                #        file_name = "/mnt/qb/datasets/STAGING/lesanovsky/fcarnazza/rand0_con_proc/rand0_cp_kappa_%.2f_gamma_0.1.h5" % kappa
                #        xs, ts = cp_loader(t0,t1,t_final,file_name,device)
                #        xs_avgs.append(xs.mean(dim = 0).flatten()[-1])
                #        #plt.scatter(kappa, xs.mean(dim = 0).flatten()[-1],marker='^',)
                #        kappas.append(kappa)
