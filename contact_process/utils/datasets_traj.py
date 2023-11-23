import torch
import matplotlib.pyplot as plt
import h5py
import argparse
from torch import optim
import torch
import numpy as np
from torch.utils.data import Dataset
import torch
from torch import nn
import tqdm
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from mu_diff import dataset 
class dataset_ctime(Dataset):
    def __init__(self, file_name,thres=0.05,num_trajs=100,t_max = 100000,N=100):
        self.ds = h5py.File(file_name,'r')
        self.trajs = np.zeros(num_trajs*t_max).reshape( num_trajs,t_max  )
        #self.times = np.zeros(num_trajs*t_max).reshape( num_trajs,t_max  )
        for j in range(1,num_trajs+1):
                m = np.array(self.ds["mag_"+str(j)])[:t_max]
                t = np.array(self.ds["T_"+str(j)])[:t_max]
                #self.times[j-1,:][:len(m)]=t[:len(m)]
                self.trajs[j-1,:][:len(m)]=m[:len(m)]
        self.thres = thres
        self.max = np.max(self.trajs)
        self.min = np.min(self.trajs)+4*self.thres
        self.num_trajs = num_trajs
        self.t_max = t_max
    def __getitem__(self,idx):
        x=self.trajs[:self.num_trajs,:self.t_max]
        #t = self.times[:self.num_trajs,:self.t_max]
        x0=np.random.rand()*(self.max-self.min) + self.min
        idx = np.argwhere( np.abs(x[:,:-1]-x0)<self.thres  )
        x0s = x[idx[:,0],idx[:,1]]
        x_dt = x[:,1:][idx[:,0],idx[:,1]]
        #dt = t[:,1:][idx[:,0],idx[:,1]] - t[idx[:,0],idx[:,1]]
        mu_x = torch.mean( torch.from_numpy(x_dt-x0s))
        #mu_x = torch.mean( torch.from_numpy(x_dt))-x0
        _x0s_ = torch.from_numpy(x0.reshape(1,))
        return _x0s_.reshape(1,).float(), mu_x.reshape(1,).float()
    def __len__(self):
        return self.trajs.shape[0]

class dataset_dtime(Dataset):
    def __init__(self, file_name,thres=0.05,num_trajs=1000,t_max = 1000,lin_dim=100):
        self.ds = h5py.File(file_name,'r')
        self.trajs = np.array(self.ds["sim"]) 
        self.thres = thres
        self.max = np.max(self.trajs)-2*self.thres
        self.min = np.min(self.trajs)+2*self.thres
        self.num_trajs = num_trajs
        self.t_max = t_max
    def __getitem__(self,idx):
        x=self.trajs[:self.num_trajs,:self.t_max]

        x0=np.random.rand()*(self.max-self.min) + self.min
        idx = np.argwhere( np.abs(x[:,:-1]-x0)<self.thres  )
        x0s = x[idx[:,0],idx[:,1]]
        x_dt = x[:,1:][idx[:,0],idx[:,1]]
        mu_x = torch.mean( torch.from_numpy(x_dt-x0s))
        #mu_x = torch.mean( torch.from_numpy(x_dt))-x0
        _x0s_ = torch.from_numpy(x0.reshape(1,))
        return _x0s_.reshape(1,).float(), mu_x.reshape(1,).float()
    def __len__(self):
        return self.trajs.shape[0]

parser = argparse.ArgumentParser('ising check traj')

parser.add_argument('--kappa', type=float, default=3.3559322033898304)
parser.add_argument('--N', type=int, default=100)
args = parser.parse_args()

if __name__ == "__main__":
    data_dir = "./dataset/" 
    #file_name  = data_dir + "N_"+str(args.N)+"_perc_1.0_nsteps_10000_inf_rate_"+str(args.kappa)+"_rec_rate_1_surv_prob_False_.h5" 
    #file_name = "dataset/N_"+str(args.N)+"_t_final200_up0_ctime_cp_k_"+str(args.kappa)+"_gamma_1.0.h5"
    ile_name = "./dataset/N_"+str(args.N)+"_t_final100000_up0_ctime_cp_kappa_"+str(args.kappa)+"_gamma_1.0.h5"
    data = dataset(file_name)
    print(data.trajs.shape)
    [plt.plot(data.trajs[i]) for i in range(10)]
    plt.savefig("images/check_traj_N_%d_kappa_%4f.pdf"%(args.N,float(args.kappa)))
    print("trajectory plotted")
