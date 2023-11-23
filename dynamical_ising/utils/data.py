import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import h5py
import argparse
import time
import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser('ODE cp')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--kappa', type=str, default="3.29")
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=7000)
parser.add_argument('--test_freq', type=int, default=100)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--t_max', type=int, default=9700)
parser.add_argument('--lr', type=float, default=0.7e-4)
parser.add_argument('--pre_kappa', type=float, default=0.250)
parser.add_argument('--pre_trained', type=bool, default=False)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--n_trains', type=int,default=10)
parser.add_argument('--N', type=str,default="100")
args = parser.parse_args()

class Data(Dataset):
    def __init__(self, file_name,device= "cpu",every=1,t_max=1000,t_step=0.01):
        self.ds = h5py.File(file_name,'r')
        keys = self.ds.keys()
        self.x =    np.abs(   np.array([self.ds[key][:t_max] for key in keys]))
        self.x = torch.from_numpy(np.mean(self.x,axis=0))#[:t_max]
        self.t_step = t_step
        self.x = self.x.reshape(len(self.x),1).float() 
        self.time = torch.tensor([t for t in np.arange(0,len(self.x) ,t_step)]) 
        self.x = self.x[:len(self.time)] [0::every].to(device)
        self.time = self.time[0::every][:len(self.x)].to(device)
        self.batch_t = self.time[:args.batch_time].to(device)
        self.num = len(self.x) - args.batch_time
    def __getitem__(self,idx):
            s = torch.from_numpy(np.random.choice(np.arange(self.num, dtype=np.int64), args.batch_size, replace=False))
            batch_y0 = self.x[s]  # 
            batch_y = torch.stack([self.x[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
            return batch_y0.to(device), self.batch_t.to(device), batch_y.to(device)
    def __len__(self):
        return self.num

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if __name__ == '__main__':
        file_name =  "/mnt/qb/datasets/STAGING/lesanovsky/fcarnazza/dyn_ising/dataset/N_128_T_2.0991_num_traj_1000.h5"
        file_name =  "dataset/N_128_T_2.1416_num_traj_1000.h5"

        data = Data(file_name)
        keys = data.ds.keys()
        x =    np.abs(   np.array([data.ds[key][:50000] for key in keys]))
        print(x.shape)
#        x = torch.from_numpy(np.mean(x,axis=0))#[:t_max]
        print(x.shape)
        plt.plot( x[3])
        plt.savefig("./images/check.pdf")
        x = torch.from_numpy(np.mean(x,axis=0))#[:t_max]
        print(x.shape)
        plt.plot( x)
        plt.savefig("./images/check1.pdf")


