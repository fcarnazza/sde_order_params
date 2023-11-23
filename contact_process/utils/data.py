import os
import sys
import matplotlib.pyplot as plt
from cp_loader import Data as Data_sigma
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mu_ode_solve import Data

if __name__ == '__main__':
        kappa = 3.29
        dataset = "../../Autoencoder/HGN/nn_sde/c_time_con_proc/"
        file_name  = dataset+"N_100_perc_1.0_nsteps_1000_inf_rate_"+str(kappa)+"_rec_rate_1_surv_prob_False_.h5" 
        #data = Data(file_name)
        #for i in range(100):
        #        print(i,data.x[i].item())
        dataset = Data_sigma(file_name,
            t_final = 1000,
            device = "cpu",
            every = 10)
        qv = []
        dt_qv = []
        t = []
        pred = []
        x = []
        mult = 1.
        for i in range(0, 100):
            v = dataset[i]
            qv.append(v[0])
            t.append(v[3])
            dt_qv.append((v[1]-v[0])/v[3]/mult**2)
            x.append(v[2])
            print(((v[1]-v[0])/v[3]).item())
