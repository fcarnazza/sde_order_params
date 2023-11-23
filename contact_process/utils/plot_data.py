import matplotlib.pyplot as plt
import h5py
import numpy as np
import os
def plot(file_name):
    ds = h5py.File(file_name,"r")
    m1 = np.array(ds["mag_1"])
    T1 = np.array(ds["T_1"])
    print(m1.shape)
    print(T1.shape)
    plt.plot(T1[:len(m1)],m1[:len(T1)])
    plt.savefig("c_time_con_proc/image.pdf")

def launch(h5_file,shell_script):
    """
    it will launch the bash script shell_script
    with all the entries of dataset "temps" for the h5 file
    """
    kappas = h5py.File(h5_file,'r')
    for k in np.array(kappas["kappas"]):
        os.system("sh %s %s"%(shell_script,k))
 
if __name__=="__main__":
    kappas =[3.3559322033898304]
    m_stat = np.zeros(len(kappas))
    for i in range(len(kappas)):
        k = kappas[i]
        file_name = "dataset/N_100_t_final100000_up0_ctime_cp_k_"+str(k)+"_gamma_1.0.h5"
        ds = h5py.File(file_name,"r")
        m = np.array([ np.array(ds["mag_"+str(j)])[-1] for j in range(1,10)  ])
        m_stat[i] = m.flatten().mean()
        t1 = np.array(ds["T_1"])
        print(k,m_stat[i])
        m1 = np.array(ds["mag_2"])
        m = m1[0::100]
        print(len(m1))
        qv= (np.cumsum( (m[:-1]-m[1:])**2 ))[0::20]
        plt.plot( -qv[:-1]+qv[1:]   )
    plt.savefig("images/crt1.pdf")
    plt.close()
    plt.plot(kappas,m_stat)
    plt.savefig("images/crt.pdf")

    exit() 

    hf = h5py.File('kappa_3.3559.h5', 'w')
    hf.create_dataset('kappas', data=kappas)
    hf.close()
    launch('kappa_3.3559.h5',"launch.sh")
    exit()
    #kappas = np.linspace(2,4,20)
