import h5py
import argparse
import numpy as np 


parser = argparse.ArgumentParser('temperature list')
parser.add_argument('--T_min', type=float, default=2.)
parser.add_argument('--T_max', type=float, default=4.)
parser.add_argument('--N', type=int, default=60)

args = parser.parse_args()
def create_T_list(
                T_list
                ):
        """
        a function which creates an h5 file containing
        the tempratures in T_list
        """
        T_list_f = np.array([float(T) for T in T_list])
        name_file = "./dataset/kappas_from_%.4f_to_%.4f_N_%d.h5"%(T_list_f.min(),T_list_f.max(),len(T_list))
        hf = h5py.File(name_file, 'w')
        hf.create_dataset("kappas", data=T_list)
        hf.close()
        
        return name_file

if __name__ == "__main__":
        T_list = [
                        "2.0",
                        "2.0555555555555554",
                        "2.111111111111111",
                        "2.1666666666666665",
                        "2.2222222222222223",
                        "2.2777777777777777",
                        "2.3333333333333335",
                        "2.388888888888889",
                        "2.4444444444444446",
                        "2.5",
                        "3.0",
                        "3.0332222222222223",
                        "3.0664444444444445",
                        "3.099666666666667",
                        "3.132888888888889",
                        "3.166111111111111",
                        "3.199333333333333",
                        "3.2325555555555554",
                        "3.29",
                        "3.3144827586206898",
                        "3.3389655172413795",
                        "3.363448275862069",
                        "3.3879310344827585",
                        "3.412413793103448",
                        "3.436896551724138",
                        "3.4613793103448276",
                        "3.4858620689655173",
                        "3.510344827586207",
                        "3.5348275862068967",
                        "3.559310344827586",
                        "3.5837931034482757",
                        "3.6082758620689654",
                        "3.632758620689655",
                        "3.657241379310345",
                        "3.6817241379310346",
                        "3.7062068965517243",
                        "3.730689655172414",
                        "3.7551724137931033",
                        "3.779655172413793",
                        "3.8041379310344827",
                        "3.8286206896551724",
                        "3.853103448275862",
                        "3.8775862068965514",
                        "3.902068965517241",
                        "3.926551724137931",
                        "4.0",
                        "4.0344827586206895",
                        "4.068965517241379",
                        "4.103448275862069",
                        "4.137931034482759",
                        "4.172413793103448",
                        "4.206896551724138",
                        "4.241379310344827",
                        "4.275862068965517",
                        "4.310344827586207",
                        "4.344827586206897",
                        "4.379310344827586",
                        "4.413793103448276",
                        "4.448275862068965",
                        "4.482758620689655",
                        "4.517241379310345",
                        "4.551724137931035",
                        "4.586206896551724",
        ]
        T_list = np.linspace(args.T_min,args.T_max,args.N)
        name_file = create_T_list(T_list)
        T_list_f = np.array([float(T) for T in T_list])
        name_file = "./dataset/kappas_from_%.4f_to_%.4f_N_%d.h5"%(T_list_f.min(),T_list_f.max(),len(T_list))
        Ts = h5py.File(name_file,'r')
        print("test_di_"+str(np.array(Ts["kappas"])[0].decode("utf-8"))+"_prova")
