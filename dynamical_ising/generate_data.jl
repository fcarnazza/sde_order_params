using HDF5
using ArgParse
using OffsetArrays
using Statistics
using ProgressBars
using Printf
module Ising2D
using Random
#       "simulating the Ising model"
       function mcmove(N::Integer,beta::AbstractFloat,config::AbstractArray)             
               #execute Monte Carlo Move
               for i = 1:N 
                       for j = 1:N 
                              a = rand(0:(N-1))
                              b = rand(0:(N-1))
                              s = config[a,b]
                              nb = config[(a+1)%N,b] + config[a,(b+1)%N] + config[abs(a-1)%N,b] + config[a,abs(b-1)%N]
                              cost = 2*s*nb
                              if cost < 0
                                      s *= -1 
                              elseif rand() < exp(-cost*beta)
                                      s *= -1
                              end
                              config[a,b] = s
                       end
               end
               return config
       end
end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--T"
            help = "Temperature"
            arg_type = Float64
            default = 2. 
        "--Size_system"
            help = "System size"
            arg_type = Int
            default = 64 
        "--Num_traj"
            help = "Number of trajectories"
            arg_type = Int
            default = 100
        "--t_max"
            help = "tmax"
            arg_type = Int
            default = 1000
    end

    return parse_args(s)
end

############
### Main ###
############

function main()
        """
        This function creates num_traj trajectories for a Metropolis Hestings dynamics for the Ising model
        for a K x K 2d dimensional system at temperature T_c frac_T up to t_max
        """
        parsed_args = parse_commandline()
        K = parsed_args["Size_system"]
        num_traj = parsed_args["Num_traj"] 
        T_c = 2/log(1+sqrt(2))
        T_min = 2.269
        T_max = 3
        t_max = parsed_args["t_max"]
        T = parsed_args["T"]
        #global config = OffsetArray( ones(Int8,K,K),0:(K-1),0:(K-1))
        global config = OffsetArray(2*rand(Bool,K,K) - ones(Int8,K,K),0:(K-1),0:(K-1))
        config0 = config
        global config_list = Vector{Float64}()
        global times = Vector{Int64}()
        global m = Vector{Float64}()
        ising = Ising2D
        beta = 1/T 
        file_name = "./dataset/N_"*string(K)*"_T_"*string(round(T,digits=4))*"_num_traj_"*string(num_traj)*"t_max_"*string(t_max)*".h5"

        for _m in ProgressBar(1:num_traj)
                for k = 1:t_max
                       global config = ising.mcmove(K,beta, config)
                       append!(m, mean(config))
                end
                mag_traj = "m1_"*string(_m)
                h5open(file_name, "cw") do file
                        write(file, mag_traj,m)  # alternatively, say "@write file A"
                        close(file)
                end

                #global config = OffsetArray( ones(Int8,K,K),0:(K-1),0:(K-1))
                global config = OffsetArray(2*rand(Bool,K,K) - ones(Int8,K,K),0:(K-1),0:(K-1))
                config0 = config
                global m = Vector{Float64}()
        end

end


main()



