using HDF5
using ArgParse
using Random
using Interpolations
using ProgressBars
function interp1(xpt, ypt, x; method="linear", extrapvalue=nothing)

    if extrapvalue == nothing
        y = zeros(size(x))
        idx = trues(size(x))
    else
        y = extrapvalue*ones(size(x))
        idx = (x .>= xpt[1]) .& (x .<= xpt[end])
    end
    
    if method == "linear"
        intf = interpolate((xpt,), ypt, Gridded(Linear()))
        y[idx] = intf[x[idx]]

    elseif method == "cubic"
        itp = interpolate(ypt, BSpline(Cubic(Natural())), OnGrid())
        intf = scale(itp, xpt)
        y[idx] = [intf[xi] for xi in x[idx]]
    end
    
    return y
end

function contact_process(N::Integer,kappa::AbstractFloat,gamma::AbstractFloat,t_final::Integer)
        rate = Vector{Float64}(undef,N)
        #initial state
        initial = ones(Int8,1,N)
        #initial = floor.(Int8,(sign.(rand(1,N)-rand()*ones(1,N)) + ones(1,N))/2)
        #initial[26] = 1 
        #counting index and time vector
        global x = 1
        global T = Vector{Float64}()
        global state = Vector{Int8}()
        global mag = Vector{Float64}()
        append!(T,0)
        global time=T[x]
        append!(state,initial)
        append!(mag,sum(initial)/N)
        while time < t_final
                global x = x+1
                #calculate rates

                if sum(initial)<1
                        append!(T,t_final)
                        append!(state,initial)
                        break
                end
                for k =1:1:N
                        #final= copy(initial)
                        #final[k] = 1-final[k]
                        # periodic boundaries
                        left= k-1
                        if left == 0
                                left= N
                        end

                        right = k+1
                        if right == N+1
                                right=1
                        end
                        # flip rate
                        global rate[k]=gamma*(initial[k])+kappa*0.5*(initial[left]+initial[right])*(1-initial[k])

                end
                S=cumsum(rate,dims=1)/sum(rate)
                jump_ind = searchsortedlast(S, rand())+1#sum(rand>S)+1
                initial[jump_ind]=1-initial[jump_ind]
                append!(state, initial)
                append!(mag,sum(initial)/N)
                #advance the clock
                append!(T,T[x-1]-log(rand())/sum(rate))
                global time = T[x]

        end
        return state, mag, T
end

function contact_process_resume(
                N::Integer,
                kappa::AbstractFloat,
                gamma::AbstractFloat,
                t_final::Integer,
                num_traj::Integer,
                file_name::AbstractString,

        )
        rate = Vector{Float64}(undef,N)
        #initial = floor.(Int8,(sign.(rand(1,N)-rand()*ones(1,N)) + ones(1,N))/2)
        file = h5open(file_name,"cw")
        initial = last(file["process_"*string(num_traj)]) 
        #counting index and time vector
        global x = size(file["T_"*string(num_traj)])[1] 
        global T = h5read(file_name,"T_"*string(num_traj))
        global state = h5read(file_name,"process_"*string(num_traj))
        global mag = h5read(file_name,"mag_"*string(num_traj))
        global time=T[x]
        append!(state,initial)
        append!(mag,sum(initial)/N)
        while time < t_final
                global x = x+1
                #calculate rates

                if sum(initial)<1
                        append!(T,t_final)
                        append!(state,initial)
                        break
                end
                for k =1:1:N
                        #final= copy(initial)
                        #final[k] = 1-final[k]
                        # periodic boundaries
                        left= k-1
                        if left == 0
                                left= N
                        end

                        right = k+1
                        if right == N+1
                                right=1
                        end
                        # flip rate
                        global rate[k]=gamma*(initial[k])+kappa*0.5*(initial[left]+initial[right])*(1-initial[k])

                end
                S=cumsum(rate,dims=1)/sum(rate)
                jump_ind = sum(rand()>S)+1#searchsortedlast(S, rand())+1
                initial[jump_ind]=1-initial[jump_ind]
                append!(state, initial)
                append!(mag,sum(initial)/N)
                append!(T,T[x-1]-log(rand())/sum(rate))
                global time = T[x]

        end
        len = min(length(T),length(m))
        return state, mag[1:len], T[1:len]
end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--kappa"
            help = "infection rate"
            arg_type = Float64
            default = 2. 
        "--gamma"
            help = "recovery rate"
            arg_type = Float64 
            default = 1. 
        "--Num_traj"
            help = "Number of trajectories"
            arg_type = Int 
            default = 1000 
        "--N"
            help = "system size"
            arg_type = Int 
            default = 100
        "--t_final"
            help = "iteration"
            arg_type = Int 
            default = 100
    end 

    return parse_args(s)
end

function main()
        parsed_args = parse_commandline()
        # Parameters
        N = parsed_args["N"] #system size
        kappa = parsed_args["kappa"] # branching rate
        gamma = parsed_args["gamma"] # decay rate
        t_final = parsed_args["t_final"] # max time
        Num_traj = parsed_args["Num_traj"]
        #file_name = "c_time_cp_kappa_"*string(kappa)*"_gamma_"*string(gamma)*".h5"
        file_name  = "./dataset/N_"*string(N)*"_t_final"*string(t_final)*"_up0_ctime_cp_kappa_"*string(kappa)*"_gamma_"*string(gamma)*".h5" 
                for _m in ProgressBar(1:Num_traj)
                        h5open(file_name, "cw") do file
                                name_pro = "process_"*string(_m)
                                name_mag = "mag_"*string(_m) 
                                name_T = "T_"*string(_m) 
                                if haskey(file, name_pro)
                                        process,mag,T = contact_process_resume(N,kappa,gamma,t_final,_m,file_name)
                                        delete_object(file,name_mag)
                                        delete_object(file,name_T)
                                        write(file,name_mag ,mag)  
                                        write(file,name_T ,T)  
                                else 
                                        process,mag,T = contact_process(N,kappa,gamma,t_final)
                                        write(file,name_mag ,mag)  
                                        write(file,name_T ,T)  
                                end
                                close(file)
                        end
                end
end
main()
