let
	using Pkg
	Pkg.activate(".")
end

using JSON, Plots, StatsBase, Flux, JLD2, CSV

include("constants_and_utils.jl")
include("environment.jl")
include("ddpg.jl")

using .UtilsAndConstants
using .Env
using .DDPG

#State and action sizes
begin
	const N_STATE = N_SERVICE * N_RSU + 2
	const N_ACTION = N_SERVICE*N_RSU + N_SERVICE + 1
end

STATE_ORDER = vcat(["($(i), $(j))" for i=1:5 for j=1:3],["rem_time_2","allocs_2"]) #Same as training.jl

begin
	test_agent = Agent(
		Chain(Dense(N_STATE => 100,tanh),Dense(100 => N_ACTION, σ)),
		Chain(Dense(N_STATE+N_ACTION => 100,tanh),Dense(100 => 1,relu)),
		Adam(0.01),
		Adam(0.01),
		ReplayBuffer(;mem_size = 1, state_size = N_STATE, action_size = N_ACTION),
		120,
		0.98,
		0.01,
		1
	)
	#Load the params for the networks...
	let
		str  = "3001"
		actor_state = JLD2.load("actor_params/actor_$(str).jld2", "model_state")
		critic_state = JLD2.load("critic_params/critic_$(str).jld2", "model_state")
		Flux.loadmodel!(test_agent.actor, actor_state)
		Flux.loadmodel!(test_agent.critic, critic_state)
	    #Target models not needed for evaluation, so no need to
        #initialize them with the model parameters. You need to reinit the
        #target models only if you wish to resume training.
    end
	test_agent.training_mode = false
end


#Load the test data set
begin
	noised_time_test = let
	json_str = open("path_to_time_series_file","r") do file
		read(file,String)
	end
	JSON.parse(json_str)
	end
end

#Evaluating the models performance based on various metrics:
#Refer to paper for all performance metrics
begin
	service_dels = zeros(length(SERVICE_SET)) #Mean service delay over all t and all episodes
	vehicle_counts = zeros(length(SERVICE_SET)) #Averaged counts of vehicles running service Sᵢ.
	allocs = zeros(length(SERVICE_SET)) #Averaged allocations
	non_allocs = zeros(length(SERVICE_SET))
	violations  = zeros(length(SERVICE_SET))
	rsu_allocs = [zeros(length(SERVICE_SET)) for _ in 1:N_RSU]
	migs = zero(Float64)
	N_episodes = 50
	for episode in 1:N_episodes #We have N_episode=50 number of test episodes
		Delays = [zeros(40) for _ in SERVICE_SET]
		Counts = [zeros(40) for _ in SERVICE_SET]
		Allocs = [zeros(40) for _ in SERVICE_SET]
		Non_allocs = [zeros(40) for _ in SERVICE_SET]
		Violations = [zeros(40) for _ in SERVICE_SET]
		Rsu_allocs = [[zeros(40) for _ in SERVICE_SET] for _ in 1:N_RSU]
		Migrations = zeros(40)
		Possible_migs = zeros(40)
		for t in 1:40 # 40 time slots long
			st = [noised_time_test[key][episode][t] for key in STATE_ORDER]
			s = Flux.normalize(st) .|> Float32
			a = choose_action(test_agent, s)
			users = snapshot(st,action_mapper(a)) #Obtain the snapshot of the users
			#Count number of migrations
			Migrations[t] = count(user->user.mig_service, users)
			for (app_indx,app) in enumerate(SERVICE_SET)
				service_users = filter(users) do user
					user.service == app
				end 

				Allocs[app_indx][t] = length(service_users) > 1 ? service_users[1].CRB * length(service_users) : 0.0
				
				Counts[app_indx][t] += length(service_users)

				delays = map(service_users) do user
					Rsu_allocs[user.rsu_id][app_indx][t] += user.BRB
					bv = app.data_size * 1e6
					Cv = app.crb_needed
					R = transmit_rate(200,user.BRB)
					if user.BRB == 0 || user.CRB == 0
						del = app.max_thresh
						Non_allocs[app_indx][t] += 1
						Violations[app_indx][t] += 1
					else
						del = bv/R + (Cv/user.CRB)
						del > app.max_thresh ? Violations[app_indx][t] += 1 : nothing
					end
					del
				end
				delays = length(delays) > 0 ? delays : [0.0]
				push!(Delays[app_indx], mean(delays)) #mean(delays) is the mean service delay at time t
			end
		end
		#Compute the mean quatity over all time slots t
		mean_delay = Delays .|> mean
		mean_counts = Counts .|> mean
		mean_allocs = Allocs .|> mean
		mean_non_allocs = Non_allocs .|> mean
		mean_violations = Violations .|> mean
		mean_migrations = mean(Migrations)
		mean_rsu_allocs = map(Rsu_allocs) do service_allocs
								map(service_allocs) do service_alloc
									mean(service_alloc)
								end
							end
		#add this to the episodic mean quantity
		service_dels .= service_dels .+ mean_delay
		vehicle_counts .= vehicle_counts .+ mean_counts
		allocs .= allocs .+ mean_allocs
		non_allocs .= non_allocs .+ mean_non_allocs
		violations .= violations .+ mean_violations
		global migs += mean_migrations
		rsu_allocs .= rsu_allocs .+ mean_rsu_allocs
	end
    #Ensemble average over the N_episodes episodes
	service_dels .= service_dels ./ N_episodes 
	vehicle_counts .= vehicle_counts ./ N_episodes
	allocs .= allocs ./ N_episodes
	non_allocs .= non_allocs ./ N_episodes
	violations .= violations ./ N_episodes
	migs /= N_episodes
	rsu_allocs .= rsu_allocs ./ N_episodes
	possible_migs = [mean(noised_time_test["rem_time_2"][ep]) for ep in 1:N_episodes] |> mean


	Info_dict = Dict()
	Info_dict["service_dels"] = service_dels
	Info_dict["vehicle_counts"] = vehicle_counts
	Info_dict["allocs"] = allocs
	Info_dict["non_allocs"] = non_allocs
	Info_dict["violations"] = violations
	Info_dict["migs"] = migs
	Info_dict["rsu_allocs"] = rsu_allocs
	Info_dict["possible_migs"] = possible_migs
	for (key,val) in Info_dict
		println("$(key) ----> $(val)")
	end
	#save("nw_init_test.jld2", Info_dict)
	CSV.write("nw_init.csv", Info_dict)
end
