module Env #Defines types, attributes and useful functions for environment

module EnvParams

const SLOT_DUR = 1.5 #In minutes
const RSU_R = 0.5 #In Kilometers
const N_RSU = 5 #number of RSUs
const B̄ = 50 #Max number of BRBs
const C̄ = 100 #MAX number of CRBs

end

module ServiceType #Defines the Service type and the service set Κ along with its attributes

export Service, SERVICE_SET, N_SERVICE

struct Service
	service_indx::Int
	data_size::Float64 #In Mb
	crb_needed::Float64 #In Gcycles i.e x10⁹
	thresh::Float64
	max_thresh::Float64
end

begin
	const service1 = Service(
			1,
			4,
			.5,
			0.35,
			0.5
		)

		const service2 = Service(
			2,
			6,
			0.6,
			0.35,
			0.5
		)
		const service3 = Service(
			3,
			8,
			0.8,
			0.35,
			0.5
		)
	const SERVICE_SET = (service1, service2, service3)
	const N_SERVICE = length(SERVICE_SET)
end

end 

module MobileUser #Defines the User type and its utility functions. 

export User, action_mapper, snapshot, reward 
include("constants_and_utils.jl")
using .UtilsAndConstants:transmit_rate
using ..ServiceType 
using ..EnvParams:N_RSU, B̄, C̄ 
using StatsBase

mutable struct User
	rsu_id::Int
	service::Service
	BRB::Int
	CRB::Int
	mig_service::Bool
end

function action_mapper(x) #Map op from neural network to a workable form.
	@assert length(x) == (N_SERVICE*N_RSU + N_SERVICE + 1)
    action = []
    for rsu_indx in 1:N_RSU
        rsu_allocs = x[(rsu_indx-1)*N_SERVICE + 1: (rsu_indx-1)*N_SERVICE + N_SERVICE]
        #Normalize the allocations to sum to 1
        rsu_allocs .= rsu_allocs ./ sum(rsu_allocs)
        push!(action,rsu_allocs)
    end
    crb_allocs = x[N_RSU*N_SERVICE + 1 : N_RSU*N_SERVICE + N_SERVICE]
    #Normalize the CRB allocations to sum to 1
    crb_allocs .= crb_allocs ./ sum(crb_allocs)
    push!(action,crb_allocs)
    mig_fraction = x[end]
 push!(action, mig_fraction)
end	

#Define the function which generates a snapshot of the environment given the state and action.
function snapshot(state, action)
	users = []
    users_per_service = zeros(Int,N_SERVICE)
    #Allocate B/W
	for rsu_indx in 1:N_RSU
		#Convert the fractional allocations to integers
		rsu_allocs = (action[rsu_indx] .* B̄) .|> x -> floor(Int,x)
		for (app_indx,app) in enumerate(SERVICE_SET)
			veh_cnt = state[(rsu_indx-1)*N_SERVICE + app_indx] #Get count of vehicles
			users_per_service[app_indx] += veh_cnt
			for _ in 1:veh_cnt
				BRB = rsu_allocs[app_indx] ÷ veh_cnt
				push!(users, User(rsu_indx,app,BRB,0,false))
			end
		end
	end
	#Sample some users for migration
	#And give them their CRB which is allocated equally
	W = state[N_RSU*N_SERVICE + 1]
	mig_cnt = floor(Int, W * action[end])
	#mig_descision = action[end] > 0.5 ? true : false
	mig_users = []
	if mig_cnt>0 && W>0
		Y = state[end] #Allocs of next server..
		mig_candidates = filter(users) do user
			user.rsu_id == 5 || user.rsu_id == 4
		end
		n_mig_candidates = length(mig_candidates)
		#1)Random sampling for migration....
		#=
        mig_users = StatsBase.sample(mig_candidates, mig_cnt, replace = false)
		=#
        #2)Sample users based on service "weight" in descending order.
		#=
		mig_users = sort(mig_candidates,
			by = user -> user.service.service_indx,
			rev=true)[1:mig_cnt]
		=#
		#3)Sample users based on service "weight" in ascending order.
		mig_users = sort(mig_candidates, 
			by = user -> user.service.service_indx, 
			rev=false)[1:mig_cnt]
		
        map(mig_users) do user
			#Due to migrations we have now freed up some CRBs
			indx = user.service.service_indx
			users_per_service[indx] -= 1 
			user.CRB = (Y ÷ length(mig_users)) #Integer division because cannot allocate fraction of BRB/CRB
			user.mig_service = true
		end
	end
	#Allocate the CRBs to the remaining vehicles
	crb_allocs = (action[N_RSU + 1] .* C̄) .|> x->floor(Int,x)
	for (app_indx,app) in enumerate(SERVICE_SET)
		rem_users = filter(users) do user
			user.mig_service == false && user.service == app
		end
		map(rem_users) do user
			V = users_per_service[app_indx]
			user.CRB = crb_allocs[app_indx] ÷ V
		end
	end
	return users
end

#Utility function Uᵥ,ₜ
function utility(user::User)
	#reward_mul = [0.5,0.75,1]
	service = user.service
	bv = service.data_size * 1e6
	Cv = service.crb_needed
	τ = service.thresh
	τₘ = service.max_thresh
	R = transmit_rate(200,user.BRB)
	if user.BRB == 0 || user.CRB == 0  # utility for failing to allocate resources
		u = 0
	else
		del = bv/R + (Cv/user.CRB)
		u = del < τ ? 1.0 : del > τₘ ? 0.0 : -(del - τₘ)/(τₘ - τ)
		u = u 
	end
	return u
end

#Reward function
function reward(state, action)
	users = snapshot(state, action)
	return users .|> utility |> sum
end


end

#Export useful stuff to calling scope of Env
export N_RSU

using .MobileUser
export User, action_mapper, snapshot, reward 

using .ServiceType
export Service, SERVICE_SET, N_SERVICE

end
