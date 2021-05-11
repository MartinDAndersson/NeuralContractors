using IntervalArithmetic, IntervalConstraintProgramming, ModelingToolkit, Distributions,Random,LinearAlgebra, JLD2, FileIO
# export create_constraints
# export create_separator
#export model_truth



function set_up(g,parameter_space,flex)
    #uni = LinRange(0,2*pi,100)
    dims = length(parameter_space)
    #x = [[cos(x),sin(x)] for x in uni]
    index = collect(1:dims)
    # θ = Dict("w$i"=>rand(Uniform(-lims,lims)) for i in index)
    θ = rand(parameter_space)
    function model_truth(x)
        sum = 0
        for i in 1:min(flex,length(θ)÷2)
            sum += (-1)^(i+1)*g(x[1]*θ[1*i] + x[2]*θ[2*i])
        end
        return sum
    end
    inputs = rand(Uniform(-5,5),(2,100))
    inputs = [inputs[:,i]/norm(inputs[:,i]) for i in 1:size(inputs,2)]
    y = [model_truth(x) for x in inputs]
    max_model = maximum(abs,y)
    y = 1/max_model*y
    data = [(inputs[i],y[i]) for i in 1:length(y)]
    # δ = 10.
    # noise = -δ..δ
    # W = [@interval(θ["w$i"])+noise for i in 1:dim]
    # W = reduce(×,W)
    W = 1/max_model*parameter_space
    return data,θ,W,model_truth
end


function create_separator(constraints,x)
    C = constraints["c1"]
    for index in 2:length(x)
        C = C ∩ constraints["c$index"]
    end
    return C
end

function trained_model(θ,g=x->max(0,x))
    #return x-> g(θ[1]*x[1]+θ[2]*x[2]) - g(θ[3]*x[1]+θ[4]*x[2])
    return x-> g(θ[1]*x[1]+θ[2]*x[2]) - g(θ[3]*x[1]+θ[4]*x[2])
end

function test_function(x,model_truth,model_trained)
    for x_0 in x
        println(model_truth(x_0)-model_trained(x_0))
    end
end



function create_constraints(data,flex,act_funct,dims)
    g = act_funct
    node(w...;x) = g(w[1]*x[1]+w[2]*x[2])
    Loss(x) = x^2
    function net(W;x)
        dims = length(W)
        sum = 0
        dims = length(W)
        for i in 1:min(flex,dims÷2)
            sum += (-1)^(i+1)*node(W[2*(i-1)+1:2*(i-1)+2]...,x=x)
        end
        return sum
    end
    
    function cost_function(W;data)
        sum = 0
        for point in data
            sum += Loss(net(W;x=point[1])-point[2])
        end
        sum /= length(data)
    end
    # if act_funct == "Sigmoid"
    #     @function g(x) = 1(1+exp(-x))
    # else
    #     @function g(x) = max(0,x)
    # end
    # @function g(x) = max(0,x)
    W = @variables w1 w2 w3 w4 w5 w6
    #c(W...;data) = cost_function(W;data) < 0.001
    #sep = Separator(W...,c(W;data=data))
    d(W...) = cost_function(W;data=data) < 1.e-10
    sep = Separator(W[1:dims],d)
	return sep
end


function automate_S1(inputs, parameter_space,
    num_samples, accuracy,flex, act_funct)
    dim = length(parameter_space)
    #g = act_funct
    local paving
    for i in 1:num_samples
        println("run = ",i)
        true_W = rand(parameter_space)
        paving, C, W, θ, X, Y, model_truth = main(inputs,parameter_space,act_funct,accuracy,flex,dim)
        #zero_set = plot_with_makie(paving)
        #diffusion_plot = diffusion_map(matrix_mids,t,dim)
        boundary = paving.boundary
        inner = paving.inner
        #Makie.save("./plots/Zero_set_run=$(i).jpg",zero_set)
        #Makie.save("./plots/Diffusion_run=$(i).jpg",diffusion_plot)
        println("what")
        @save "./data/paving_d=$(dim)_a=$(act_funct)_flex=$(flex)_acc=$(accuracy)_run=$(i).jld2" inner boundary θ X Y act_funct flex dim W
    end
    return paving
end


function main(parameter_space,act_funct,accuracy,flex,dims)
    # if act_funct == "ReLU"
    #     g(x) = max(0,x)
    # else
    #     g(x) = 1/(1+exp(-x)) # sin(x), cos(x), tanh(x), x^2, x^3, 1/(1+exp(-x))
    # end
    data,θ,W,model_truth = set_up(act_funct,parameter_space,flex)
    #data = [(X[i],Y[i]) for i in 1:length(Y)]
    sep = create_constraints(data,flex,act_funct,dims)
    #C = create_separator(constraints,X)
    # # C = create_sep_alt(g,x,y)
    paving = pave(sep, W, accuracy)
    # plt = Plots.plot(paving.inner, aspect_ratio=:equal, legend=:false)
    # Plots.plot!(paving.boundary, aspect_ratio=:equal, linewidth=0, label="", color="gray")
    θ_test = mid(rand(paving.boundary))
    #model_trained = trained_model(g,θ_test)
    #plt = plot_with_makie(paving)
    # display(plt)
    # test_function(x,model_truth,model_trained)
    # # Plots.plot(model_truth,label = "true function")
    # # display(Plots.plot!(model_trained,color="red",label="contracted function"))
    return paving, C, W, θ, X, Y, model_truth# C,x,y,W, model_truth,g, paving, constraints, model_trained
end



