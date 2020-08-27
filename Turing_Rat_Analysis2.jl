"""
Turing analysis of frozen_noise stimon rats
Zach Yazdani
Code adapted from Bayesian Logistic Regression tutorial: https://turing.ml/dev/tutorials/2-logisticregression/ tutorial

The code uses click difference data from the Poissont clicks task to fit 3 Markov Chain models.
MODELS
Theoretically models 1) and 2) should be exactly the same but they are not.
1) bd model (element wise) **THIS MODEL IS PERFORMING VERY POORLY**
Fits a model by weighting the click difference (#R-#L clicks) for each of 10 time bins.
2) bd model (using a Matrix)
Uses a multivariate distribution of w's to do the same thing as 1)
3) LVR model **THIS MODEL IS PERFORMING VERY POORLY**
Also a time bin based click approach but clicks from the left and right are weighted seperately
Each of 10 time bins have 2 weights for a total of 20 weights
PLOTS
- ROC scores are calculated
- 1D Psychometric
- 2D Psychometric
"""

"""
Action items left to complete for Stage 1
-Extend model to make it higherarchical
-Might also want to plot confidence interval

"""
"""
cfg
Establishes import and export paths
"""
cfg = (
## io options
TITLE           = "Turing_Rat_Analysis2",
PROGRAM_NAME    = "Turing_Rat_Analysis2.jl",
IMPORTPATH_DATA = "data/regrMats_allrats_frozen_noise_500msLim_50msBin_0msOverlap.jld2",
#IMPORTPATH_DATA = "data/regrMats_allrats_chuckrats_update_500msLim_50msBin_0msOverlap.jld2",
EXPORTPATH_DATA = "data/",
SAVE_DATA       = true,

EXPORTPATH_FIGS = "figs/",
SAVE_FIGS       = false,

)

cfg2 = (
## io options
TITLE           = "Turing_Rat_Analysis2",
PROGRAM_NAME    = "Turing_Rat_Analysis2.jl",
IMPORTPATH_DATA = "Turing_Rats_figs&chains/Markov Chains/",
#IMPORTPATH_DATA = "data/regrMats_allrats_chuckrats_update_500msLim_50msBin_0msOverlap.jld2",
EXPORTPATH_DATA = "Turing_Rats_figs&chains/",
SAVE_DATA       = true,

EXPORTPATH_FIGS = "figs/",
SAVE_FIGS       = false,
)
"""
Imports
Imports necessary packages
"""
# Import Turing and Distributions.
using Turing, Distributions

# Import RDatasets.
using RDatasets

# Import MCMCChains, Plots, and StatsPlots for visualizations and diagnostics.
using MCMCChains, Plots, StatsPlots

# We need a logistic function, which is provided by StatsFuns.
using StatsFuns: logistic

# Functionality for splitting and normalizing the data
using MLDataUtils: shuffleobs, stratifiedobs, rescale!

# Set a seed for reproducibility.
using Random
Random.seed!(0);

using MAT
using DataFrames
using JLD2, FileIO
using Dates
using LinearAlgebra
#using GLM

using Printf
#using PyPlot
using PyCall
using MLBase
using Statistics
using DataFrames
using Conda

sklmetrics = pyimport("sklearn.metrics")
#include("utils.jl")

# Turn off progress monitor.
#Turing.turnprogress(false)

# Function to be used to split the data into training and test sets ß
function split_data(df, target; at = 0.70)
    shuffled = shuffleobs(df)
    trainset, testset = stratifiedobs(row -> row[target], shuffled, p = at)
end


#The column that we want to predict (gr is whether the rat went right or not)
target = :gr


#Imports data
cd("/Users/zyazdani/Desktop")
data = load(cfg.IMPORTPATH_DATA)["regrMats"]
# Imports previously made chains
#LVR
chains_LVR_list = load((cfg2.IMPORTPATH_DATA) * "chains_allrats_LVR_stimon2Bayesian Logistic Regression LVR.jld2")["chains_all_rats_LVR"]
#Element
chains_element_list = load((cfg2.IMPORTPATH_DATA) * "chains_allrats_Element_stimon2Bayesian Logistic Regression Element.jld2")["chains_all_rats_Element"]
#Matrix
chains_Matrix_list = load((cfg2.IMPORTPATH_DATA) * "chains_allrats_Matrix_stimon2Bayesian Logistic Regression Matrix.jld2")["chains_all_rats_Matrix"]
"""
Model fits using Turing
"""
# bd model using a Matrix of w's
# Weights click differences at each time bin seperately
@model logistic_regression_Matrix(x, y, n, σ, bigSigma) = begin
        #Make normal distribution n+1 dimensional
        beta ~ Normal(0, σ)
        #w ~ MultivariateNormal(zeros(10), covarianceMatrix #(10x10 bigSigma)
        # mu is a vector and sigma is a PSD matrix
        #MvNormal(mu, sigma)
        w ~ MvNormal(zeros(10), bigSigma)
        #HIGHERARCHY NEEDS TO GO HERE
        for i = 1:
            v = logistic(beta + transpose(x[i,:]) * w)
            y[i] ~ Bernoulli(v)
        end
end;

# LVR model
# Weights timebins sepeartely but also weights left and right clicks seperately
@model logistic_regression_LVR(x, y, n, σ, bigSigma) = begin
        beta ~ Normal(0, σ)
        #MvNormal(mu, sigma)
        # mu is a vector and sigma is a PSD matrix
        w_R ~ MvNormal(zeros(10), bigSigma)
        w_L ~ MvNormal(zeros(10), bigSigma)
        #Seperate out LVR matrices
        xL = x[:,1:10]
        xR = x[:,11:20]
        for i = 1:n
            v = logistic(beta .+ transpose(xR[i,:])*w_R + transpose(xL[i,:])*w_L)
            y[i] ~ Bernoulli(v)
        end
end;



# bd model using 10 seperate w's
@model logistic_regression(x, y, n, σ) = begin
            #Make normal distribution n+1 dimensional
            beta ~ Normal(0, σ)
            # Multivariate Normal instead of normal
            #w ~ MultivariateNormal(zeros(10), covarianceMatrix #(10x10 bigSigma)
            # mu is a vector and sigma is a PSD matrix
            #MvNormal(mu, sigma)
            #w ~ MvNormal(zeros(10), bigSigma)
            #Change w1 to wtd_1 and w2 to wtd_2 and so on
             wtd_1 ~  Normal(0, σ)
             wtd_2 ~  Normal(0, σ)
             wtd_3 ~  Normal(0, σ)
             wtd_4 ~  Normal(0, σ)
             wtd_5 ~  Normal(0, σ)
             wtd_6 ~  Normal(0, σ)
             wtd_7 ~  Normal(0, σ)
             wtd_8 ~  Normal(0, σ)
             wtd_9 ~  Normal(0, σ)
             wtd_10 ~  Normal(0, σ)
             for i = 1:n
                 v = logistic(beta + wtd_1 * x[i,1] + wtd_2*x[i,2] + wtd_3*x[i,3] + wtd_4*x[i,4] + wtd_5*x[i,5] + wtd_6*x[i,6] + wtd_7*x[i,7] + wtd_8*x[i,8] + wtd_9*x[i,9] + wtd_10*x[i,10])
                 y[i] ~ Bernoulli(v)
             end
end;

# Determines the cutoff point for the number of trials
min_trial = 2^63 -1
for irat= 1 : 15
        # Provides the columns we need for that rat
        regrData = select(data[irat]["stimon"]["X"], :gr,28:37)
        #println("rat ", irat, "'s # of trials: ", size(regrData)[1])
        temp = size(regrData)[1]
        if(temp < min_trial)
            global min_trial = temp
        end
end
# Assuming an 80-20 train-test split these will be the length of the trial dimension of
# of those matrices respectively
train_length = Int(min_trial * 0.8)
test_length = Int(min_trial * 0.2)


"""
Data processing
"""
# Creates a Markov chain using LVR model to be used in model predictions
function data_processing_LVR(irat)
        println("Processing rat ", irat)
        # Grabs the click difference left and right columns from the data set
        LVR_data = select(data[irat]["stimon"]["X"], :gr, 6:15,17:26)
        # Cuts off the number of trials so all rats can be put into one 3D matrix
        LVR_data = LVR_data[1:min_trial,:]
        #Calls earlier function, splitting the data into training and test sets (80-20:training-test split)
        trainset, testset = split_data(LVR_data, target, at = 0.8)
        # Converts training and test data into matrices
        train = Matrix(trainset[:,2:21])
        test = Matrix(testset[:, 2:21])
        # Stores each rats train and test matrix into a 3D matrix
        train_rats_LVR[:,:, irat] = train
        test_rats_LVR[:,:,irat] = test
        # # Converts training and test gr values into column vectors
        # # Stores each rats train and test labels into a 2D matrix
        train_label_LVR[:,irat] = trainset[:, target]
        test_label_LVR[:, irat] = testset[:, target]
        n, _ = size(train)
        chain = mapreduce(c -> sample(logistic_regression_LVR(train_rats_LVR[:,:,irat], train_label_LVR[:,irat], train_length, 1, I), HMC(0.05, 10), 1500),
           chainscat,
               1:3
        )
        return chain
end
#Creates an empty list for the LVR chains of each rat to be stored
chains_LVR_list = []
train_rats_LVR = zeros(train_length, 20, 15)
train_label_LVR = zeros(train_length, 15)
test_rats_LVR = zeros(test_length, 20, 15)
test_label_LVR = zeros(test_length, 15)
for irat = 1:15
    push!(chains_LVR_list, data_processing_LVR(irat))
end
# Creates a Markov chain using bd model (fitted with a matrix of w's) to be used in
#model predictions
function data_processing_bd(irat)
    println("Processing rat ", irat)
    # Provides the columns we need for that rat
    regrData =select(data[irat]["stimon"]["X"], :gr,28:37)
    # Grabs the click differences from the data set
    regrData = regrData[1:min_trial,:]
    #Calls earlier function, splitting the data into training and test sets (80-20:training-test split)
    trainset, testset = split_data(regrData, target, at = 0.8)
    # Converts training and test data into matrices
    train = Matrix(trainset[:, 2:11])
    test = Matrix(testset[:, 2:11])
    # Stores each rats train and test matrix into a 3D matrix
    train_rats[:,:,irat] = train;
    test_rats[:,:, irat] = test;
    # Stores each rats train and test labels into a 2D matrix
    train_label_rats[:,irat] = trainset[:, target]
    test_label_rats[:, irat] = testset[:,target]
    n, _ = size(train)
    # BD Matrix chain
    chain = mapreduce(c -> sample(logistic_regression_Matrix(train_rats[:,:,irat], train_label_rats[:,irat], n, 1, I), HMC(0.05, 10), 3000),
    chainscat,
        1:3
    )
    #Tweaking HMC
    # covariance = 2*I
    # chain = mapreduce(c -> sample(logistic_regression_Matrix(train_rats[:,:,irat], train_label_rats[:,irat], n, 1, I), HMC(0.01, 10), 5000),
    # chainscat,
    #     1:3
    # )
    return chain
end

#Creates an empty list for the Matrix chains of each rat to be stored
chains_Matrix_list = []
train_rats = zeros(train_length,10,15)
train_label_rats = zeros(train_length, 15)
test_rats = zeros(test_length, 10, 15)
test_label_rats = zeros(test_length, 15)
for irat = 1:15
    push!(chains_Matrix_list, data_processing_bd(irat))
end
# Creates a Markov chain using bd model (fitted with 10 seperate w's) to be used in
#model predictions
function data_processing_bd_element(irat)
    println("Processing rat ", irat)
    # Provides the columns we need for that rat
    regrData =select(data[irat]["stimon"]["X"], :gr,28:37)
    # Grabs the click differences from the data set
    regrData = regrData[1:min_trial,:]
    #Calls earlier function, splitting the data into training and test sets (80-20:training-test split)
    trainset, testset = split_data(regrData, target, at = 0.8)
    # Converts training and test data into matrices
    train = Matrix(trainset[:, 2:11])
    test = Matrix(testset[:, 2:11])
    # Stores each rats train and test matrix into a 3D matrix
    train_rats[:,:,irat] = train;
    test_rats[:,:, irat] = test;
    # Stores each rats train and test labels into a 2D matrix
    train_label_rats[:,irat] = trainset[:, target]
    test_label_rats[:, irat] = testset[:,target]
    n, _ = size(train)
    chain = mapreduce(c -> sample(logistic_regression(train_rats[:,:,irat], train_label_rats[:,irat], n, 1), HMC(0.05, 10), 3000),
    chainscat,
        1:3
    )
    # BD Matrix chain
    # chain = mapreduce(c -> sample(logistic_regression_Matrix(train_rats[:,:,irat], train_label_rats[:,irat], n, 1, I), HMC(0.05, 10), 3000),
    # chainscat,
    #     1:3
    # )
    return chain
end
chains_element_list = []
train_rats = zeros(train_length,10,15)
train_label_rats = zeros(train_length, 15)
test_rats = zeros(test_length, 10, 15)
test_label_rats = zeros(test_length, 15)
for irat = 1:15
    push!(chains_element_list, data_processing_bd_element(irat))
end

# Prediction (bd model using a matrix)
function prediction(x::Matrix, chain)
    # Pull the means from each parameter's sampled values in the chain.
    beta = mean(chain[:beta].value)
    w = zeros(10)
    for i = 1:10
        w[i] = mean(chain[:w][i].value)
    end
    n, _ = size(x)

    # Generate a vector to store our predictions.
    v = Vector{Float64}(undef, n)

    # Calculate the logistic function for each element in the test set.
    for i in 1:n
        v[i] = logistic(beta + transpose(x[i,:]) * w)
    end
    return v
end;

# Prediction (bd model done element wise)
function prediction_element(x::Matrix, chain)
    # Pull the means from each parameter's sampled values in the chain.
    beta = mean(chain[:beta].value)
    wtd_1 = mean(chain[:wtd_1].value)
    wtd_2 = mean(chain[:wtd_2].value)
    wtd_3 = mean(chain[:wtd_3].value)
    wtd_4 = mean(chain[:wtd_4].value)
    wtd_5 = mean(chain[:wtd_5].value)
    wtd_6 = mean(chain[:wtd_6].value)
    wtd_7 = mean(chain[:wtd_7].value)
    wtd_8 = mean(chain[:wtd_8].value)
    wtd_9 = mean(chain[:wtd_9].value)
    wtd_10 = mean(chain[:wtd_10].value)

    n, _ = size(x)

    # Generate a vector to store our predictions.
    v = Vector{Float64}(undef, n)

    # Calculate the logistic function for each element in the test set.
    for i in 1:n
        num = logistic(beta .+ wtd_1 * x[i,1] .+ wtd_2 * x[i,2] .+ wtd_3 * x[i,3] .+ wtd_4 * x[i,4] .+ wtd_5 * x[i,5] .+ wtd_6 * x[i,6] .+ wtd_7 * x[i,7] .+ wtd_8 * x[i,8] .+ wtd_9 * x[i,9] .+ wtd_10 * x[i,10])
        v[i] = num;
    end
    return v
end;

#Prediction LVR model
function prediction_LVR(x::Matrix, chain)
    # Pull the means from each parameter's sampled values in the chain.
    beta = mean(chain[:beta].value)
    w_L = zeros(10)
    w_R = zeros(10)
    for i = 1:10
        w_L[i] = mean(chain[:w_L][i].value)
        w_R[i] = mean(chain[:w_R][i].value)
    end
    n, _ = size(x)

    # Generate a vector to store our predictions.
    v = Vector{Float64}(undef, n)
    #Split Left and right
    xL = x[:,1:10]
    xR = x[:,11:20]
    # Calculate the logistic function for each element in the test set.
        v = logistic.(beta .+ (xR * w_R) + (xL * w_L))
    return v
end;

# Set the prediction threshold.
threshold = 0.5

#Matrix model predictions list
predictions_Matrix_list = []
#List of Matrix of ROC scores
#Matrix_ROC_scores = []
# Prediction for each rat using Matrix model
for irat = 1:15
    predictions_Matrix = prediction(test_rats[:,:,irat], chains_Matrix_list[irat])
    #ROC_score = sklmetrics.roc_auc_score(test_label_rats[:,irat], predictions_Matrix)
    #push!(Matrix_ROC_scores, ROC_score)
    push!(predictions_Matrix_list, predictions_Matrix)
end
# LVR model predictions list
predictions_LVR_list = []
#List of Matrix of ROC scores
#LVR_ROC_scores = []
# Makes predictions for each rat using LVR model and stores into list
for irat = 1:15
    predictions_LVR = prediction_LVR(test_rats_LVR[:,:,irat], chains_LVR_list[irat])
    #ROC_score_LVR = sklmetrics.roc_auc_score(test_label_rats[:,irat], predictions_LVR)
    #push!(LVR_ROC_scores, ROC_score_LVR)
    push!(predictions_LVR_list, predictions_LVR)
end

# Prediction for rat 1 using Element model
predictions_element_list = []
for irat = 1:15
    predictions_element = prediction_element(test_rats[:, :, irat], chains_element[irat], threshold)
    push!(predictions_element_list, predictions_element)
end


"""
Plots
"""
n = test_length
"""
1D Psychometric
"""
# Creates a Psychometric plot for bd models and puts it in a list
function Psychometric_1D(irat, predictions, n = test_length)
    # Computes an array of click difference for each trial
    bd_test = zeros(n)
    #If we are looking at performance of LVR model than click difference for each trial is
    # calculated a little differently because the matrix you start with is formatted differently
    for j = 1:n
        # Computes the click difference (#R-#L) for each trial for the other models
        bd_test[j] = sum(test_rats[j,:,irat])
    end
    # Concatonates the rats choice for that trial (1 for right 0 for left) to the recently created bd_train
    bd_test = hcat(bd_test, test_label_rats[:,irat])
    # Concatonates the probability of going right for each trial that was produced by the model
    bd_test = hcat(bd_test, predictions)
    # Find the minimum and maximum click difference and create an array of the length of their difference
    # this will be the (x axis) of our psychometric graph
    min_bd = minimum(bd_test[:,1])
    max_bd = maximum(bd_test[:,1])
    x_axis = max_bd-min_bd
    # Array to count the frequency that of each click difference
    frequ = zeros(Int(x_axis) + 1)
    # Array to count the number of grs for each click difference
    gr_by_click = zeros(Int(x_axis)+ 1)
    # (y axis) of psychometric curve
    #Array of the probabilities of going right for each click difference
    prs = zeros(Int(x_axis) + 1)
    # Same thing for pred
    prs_pred = zeros(Int(x_axis) + 1)
    # Iterates through all trials for that rat
    for j = 1 : n
        # Searches through all click differences and increments the proper
        # element of the frequency array
        frequ[Int(bd_test[j,1] + 1 - min_bd)]+= 1;
        # Properly increments gr_by_click if that click difference ended up in a rightward choice
        if(bd_test[j,2] == 1.0)
            gr_by_click[Int(bd_test[j,1]+1 - min_bd)]+=1;
        end
        prs_pred[Int(bd_test[j,1]+1 - min_bd)]+=bd_test[j,3];
    end
    # Fills probabilities array with probability of right choice (y axis) of psychometric curve
    for j = 1:length(prs)
        prs[j] = (gr_by_click[j]/frequ[j])
        prs_pred[j] = (prs_pred[j]/frequ[j])
    end
    Psycho_1D = plot(min_bd:max_bd, prs, label = "Actual rat")
    Psycho_1D = xlabel!("Click difference (#R - #L)")
    Psycho_1D = ylabel!("Probability of going right")
    Psycho_1D = plot!(min_bd:max_bd, prs_pred, label = "Model prediction")
    push!(Psychometric_1D_plots, Psycho_1D)
end

"""
Matrix model 1D Psychometric (model vs actual rats 1-15)
"""
Psychometric_1D_plots = [];
# Generates Psychometric Plots for Matrix model and puts them in a list
for irat = 1:15
    Psychometric_1D(irat, predictions_Matrix_list[irat])
end
plot(Psychometric_1D_plots[1],Psychometric_1D_plots[2],Psychometric_1D_plots[3],Psychometric_1D_plots[4],Psychometric_1D_plots[5],Psychometric_1D_plots[6], layout = (2, 3), title = "1D Psychometric", xlabel = "#R-L clicks", ylabel = "Probability of gr", legend = false)

plot(Psychometric_1D_plots[7],Psychometric_1D_plots[8],Psychometric_1D_plots[9],Psychometric_1D_plots[10],Psychometric_1D_plots[11],Psychometric_1D_plots[12], layout = (2, 3), title = "1D Psychometric", xlabel = "#R-L clicks", ylabel = "Probability of gr", legend = false)

plot(Psychometric_1D_plots[13],Psychometric_1D_plots[14],Psychometric_1D_plots[15], layout = (1, 3), title = "1D Psychometric", xlabel = "#R-L clicks", ylabel = "Probability of gr", legend = false)

"""
Element model 1D Psychometric (model vs actual rats 1-15)
"""
Psychometric_1D_plots = [];
# Generates Psychometric Plots for element model and puts them in a list
for irat = 1:15
    Psychometric_1D(irat, predictions_element_list[irat])
end
plot(Psychometric_1D_plots[1],Psychometric_1D_plots[2],Psychometric_1D_plots[3],Psychometric_1D_plots[4],Psychometric_1D_plots[5],Psychometric_1D_plots[6], layout = (2, 3), title = "1D Psychometric", xlabel = "#R-L clicks", ylabel = "Probability of gr", legend = false)

plot(Psychometric_1D_plots[7],Psychometric_1D_plots[8],Psychometric_1D_plots[9],Psychometric_1D_plots[10],Psychometric_1D_plots[11],Psychometric_1D_plots[12], layout = (2, 3), title = "1D Psychometric", xlabel = "#R-L clicks", ylabel = "Probability of gr", legend = false)

plot(Psychometric_1D_plots[13],Psychometric_1D_plots[14],Psychometric_1D_plots[15], layout = (1, 3), title = "1D Psychometric", xlabel = "#R-L clicks", ylabel = "Probability of gr", legend = false)

"""
LVR model 1D Psychometric (model vs actual rats 1-15)
"""
# Creates a Psychometric plot for LVR model and puts it in a list
function Psychometric_1D_LVR(irat, predictions, n = test_length)
    # Computes an array of click difference for each trial
    bd_test = zeros(n)
    #If we are looking at performance of LVR model than click difference for each trial is
    # calculated a little differently because the matrix you start with is formatted differently
    for j = 1:n
        #Computes the click difference (#R-#L)for each trial for the LVR model
        bd_test[j] = sum(test_rats_LVR[j,11:20,irat])-sum(test_rats_LVR[j,1:10,irat])
    end
    # Concatonates the rats choice for that trial (1 for right 0 for left) to the recently created bd_train
    bd_test = hcat(bd_test, test_label_rats[:,irat])
    # Concatonates the probability of going right for each trial that was produced by the model
    bd_test = hcat(bd_test, predictions)
    # Find the minimum and maximum click difference and create an array of the length of their difference
    # this will be the (x axis) of our psychometric graph
    min_bd = minimum(bd_test[:,1])
    max_bd = maximum(bd_test[:,1])
    x_axis = max_bd-min_bd
    # Array to count the frequency that of each click difference
    frequ = zeros(Int(x_axis) + 1)
    # Array to count the number of grs for each click difference
    gr_by_click = zeros(Int(x_axis)+ 1)
    # (y axis) of psychometric curve
    #Array of the probabilities of going right for each click difference
    prs = zeros(Int(x_axis) + 1)
    # Same thing for pred
    prs_pred = zeros(Int(x_axis) + 1)
    # Iterates through all trials for that rat
    for j = 1 : n
        # Searches through all click differences and increments the proper
        # element of the frequency array
        frequ[Int(bd_test[j,1] + 1 - min_bd)]+= 1;
        # Properly increments gr_by_click if that click difference ended up in a rightward choice
        if(bd_test[j,2] == 1.0)
            gr_by_click[Int(bd_test[j,1]+1 - min_bd)]+=1;
        end
        prs_pred[Int(bd_test[j,1]+1 - min_bd)]+=bd_test[j,3];
    end
    # Fills probabilities array with probability of right choice (y axis) of psychometric curve
    for j = 1:length(prs)
        prs[j] = (gr_by_click[j]/frequ[j])
        prs_pred[j] = (prs_pred[j]/frequ[j])
    end
    Psycho_1D = plot(min_bd:max_bd, prs, label = "Actual rat")
    Psycho_1D = xlabel!("Click difference (#R - #L)")
    Psycho_1D = ylabel!("Probability of going right")
    Psycho_1D = plot!(min_bd:max_bd, prs_pred, label = "Model prediction")
    push!(Psychometric_1D_plots, Psycho_1D)
end

Psychometric_1D_plots = [];
# Generates Psychometric Plots for LVR model and puts them in a list
for irat = 1:15
    Psychometric_1D(irat, predictions_LVR_list[irat])
end
plot(Psychometric_1D_plots[1],Psychometric_1D_plots[2],Psychometric_1D_plots[3],Psychometric_1D_plots[4],Psychometric_1D_plots[5],Psychometric_1D_plots[6], layout = (2, 3), title = "1D Psychometric", xlabel = "#R-L clicks", ylabel = "Probability of gr", legend = false)

plot(Psychometric_1D_plots[7],Psychometric_1D_plots[8],Psychometric_1D_plots[9],Psychometric_1D_plots[10],Psychometric_1D_plots[11],Psychometric_1D_plots[12], layout = (2, 3), title = "1D Psychometric", xlabel = "#R-L clicks", ylabel = "Probability of gr", legend = false)

plot(Psychometric_1D_plots[13],Psychometric_1D_plots[14],Psychometric_1D_plots[15], layout = (1, 3), title = "1D Psychometric", xlabel = "#R-L clicks", ylabel = "Probability of gr", legend = false)

"""
2D Psychometric (LVR model only)
"""
#Creates a 2D Psychometric plot of the LVR model for a given rat
function Psychometric_LVR_2D_pred(irat, predictions, list, n = test_length)
        # Computes an array of click difference for each trial
        bd_test_LVR_predictions = zeros(n,2)
        bd_test_LVR = zeros(n,2)
        bd_test = zeros(n)
        for j = 1 : n
            #1st column is total left clicks for that trial
            bd_test_LVR[j,1] = sum(test_rats_LVR[j,1:10,irat])
            #2nd column is total right clicks for that trial
            bd_test_LVR[j,2] = sum(test_rats_LVR[j,11:20,irat])
            #bd_test[j] = sum(test_rats_LVR[j,11:20,irat])-sum(test_rats_LVR[j,1:10,irat])
        end
        # Concatonates the actual choice for each trial (1 for right, 0 for left)
        #bd_test = hcat(bd_test, test_label_LVR[:,irat])
        bd_test_LVR = hcat(bd_test_LVR, test_label_LVR[:,irat])
        # Concatonates the probability of going right for each trial that was produced by the model
        bd_test_LVR = hcat(bd_test_LVR, predictions)
        #Goes through predictions and rounds them to 0 or 1
        for k = 1:test_length
            if(bd_test_LVR[k,4] >= 0.5)
                bd_test_LVR[k,4] = 1
            else
                bd_test_LVR[k,4] = 0
            end
        end
    # 2D Psychometric Code
        # # This is for prediction
        grs = Int(sum(bd_test_LVR[:,4]))
        gls = Int(abs(test_length - grs))
        bd_test_LVR_right = zeros(grs,4)
        bd_test_LVR_left = zeros(gls,4)
        grcount = 1
        glcount = 1
        for i = 1:test_length
             # Use this if predictions
             #Put all 1's (rightward predictions) from model into bd_test_LVR_right array
             if(bd_test_LVR[i,4] == 1.0)
                bd_test_LVR_right[grcount,:] = bd_test_LVR[i,:]
                grcount+= 1
                 # Put all 0's (leftward predictions) from model into bd_test_LVR_left array
             else
                 bd_test_LVR_left[glcount,:] = bd_test_LVR[i,:]
                 glcount += 1
             end
        end
        #2D psychometric
        Psycho_2D = scatter(bd_test_LVR_left[:,2], bd_test_LVR_left[:,1], color = "blue", label = "Choice left",cbar = colorbar())
        Psycho_2D = scatter!(bd_test_LVR_right[:,2], bd_test_LVR_right[:,1],  color = "red", label = "Choice Right")
        push!(list, Psycho_2D)
end

#Creates a 2D Psychometric plot of the actual trial for a given rat
function Psychometric_LVR_2D_actual(irat, predictions, list, n = test_length)
        # Computes an array of click difference for each trial
        bd_test_LVR_predictions = zeros(n,2)
        bd_test_LVR = zeros(n,2)
        bd_test = zeros(n)
        for j = 1 : n
            #1st column is total left clicks for that trial
            bd_test_LVR[j,1] = sum(test_rats_LVR[j,1:10,irat])
            #2nd column is total right clicks for that trial
            bd_test_LVR[j,2] = sum(test_rats_LVR[j,11:20,irat])
            #bd_test[j] = sum(test_rats_LVR[j,11:20,irat])-sum(test_rats_LVR[j,1:10,irat])
        end
        # Concatonates the actual choice for each trial (1 for right, 0 for left)
        #bd_test = hcat(bd_test, test_label_LVR[:,irat])
        bd_test_LVR = hcat(bd_test_LVR, test_label_LVR[:,irat])
        # Concatonates the probability of going right for each trial that was produced by the model
        bd_test_LVR = hcat(bd_test_LVR, predictions)
        #Goes through predictions and rounds them to 0 or 1
        for k = 1:test_length
            if(bd_test_LVR[k,4] >= 0.5)
                bd_test_LVR[k,4] = 1
            else
                bd_test_LVR[k,4] = 0
            end
        end
        # 2D Psychometric
        #This is for actual
        grs = Int(sum(bd_test_LVR[:,3]))
        gls = Int(abs(test_length - grs))
        bd_test_LVR_right = zeros(grs,4)
        bd_test_LVR_left = zeros(gls,4)
        grcount = 1
        glcount = 1
        for i = 1:test_length
            #Put all 1's (rightward choices) from actual rat choice into bd_test_LVR_right array
            if(bd_test_LVR[i,3] == 1.0)
                bd_test_LVR_right[grcount,:] = bd_test_LVR[i,:]
                grcount+= 1
                # Put all 0's (leftward choices) from actual rat into bd_test_LVR_left array
            else
                bd_test_LVR_left[glcount,:] = bd_test_LVR[i,:]
                glcount += 1

            end
        end
        #2D psychometric
        Psycho_2D = scatter(bd_test_LVR_left[:,2], bd_test_LVR_left[:,1], color = "blue", label = "Choice left", )
        Psycho_2D = scatter!(bd_test_LVR_right[:,2], bd_test_LVR_right[:,1],  color = "red", label = "Choice Right")
        push!(list, Psycho_2D)
end


Psychometric_2D_plots = [];
#Puts Psycho 2D plots of actual in a list
for irat = 1:15
    Psychometric_LVR_2D_actual(irat, predictions_LVR_list[irat], Psychometric_2D_plots)
end

Psychometric_2D_plots_pred = [];
#Puts Psycho 2D plots of actual in a list
for irat = 1:15
    Psychometric_LVR_2D_pred(irat, predictions_LVR_list[irat], Psychometric_2D_plots_pred)
end
"""
Plots subplots (2x2) of 2D Psychometric Curves with actual rat on the left and model prediction on the right
"""
plot(Psychometric_2D_plots[1],Psychometric_2D_plots_pred[1],Psychometric_2D_plots[2],Psychometric_2D_plots_pred[2], layout = (2, 2), title = "2D Psychometric", xlabel = "Right clicks", ylabel = "Left clicks", legend = false)

plot(Psychometric_2D_plots[3],Psychometric_2D_plots_pred[3],Psychometric_2D_plots[4],Psychometric_2D_plots_pred[4], layout = (2, 2), title = "2D Psychometric", xlabel = "Right clicks", ylabel = "Left clicks", legend = false)

plot(Psychometric_2D_plots[5],Psychometric_2D_plots_pred[5],Psychometric_2D_plots[6],Psychometric_2D_plots_pred[6], layout = (2, 2), title = "2D Psychometric", xlabel = "Right clicks", ylabel = "Left clicks", legend = false)

plot(Psychometric_2D_plots[7],Psychometric_2D_plots_pred[7],Psychometric_2D_plots[8],Psychometric_2D_plots_pred[8], layout = (2, 2), title = "2D Psychometric", xlabel = "Right clicks", ylabel = "Left clicks", legend = false)

plot(Psychometric_2D_plots[9],Psychometric_2D_plots_pred[9],Psychometric_2D_plots[10],Psychometric_2D_plots_pred[10], layout = (2, 2), title = "2D Psychometric", xlabel = "Right clicks", ylabel = "Left clicks", legend = false)

plot(Psychometric_2D_plots[11],Psychometric_2D_plots_pred[11],Psychometric_2D_plots[12],Psychometric_2D_plots_pred[12], layout = (2, 2), title = "2D Psychometric", xlabel = "Right clicks", ylabel = "Left clicks", legend = false)

plot(Psychometric_2D_plots[13],Psychometric_2D_plots_pred[13],Psychometric_2D_plots[14],Psychometric_2D_plots_pred[14], layout = (2, 2), title = "2D Psychometric", xlabel = "Right clicks", ylabel = "Left clicks", legend = false)

plot(Psychometric_2D_plots[15],Psychometric_2D_plots_pred[15], layout = (1, 2), title = "2D Psychometric", xlabel = "Right clicks", ylabel = "Left clicks", legend = false)

## Save chains_LVR_list to be reused
chain_LVR_file = cfg2.EXPORTPATH_DATA * "chains_allrats_LVR_stimon2" *
 "Bayesian Logistic Regression LVR.jld2"
println("Saving: " * string(chain_LVR_file))
save(chain_LVR_file, "chains_all_rats_LVR", chains_LVR_list)

chain_Matrix_file = cfg2.EXPORTPATH_DATA * "chains_allrats_Matrix_stimon2" *
 "Bayesian Logistic Regression Matrix.jld2"
println("Saving: " * string(chain_Matrix_file))
save(chain_Matrix_file, "chains_all_rats_Matrix", chains_Matrix_list)

chain_Element_file = cfg2.EXPORTPATH_DATA * "chains_allrats_Element_stimon2" *
 "Bayesian Logistic Regression Element.jld2"
println("Saving: " * string(chain_Element_file))
save(chain_Element_file, "chains_all_rats_Element", chains_element_list)
