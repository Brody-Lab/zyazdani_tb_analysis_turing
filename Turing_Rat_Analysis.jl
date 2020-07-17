"""
Turing analysis of Rats
Zach Yazdani
Code adapted from Bayesian Logistic Regression tutorial: https://turing.ml/dev/tutorials/2-logisticregression/ tutorial

"""

"""
Action items left to complete for Stage 1
-~Plot psychometric curves and ROC
-~Make data 3D instead of 2D so that each rats data is kept throughout flow of whole program
-Extend model to make it higherarchical
-Test model
-Clean up wtd_1, wtd_2 into a matrix
-Might also want to plot confidence interval


"""
cfg = (
## io options
TITLE           = "frozen_noise_1D_bd_auc",
#TITLE           = "chuckrats_update",
# TITLE           = "frozen_noise2",
PROGRAM_NAME    = "fit_logits_kfoldcv.jl",
IMPORTPATH_DATA = "data/regrMats_allrats_frozen_noise_500msLim_50msBin_0msOverlap.jld2",
#IMPORTPATH_DATA = "data/regrMats_allrats_chuckrats_update_500msLim_50msBin_0msOverlap.jld2",
EXPORTPATH_DATA = "data/",
SAVE_DATA       = true,

EXPORTPATH_FIGS = "figs/",
SAVE_FIGS       = false,

## analysis and plotting options
K = 15,
PLOT_KERNEL_MAGNITUDE = true, # Whether to plot L/R time-varying kernels'
                              # magnitudes, instead of opposite to one
                              # another.
PLOT_BUPDIFF_KERNEL   = false,  # Whether to plot the time-varying click
                                # difference kernel as well
ERRORBARS = "ci95"              # 'ci95', 'stderr'
)

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

using GLM

using Printf
using PyPlot
using PyCall
using MLBase
using Statistics
using DataFrames
using Conda

sklmetrics = pyimport("sklearn.metrics")



# Turn off progress monitor.
Turing.turnprogress(false)

# Function to be used to split the data into training and test sets ß
function split_data(df, target; at = 0.70)
    shuffled = shuffleobs(df)
    trainset, testset = stratifiedobs(row -> row[target],                                  shuffled, p = at)
end

#Factors used to predict gr so wtd_1 - wtd_10
features = [:wtd_1, :wtd_10, :wtd_2, :wtd_3,:wtd_4,:wtd_5,:wtd_6,:wtd_7,:wtd_8,:wtd_9]

#The column that we want to predict (gr is whether the rat went right or not)
target = :gr

#Imports data
data = load(cfg.IMPORTPATH_DATA)["regrMats"]

# Fits the model
#DEFINE MODEL AS HIGHERARCHICAL
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
         #w = hcat(wtd_1,wtd_2,wtd_3,wtd_4,wtd_5,wtd_6,wtd_7,wtd_8,wtd_9,wtd_10)

        #k = 0;
        #for j = 1:irat
        for i = 1:n
            #HIGHERARCHY NEEDS TO GO HERE
            #10 things one for each tb for that trial]
            #for j = 1:10
            #     k+= w[:,j] * x[i, j]
            # end
            # v = logistic(k)
            #v = logistic(beta + wtd_1 * x[i,1] + wtd_2*x[i,2] + wtd_3*x[i,3] + wtd_4*x[i,4] + wtd_5*x[i,5] + wtd_6*x[i,6] + wtd_7*x[i,7] + wtd_8*x[i,8] + wtd_9*x[i,9] + wtd_10*x[i,10])
            #v[irat] = logistic(beta + wtd_1 * x[i,1,j] + wtd_2*x[i,2,j] + wtd_3*x[i,3,j] + wtd_4*x[i,4] + wtd_5*x[i,5] + wtd_6*x[i,6] + wtd_7*x[i,7] + wtd_8*x[i,8] + wtd_9*x[i,9] + wtd_10*x[i,10])
            v = logistic(beta + wtd_1 * x[i,1] + wtd_2*x[i,2] + wtd_3*x[i,3] + wtd_4*x[i,4] + wtd_5*x[i,5] + wtd_6*x[i,6] + wtd_7*x[i,7] + wtd_8*x[i,8] + wtd_9*x[i,9] + wtd_10*x[i,10])
            #y[i,irat] ~ Bernoulli(v[irat])
            y[i] ~ Bernoulli(v)
    end
    end;
# Determines the cutoff point for the number of trials
min_trial = 2^63 -1
for irat= 1 : 15
        # Provides the columns we need for that rat
        regrData =select(data[irat]["stimon"]["X"], :gr,28:37)
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

chains = []
train_rats = zeros(train_length,10,15)
train_label_rats = zeros(train_length, 15)
test_rats = zeros(test_length, 10, 15)
test_label_rats = zeros(test_length, 15)
rat_num = 0;
# Crunches the data for all rats
for irat = 1 : 15
        # Provides the columns we need for that rat
        regrData =select(data[irat]["stimon"]["X"], :gr,28:37)
        # Cuts off the number of trials so all rats can be put into one 3D matrix
        regrData = regrData[1:min_trial,:]
        #Calls earlier function, splitting the data into training and test sets (80-20:training-test split)
        trainset, testset = split_data(regrData, target, at = 0.8)
        #Rescales all values to be centered at 0
         # for i in features
         #   μ, σ = rescale!(trainset[!, i], obsdim=1)
         #   rescale!(testset[!, i], μ, σ, obsdim=1)
         # end

        # Converts training and test data into matrices
        train = Matrix(trainset[:, features])
        test = Matrix(testset[:, features])
        # Stores each rats train and test matrix into a 3D matrix
        train_rats[:,:,irat] = train;
        test_rats[:,:, irat] = test;
        # # Converts training and test gr values into column vectors
        train_label = trainset[:, target]
        test_label = testset[:, target];
        # # Stores each rats train and test labels into a 2D matrix
        train_label_rats[:,irat] = train_label
        test_label_rats[:, irat] = test_label
        n, _ = size(train)


        #Example = zeros(#length(train),10, 15)
        # for(i = 1: 15)
        #     Example[:,:,i] = train 3D
              #test 3D
              #train_label 2D
              #test_label 2D

        # end


          chain = mapreduce(c -> sample(logistic_regression(train_rats[:,:,irat], train_label_rats[:,irat], n, 1), HMC(0.05, 10), 3000),
          chainscat,
              1:3
          )
          push!(chains,chain)
end
#describe(chain)

#plot(chains[1])

function prediction(x::Matrix, chain, threshold)
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
    # for i in features
    #     i = mean(chain[i].value)
    # end
    # Retrieve the number of rows.
    n, _ = size(x)

    # Generate a vector to store our predictions.
    v = Vector{Float64}(undef, n)

    # Calculate the logistic function for each element in the test set.
    for i in 1:n
        num = logistic(beta .+ wtd_1 * x[i,1] .+ wtd_2 * x[i,2] .+ wtd_3 * x[i,3] .+ wtd_4 * x[i,4] .+ wtd_5 * x[i,5] .+ wtd_6 * x[i,6] .+ wtd_7 * x[i,7] .+ wtd_8 * x[i,8] .+ wtd_9 * x[i,9] .+ wtd_10 * x[i,10])
        # if num >= threshold
        #     v[i] = 1
        # else
        #     v[i] = 0
        # end
        v[i] = num;
    end
    return v
end;

# Set the prediction threshold.
threshold = 0.5
predicted_gr_avg = 0;
# Make the predictions.

# Generalize this so that it predicts for all rats using a for loop
predictions_list = []
ROC_scores = []
# Make predictions for test set and
for irat = 1:15
    predictions = prediction(test_rats[:, :, irat], chains[irat], threshold)
    ROC_score = sklmetrics.roc_auc_score(test_label_rats[:,irat], predictions)
    push!(ROC_scores, ROC_score)
    push!(predictions_list, predictions)
end
xlabel("ROC scores")
ylabel("Frequency")
hist(ROC_scores,50)

# Some sample diagnostic measurements
# Calculate MSE for our test set.
loss = sum((predictions - test_label).^2) / length(test_label)

grs = sum(test_label_rats[:,1])
not_grs = length(test_label) - grs
predicted_grs = sum(test_label_rats[:,1] .== predictions_list[1] .== 1)
predicted_not_grs = sum(test_label .== predictions .== 0)
# Percentage accuracy of grs of the model
predicted_grs/grs



#Plot psychometric curve for each rat
# x - axis is difference number of clicks abs(L-R)
# y - axis is percentage of going to the right side
# The model's and the actual and see if they match
# Undo setting it at 0
n = test_length

#for irat = 1:15
    # Computes an array of click difference for each trial
    # Graphed 6 of 15
    bd_test = zeros(n)
    for j = 1 : n
        bd_test[j] = sum(test_rats[j,:,irat])
    end
    # Concatonates the actual choice for each trial (1 for right, 0 for left)
    bd_test = hcat(bd_test, test_label_rats[:,irat])
    # Concatonates the probability of going right for each trial that was produced by the model
    bd_test = hcat(bd_test, predictions_list[irat])
    # Find the minimum and maximum click difference and create an array of the length of their difference
    # this will be the (x axis) of our psychometric graph
    min_bd = minimum(bd_test[:,1])
    max_bd = maximum(bd_test[:,1])
    diff= max_bd-min_bd
    # Array to count the frequency that of each click difference
    frequ = zeros(Int(diff) + 1)
    # Array to count the number of grs for each click difference
    gr_by_click = zeros(Int(diff)+ 1)
    # Same thing for pred
    #gr_by_click_pred = zeros(Int(diff)+ 1)
    # (y axis) of psychometric curve
    #Array of the probabilities of going right for each click difference
    prs = zeros(Int(diff) + 1)
    # Same thing for pred
    prs_pred = zeros(Int(diff) + 1)
    # Iterates through all trials for that rat
    for j = 1 : n
        # Insert a threshold to get it back in 0's and ones.
        # if data > threshold then = 1
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
    plot(min_bd:max_bd, prs, label = "Actual rat")
    xlabel!("Click difference (#R - #L)")
    ylabel!("Probability of going right")
    plot!(min_bd:max_bd, prs_pred, label = "Model prediction")
#end



# bd_test = zeros(size(test)[1])
# for j = 1 : length(bd_test)
#     bd_test[j] = sum(test[j,:])
# end
# # Concatonates the rats choice for that trial (1 for right 0 for left)
# bd_test = hcat(bd_test, predictions)
# Find the minimum and maximum click difference and create an array of the length of their difference
# this will be the (x axis) of our psychometric graph
# min_bd = minimum(bd_test[:,1])
# max_bd = maximum(bd_test[:,1])
# diff= max_bd-min_bd
# #x axis
# frequ = zeros(Int(diff) + 1)
# gr_by_click = zeros(Int(diff)+ 1)
# prs_pred = zeros(Int(diff) + 1)
#
# for j = 1 : size(bd_test)[1]
#     frequ[Int(bd_test[j,1] + 1 - min_bd)]+= 1;
#     if(bd_test[j,2] == 1.0)
#         gr_by_click[Int(bd_test[j,1]+1 - min_bd)]+=1;
#     end
# end
# for j = 1:length(prs_pred)
#     prs_pred[j] = (gr_by_click[j]/frequ[j])
# end
#
# plot!(min_bd:max_bd, prs_pred)
# plot(min_bd:max_bd, prs)


#samplePSDMatrix = [2 -1 0; -1 2 -1; 0 -1 2]
