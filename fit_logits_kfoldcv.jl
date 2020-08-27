""" Fits logistic regression models to the preprocessed data, and
measures, for each rat, their cross-validated performance.

Currently: do simple k-fold cross-validation.
TODO:
- [ ] Nested k-fold cross-validation
"""

using MAT
using GLM
using Dates
using Printf
using PyPlot
using PyCall
using MLBase
using Statistics
using DataFrames
using JLD2, FileIO
using Conda
include("utils.jl")
sklmetrics = pyimport("sklearn.metrics")

###############################################################################

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

###############################################################################

## Load data
data = load(cfg.IMPORTPATH_DATA)["regrMats"]
nbins = data["nbins"]
nrats = data["nrats"]
xaxis_stimon  = data["xaxis_stimon"]
xaxis_stimoff = data["xaxis_stimoff"]

## Iterate over each rat, fitting models and computing AUC
for irat = 1 : data["nrats"]
    ratid = data[irat]["fname"][1:4]
    println("Processing: " * string(irat) * ": " * ratid * " ...")
    ## Create arrays to store CV results
    data[irat]["wholetrl"]["auc"]    = zeros(cfg.K)
    data[irat]["wholetrl"]["auc_bd"] = zeros(cfg.K)
    for timelock in ["stimoff", "stimon"]
        data[irat][timelock]["auc_bd"]    = zeros(cfg.K)
        data[irat][timelock]["auc_rl"]    = zeros(cfg.K)
        data[irat][timelock]["auc_total"] = zeros(cfg.K)
    end

    ## Generate expressions for fitting each of the logit models
    expr_bd = construct_logit_expr("gr", ["wtd_"], data["nbins"])
    expr_rl = construct_logit_expr("gr", ["wtR_", "wtL_"], data["nbins"])

    ## Split up data into K segments, with one test segment:
    idxs = collect(Kfold(data[irat]["ntrls"], cfg.K))

    ## K-Fold cross-validation!
    kbetas_bd = [];
    kbetas_rl = [];
    kbetas_total = [];
    for k = 1 : cfg.K
        ## Grab train and test set indices
        trainset = idxs[k]
        testset = deleteat!(collect(1:data[irat]["ntrls"]), trainset)

        ## Fit the time-varying bup-diff model (bd), time-varying L/R click model (rl),
        ## the total L/R for the WINDOW_STIM_LEN model (total), and the whole trial
        ## model (wholetrl), which contains the total number of L and R clicks for each trial
        X = data[irat]["wholetrl"]["X"]
        logit_wholetrl = glm(@formula(gr ~ wtRtot + wtLtot), X[trainset,:], Binomial(), LogitLink())
        data[irat]["wholetrl"]["auc"][k] = sklmetrics.roc_auc_score(X[testset,:].gr, predict(logit_wholetrl, X[testset,:]))

        bd = data[irat]["wholetrl"]["X"].wtRtot - data[irat]["wholetrl"]["X"].wtLtot
        df = DataFrame(gr=data[irat]["wholetrl"]["X"].gr, bd=bd)
        logit_wholetrlbd = glm(@formula(gr ~ bd), df, Binomial(), LogitLink())
        data[irat]["wholetrl"]["auc_bd"][k] = sklmetrics.roc_auc_score(df[testset,:].gr, predict(logit_wholetrlbd, df[testset,:]))

        for timelock in ["stimoff", "stimon"]
            ## Grab Xtrain, Xtest
            Xtrain = data[irat][timelock]["X"][trainset,:]
            Xtest  = data[irat][timelock]["X"][testset,:]

            ## Fit models
            logit_bd = glm(eval(Meta.parse(expr_bd)), Xtrain, Binomial(), LogitLink())
            logit_rl = glm(eval(Meta.parse(expr_rl)), Xtrain, Binomial(), LogitLink())
            logit_total = glm(@formula(gr ~ wtRtot + wtLtot), Xtrain, Binomial(), LogitLink())

            ## And compute performance on testing set - AUC of the ROC curve
            auc_bd    = sklmetrics.roc_auc_score(Xtest.gr, predict(logit_bd, Xtest))
            auc_rl    = sklmetrics.roc_auc_score(Xtest.gr, predict(logit_rl, Xtest))
            auc_total = sklmetrics.roc_auc_score(Xtest.gr, predict(logit_total, Xtest))

            ## Store results
            data[irat][timelock]["auc_bd"][k]    = auc_bd
            data[irat][timelock]["auc_rl"][k]    = auc_rl
            data[irat][timelock]["auc_total"][k] = auc_total
        end
    end
end

if cfg.SAVE_DATA
    filename1 = cfg.EXPORTPATH_DATA * "AUCs_allrats_" * cfg.TITLE * "_" *
               string(cfg.K) * "folds.jld2"
    ## Save dataframes for next step in logit analysis
    println("Saving: " * string(filename1))
    save(filename1, "data", data)
end
