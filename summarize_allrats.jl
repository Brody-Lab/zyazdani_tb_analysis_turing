""" Generates figure for each rat summarizing #Lbups/#Rbups decision
landscape, 2D model fits, and time-varying fits for stimoff and stimon
locking.
"""

using MAT
using GLM
using Dates
using Printf
using PyPlot
using PyCall
using Statistics
using DataFrames
using JLD2, FileIO
# include("code/utils.jl")
include("utils.jl")

###############################################################################

cfg = (
## io options
PROGRAM_NAME    = "summarize_allrats.jl",
IMPORTPATH_DATA_FN = "data/logitFits_allrats_frozen_noise_500msLim_50msBin_0msOverlap.jld2", # frozen_noise
IMPORTPATH_DATA_CU = "data/logitFits_allrats_chuckrats_update_500msLim_50msBin_0msOverlap.jld2",  # chuckrats_update
EXPORTPATH_FIGS = "figs/",
EXPORTPATH_DATA = "data/",
SAVE_FIGS       = false,
SAVE_DATA       = false,

## analysis options
RATS_TO_USE = [283, 284, 285, 289, 290, 292, 293, 295, 296, 298, 301, 304,
               305, 305, 311, 313, 314, 316, 317, 319, 322, 328, 330, 331,
               335, 336, 339],
VERS_TO_USE     = [2, 2, 2, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
CLASSIC_OR_FREQ = [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1]
)

###############################################################################

function main(cfg)
    RATIDS = ["K" * string(cfg.RATS_TO_USE[irat]) for irat = 1 : length(cfg.RATS_TO_USE)]

    if cfg.SAVE_FIGS
        exportpath = mkdir_dated(cfg.EXPORTPATH_FIGS, cfg.PROGRAM_NAME)
    end

    ## Load data and fit logit model to click totals
    data_fn = load(cfg.IMPORTPATH_DATA_FN)["data"]
    data_cu = load(cfg.IMPORTPATH_DATA_CU)["data"]
    ## Grab ratnames for both pipelines
    fn_ratnames = [data_fn[i]["fname"][1:4] for i = 1 : data_fn["nrats"]]
    cu_ratnames = [data_cu[i]["fname"][1:4] for i = 1 : data_cu["nrats"]]

    ## Organize into single dictionary
    R = Dict()
    for irat = 1 : length(RATIDS)
        if cfg.VERS_TO_USE[irat] == 1
            # Find the rat in data_fn
            ratidx = argmax(fn_ratnames .== RATIDS[irat])
            R[irat] = data_fn[ratidx]
        elseif cfg.VERS_TO_USE[irat] == 2
            ratidx = argmax(cu_ratnames .== RATIDS[irat])
            R[irat] = data_cu[ratidx]
        end
    end
    R["nrats"] = length(R)
    R["nbins"] = data_fn["nbins"]
    R["xaxis_stimon"]  = data_fn["xaxis_stimon"]
    R["xaxis_stimoff"] = data_fn["xaxis_stimoff"]

    ## Compute performance, bias, etc across all rats
    perf   = [sum(R[irat]["wholetrl"]["X"].hh)/R[irat]["ntrls"] for irat = 1 : R["nrats"]]
    rlbias = [R[irat]["wholetrl"]["logit"].model.pp.beta0[2] - abs(R[irat]["wholetrl"]["logit"].model.pp.beta0[end]) for irat = 1 : R["nrats"]]
    bias   = [R[irat]["wholetrl"]["logit"].model.pp.beta0[1] for irat = 1 : R["nrats"]]

    ## Collate
    wts_bd_off = zeros(R["nbins"])
    wts_bd_on  = zeros(R["nbins"])
    for irat = 1 : length(R["nrats"])
        wts_bd_off = wts_bd_off + R[irat]["stimoff"]["logit_d"].model.pp.beta0[2:end]
        wts_bd_on  = wts_bd_on  + R[irat]["stimon"]["logit_d"].model.pp.beta0[2:end]
    end
    wts_bd_off = wts_bd_off ./ R["nrats"]
    wts_bd_on  = wts_bd_on  ./ R["nrats"]

    # figure()
    # plot(xaxis_stimon, wts_bd_on)
    # plot(xaxis_stimoff, wts_bd_off)

    figure();
    subplot(131) ; hist(perf[cfg.CLASSIC_OR_FREQ .== 1], label="Classic", alpha=0.7);
                   hist(perf[cfg.CLASSIC_OR_FREQ .== 2], label="Frequency", alpha=0.7);
    xlabel("Performance") ; legend();

    subplot(132) ; hist(rlbias[cfg.CLASSIC_OR_FREQ .== 1], label="Classic", alpha=0.7);
                   hist(rlbias[cfg.CLASSIC_OR_FREQ .== 2], label="Frequency", alpha=0.7);
    xlabel(L"|$w_R$| - |$w_L$|") ; legend();

    subplot(133) ; hist(bias[cfg.CLASSIC_OR_FREQ .== 1], label="Classic", alpha=0.7);
                   hist(bias[cfg.CLASSIC_OR_FREQ .== 2], label="Frequency", alpha=0.7);
    xlabel("β") ; legend();

    # figure()
    # subplot(131) ; hist(perf)   ; xlabel("Performance across all rats") ; xlim([0, 1]);
    # subplot(132) ; hist(rlbias) ; xlabel("RLBias across all rats")
    # subplot(133) ; hist(bias)   ; xlabel("Bias across all rats") ; title("log odds")

    # figure() ; scatter(rlbias, perf) ; ylabel("Perf") ; xlabel("RLBias")
    # figure() ; scatter(bias, perf)   ; xlabel("Bias") ; ylabel("Perf")
    # figure() ; scatter(bias, rlbias) ; xlabel("Bias") ; ylabel("RLBias")

    if cfg.SAVE_DATA
        filename = cfg.EXPORTPATH_DATA * "R_allrats.jld2"
        ## Save dataframes for next step in logit analysis
        println("Saving: " * string(filename))
        save(filename, "R", R)
    end
end

main(cfg)
