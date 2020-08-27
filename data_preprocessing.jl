""" Preprocessing the pbups data from .mat files into usable JLD files.
Essentially, parses data out into DataFrames that can then easily be
fed into GLM.jl logistic regression models.
"""

using MAT
using DataFrames
using JLD2, FileIO
include("utils.jl")

#HIS DIRECTORY
################################################################################

# cfg = (
# TITLE           = "chuckrats_update",
# # TITLE           = "frozen_noise",
# IMPORTPATH_DATA = "data/chuckrats_update/",   # Import path to .mat behavior
# # IMPORTPATH_DATA = "data/frozen_noise/",     # Import path to .mat behavior
# EXPORTPATH_DATA = "data/",                    # Export path for processed data
# STIM_WINDOW_LEN = 500,                        # Stimulus window length, in ms
# MSPERSEG        = 50,                         # Bin width, in ms
# MSOVERLAP       = 0                           # Bin overlap, in ms
# )

#MY DIRECTORY
################################################################################
cfg = (
TITLE           = "frozen_noise",
# TITLE           = "frozen_noise",
IMPORTPATH_DATA = "data/frozen_noise",   # Import path to .mat behavior
# IMPORTPATH_DATA = "data/frozen_noise/",     # Import path to .mat behavior
EXPORTPATH_DATA = "data/",                    # Export path for processed data
STIM_WINDOW_LEN = 500,                        # Stimulus window length, in ms
MSPERSEG        = 50,                         # Bin width, in ms
MSOVERLAP       = 0                           # Bin overlap, in ms
)
##################################################
filelist = searchdir(cfg.IMPORTPATH_DATA, ".mat")

nrats = length(filelist)
nbins = Int(cfg.STIM_WINDOW_LEN / cfg.MSPERSEG)

regrMats = Dict{Any,Any}("cfg" => cfg,
                       "nrats" => nrats,
                       "nbins" => nbins,
                    "filelist" => filelist)

println("Crunching the rat data!")
for irat = 1 : nrats
    println("Running rat: " * filelist[irat] * " ...")

    ## Allocate Dict for rat
    regrMats[irat] = Dict()
    regrMats[irat]["fname"] = filelist[irat]
    regrMats[irat]["stimon"] = Dict()
    regrMats[irat]["stimoff"] = Dict()

    ## Read in data, grab violation information
    data, parsed = load_rat_behavioral_data(filelist[1], cfg.TITLE)
    #ORIGINAL
    # data, parsed = load_rat_behavioral_data(cfg.IMPORTPATH_DATA * filelist[irat], cfg.TITLE)
        #ERROR:  File "yanar_logistic_regression-master/data/frozen_noise/frozen_noise_DATAK283_classic_40_ALL.mat"
    # violtrls = isnan.(data["gr"])
    # n_rbups = [length(data["bt"][i]["right"]) for i = 1 : length(data["gr"])]
    # n_lbups = [length(data["bt"][i]["left"])  for i = 1 : length(data["gr"])]
    # bd = (n_rbups - n_lbups)'
    # regrMats[irat]["violtrls"] = Dict()
    # regrMats[irat]["violtrls"]["bd"] = bd[violtrls]
    # regrMats[irat]["violtrls"]["cr"] = bd[violtrls] .> 0  # Correct response is r
    # regrMats[irat]["violtrls"]["cl"] = bd[violtrls] .< 0  # Correct response is l

    ## Exclude trials where stimulis less than cfg["STIM_WINDOW_LEN"]
    long_trls = parsed["pd"] .>= cfg.STIM_WINDOW_LEN / 1000
    for k in keys(parsed)
        # println("key: " * string(k))
        if k == "b"
            # frozen_noise
            if cfg.TITLE == "frozen_noise"
                parsed[k]["left"]  = parsed["b"]["left"][long_trls]
                parsed[k]["right"] = parsed["b"]["right"][long_trls]
            elseif cfg.TITLE == "chuckrats_update"
                # Chuckrats_update
                parsed[k]["left"]  = parsed["b"]["left"][long_trls']
                parsed[k]["right"] = parsed["b"]["right"][long_trls']
            else
                error("Specify whether this is chuckrats_update or frozen_noise")
            end
        else
            parsed[k] = parsed[k][long_trls]
        end
    end

    ## Generate regressor matrices with bups timelocked to stimoff and stimon
    ntrls = length(parsed["pd"]) ; regrMats[irat]["ntrls"] = ntrls
    for timelock in ["stimoff", "stimon"]
        Xd = zeros(ntrls, nbins)  ## For keeping track of bup difference
        Xr = zeros(ntrls, nbins)  ## For keeping track of right bups
        Xl = zeros(ntrls, nbins)  ## For keeping track of left bups
        Xtotal = zeros(ntrls, 2)  ## For keeping track of total L/R bups in this window

        ## Iterate through trials, binning and storing clicktimes into
        ## the allocated X matrices
        for itrl = 1 : ntrls
            ## Grab data for this trial
            pd    = parsed["pd"][itrl] * 1000
            rbups = parsed["b"]["right"][itrl] .* 1000
            lbups = parsed["b"]["left"][itrl] .* 1000
            # rbups = parsed["b"][itrl]["right"] .* 1000
            # lbups = parsed["b"][itrl]["left"]  .* 1000
            if isa(rbups, Float64) rbups = [rbups] end
            if isa(lbups, Float64) lbups = [lbups] end
            ## Bin left and right clicks aligned to stimulus onset and offset
            rbinned, lbinned, binlims = bin_bups_trial(pd, rbups, lbups, nbins,
                                                       timelock, cfg.STIM_WINDOW_LEN)
            Xr[itrl,:] = rbinned
            Xl[itrl,:] = lbinned
            Xd[itrl,:] = rbinned - lbinned
            Xtotal[itrl,1], Xtotal[itrl,2] = sum(rbinned), sum(lbinned)
        end

        ## Construct dataframe
        dict_X = Dict("hh" => convert.(Int64, parsed["hh"]),
                      "gr" => convert.(Int64, parsed["gr"]),
                      "gl" => -1 .* (convert.(Int64, parsed["gr"]) .- 1),
                      "cr" => (Xtotal[:,1] - Xtotal[:,2]) .> 0,
                      "cl" => (Xtotal[:,1] - Xtotal[:,2]) .< 0,
                  "wtRtot" => Xtotal[:,1],
                  "wtLtot" => Xtotal[:,2])
        for ibin = 1 : nbins
            dict_X["wtd_" * string(ibin)] = Xd[:,ibin]
            dict_X["wtR_" * string(ibin)] = Xr[:,ibin]
            dict_X["wtL_" * string(ibin)] = Xl[:,ibin]
        end
        regrMats[irat][timelock] = Dict{Any,Any}()
        regrMats[irat][timelock]["X"] = DataFrame(dict_X)
    end

    ## Note that we also want to make a dataframe that keeps track of
    ## the absolute total number of L/R clicks per trial, for the left
    ## and right. We add the correct right / correct left params here too.
    # chuckrats_update
    dict_X = Dict("hh" => convert.(Int64, parsed["hh"]),
                  "gr" => convert.(Int64, parsed["gr"]),
                  "gl" => -1 .* (convert.(Int64, parsed["gr"]) .- 1),
              "wtRtot" => [length(parsed["b"]["right"][i]) for i = 1 : ntrls],
              "wtLtot" => [length(parsed["b"]["left"][i]) for i = 1 : ntrls])
    dict_X["cr"] = (dict_X["wtRtot"] - dict_X["wtLtot"]) .> 0
    dict_X["cl"] = (dict_X["wtRtot"] - dict_X["wtLtot"]) .< 0
    regrMats[irat]["wholetrl"] = Dict{Any,Any}("X" => DataFrame(dict_X))

    ## Generate x axis (in the form of centers of time bins) for the
    ## binned data
    regrMats["binlims_stimoff"] = Array(range(-cfg.STIM_WINDOW_LEN, 0, length=Int(nbins+1)))
    regrMats["xaxis_stimoff"] = [(regrMats["binlims_stimoff"][i]+regrMats["binlims_stimoff"][i+1])/2 for i = 1 : nbins]

    regrMats["binlims_stimon"] = Array(range(0, cfg.STIM_WINDOW_LEN, length=Int(nbins+1)))
    regrMats["xaxis_stimon"] = [(regrMats["binlims_stimon"][i]+regrMats["binlims_stimon"][i+1])/2 for i = 1 : nbins]
end

## Save dataframes for next step in logit analysis
filename1 = cfg.EXPORTPATH_DATA * "regrMats_allrats_" * cfg.TITLE * "_" *
           string(cfg.STIM_WINDOW_LEN) * "msLim_" *
           string(cfg.MSPERSEG)        * "msBin_" *
           string(cfg.MSOVERLAP) * "msOverlap.jld2"
println("Saving: " * string(filename1))
save(filename1, "regrMats", regrMats)
