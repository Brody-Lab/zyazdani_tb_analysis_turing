""" Generates figure for each rat summarizing #Lbups/#Rbups decision
landscape, 2D model fits, and time-varying fits for stimoff and stimon
locking.

TODO:
- [ ] strong need for refactor
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
include("utils.jl")

###############################################################################

cfg = (
## io options
#TITLE           = "chuckrats_update",
TITLE           = "frozen_noise",
PROGRAM_NAME    = "summarize_rats.jl",
IMPORTPATH_DATA = "data/regrMats_allrats_frozen_noise_500msLim_50msBin_0msOverlap.jld2",
IMPORTPATH_AUC  = "data/AUCs_allrats_frozen_noise_1D_bd_auc_15folds.jld2",
#IMPORTPATH_DATA = "data/regrMats_allrats_chuckrats_update_500msLim_50msBin_0msOverlap.jld2",
#IMPORTPATH_AUC  = "data/AUCs_allrats_chuckrats_1D_bd_auc_500msLim_50msBin_0msOverlap.jld2",
EXPORTPATH_DATA = "data/",
SAVE_DATA       = false,

EXPORTPATH_FIGS = "figs/",
SAVE_FIGS       = true,

## analysis and plotting options
PLOT_KERNEL_MAGNITUDE = true, # Whether to plot L/R time-varying kernels'
                              # magnitudes, instead of opposite to one
                              # another.
PLOT_BUPDIFF_KERNEL   = true,  # Whether to plot the time-varying click
                              # difference kernel as well
ERRORBARS = "ci95"            # 'ci95', 'stderr'
)

###############################################################################

if cfg.SAVE_FIGS
    exportpath = mkdir_dated(cfg.EXPORTPATH_FIGS, cfg.PROGRAM_NAME)
end

## Load data and fit logit model to click totals
# data = load(cfg["IMPORTPATH_STIMON_DATA"])["regrMats"]
data = load(cfg.IMPORTPATH_DATA)["regrMats"]
aucs = load(cfg.IMPORTPATH_AUC)["data"]
nbins = data["nbins"]
nrats = data["nrats"]
xaxis_stimon  = data["xaxis_stimon"]
xaxis_stimoff = data["xaxis_stimoff"]

for irat = 1 : data["nrats"]
    println("Processing: " * string(irat) * " ...")
    ratid = data[irat]["fname"][1:4]

    ## Grab AUCs
    auc_wholetrl = aucs[irat]["wholetrl"]["auc"]
    auc_wholetrlbd = aucs[irat]["wholetrl"]["auc_bd"]

    auc_off_rl = aucs[irat]["stimoff"]["auc_rl"]
    auc_off_bd = aucs[irat]["stimoff"]["auc_bd"]
    auc_off_total = aucs[irat]["stimoff"]["auc_total"]

    auc_on_rl = aucs[irat]["stimon"]["auc_rl"]
    auc_on_bd = aucs[irat]["stimon"]["auc_bd"]
    auc_on_total = aucs[irat]["stimon"]["auc_total"]

    ## Fitting
    ## Construct expressions for glm: "y ~ wtR_1 + wtR_2 + ...", fit to
    ## data locked to the stimulus onset and offset, as well as to the
    ## whole trial bups
    expr_d  = construct_logit_expr("gr", ["wtd_"], data["nbins"])
    expr_rl = construct_logit_expr("gr", ["wtR_", "wtL_"], data["nbins"])
    for timelock in ["stimoff", "stimon"]
        logit_d = glm(eval(Meta.parse(expr_d)), data[irat][timelock]["X"], Binomial(), LogitLink())
        logit_rl = glm(eval(Meta.parse(expr_rl)), data[irat][timelock]["X"], Binomial(), LogitLink())
        logit_total = glm(@formula(gr ~ wtRtot + wtLtot), data[irat][timelock]["X"], Binomial(), LogitLink())
        data[irat][timelock]["logit_d"] = logit_d
        data[irat][timelock]["logit_rl"] = logit_rl
        data[irat][timelock]["logit_total"] = logit_total
    end
    data[irat]["wholetrl"]["logit"] = glm(@formula(gr ~ wtRtot + wtLtot),
                                          data[irat]["wholetrl"]["X"], Binomial(), LogitLink())

    ## Construct figure
    figure(figsize=(16,10))

    ##############################
    ## RAT PSYCHOMETRIC SURFACE ##
    ##############################
    subplot(231)
    rat_decision_surf = bd_gr_surface_matrix(data[irat]["wholetrl"]["X"].wtRtot,
                                             data[irat]["wholetrl"]["X"].wtLtot,
                                             data[irat]["wholetrl"]["X"].gr)
    imshow(rat_decision_surf, origin="lower", cmap=get_cmap("RdBu"))
    xlabel("#R Clicks") ; ylabel("#L Clicks") ; cbar = colorbar() ; clim([0, 1])
    title("2D, rat. Ntrls=" * string(data[irat]["ntrls"]))
    cbar.ax.set_ylabel("% go right")

    ###################################
    ## 2D MODEL PSYCHOMETRIC SURFACE ##
    ###################################
    subplot(234)
    maxbups = maximum([data[irat]["wholetrl"]["X"].wtLtot ;
                       data[irat]["wholetrl"]["X"].wtRtot])
    nclicks = 0 : 1 : maxbups
    model_decision_surf = zeros(length(nclicks), length(nclicks))
    for nr = nclicks
        for nl = nclicks
            model_decision_surf[nl+1,nr+1] = predict(data[irat]["wholetrl"]["logit"],
                                                     DataFrame(wtRtot=Int(nr),wtLtot=Int(nl)))[1]
        end
    end
    imshow(model_decision_surf, origin="lower", cmap=get_cmap("RdBu"))
    xlabel("#R Clicks") ; ylabel("#L Clicks") ; cbar = colorbar(); clim([0, 1])
    title(@sprintf("2D, model. β=%0.2f, AUC(ROC)=%0.2f+/-%0.2f",
        data[irat]["wholetrl"]["logit"].model.pp.beta0[1],
        mean(auc_wholetrl), std(auc_wholetrl)))
    cbar.ax.set_ylabel("% go right")

    ########################################
    ## TIME-VARYING L/R MODEL, STIM ONSET ##
    ########################################
    subplot(232)
    res_rl = get_wts_sterr(data[irat]["stimon"]["logit_rl"], nbins, true)
    res_d  = get_wts_sterr(data[irat]["stimon"]["logit_d"], nbins, false)
    if cfg.PLOT_KERNEL_MAGNITUDE lwts = res_rl.lwts .* -1 end
    errorbar(data["xaxis_stimon"], res_rl.rwts, yerr=res_rl.rsterr, label="right weights", color="b")
    errorbar(data["xaxis_stimon"], lwts,        yerr=res_rl.lsterr, label="left weights", color="r")
    if cfg.PLOT_BUPDIFF_KERNEL
        errorbar(data["xaxis_stimon"], res_d.wts, yerr=res_d.wsterr, label="bupdiff weights", color="gray")
    end
    plot(data["xaxis_stimon"], zeros(length(data["xaxis_stimon"])), "k--", alpha=0.5)
    grid() ; legend()
    if cfg.PLOT_KERNEL_MAGNITUDE
        ylabel("logit weights magnitude");
    else
        ylabel("logit weights <-left -- right->")
    end
    # xlabel("Time relative to stimulus onset [ms]") ; ylabel("logit weights")
    if !cfg.PLOT_KERNEL_MAGNITUDE
        ylims = ylim() ; ylim([-maximum(abs.(ylims)), maximum(abs.(ylims))])
    end
    xlim([data["xaxis_stimon"][1]-5 , data["xaxis_stimon"][end]+5])
    title(L"Stim onset $w_{t}$, β=" * @sprintf("%0.2f ", res_rl.bias) *
        @sprintf("AUC(ROC)=%0.2f+/-%0.2f", mean(auc_on_rl), std(auc_on_rl)))

    #########################################
    ## TIME-VARYING L/R MODEL, STIM OFFSET ##
    #########################################
    subplot(235)
    res_rl = get_wts_sterr(data[irat]["stimoff"]["logit_rl"], nbins, true)
    res_d  = get_wts_sterr(data[irat]["stimoff"]["logit_d"], nbins, false)
    if cfg.PLOT_KERNEL_MAGNITUDE lwts = res_rl.lwts .* -1 end
    errorbar(data["xaxis_stimoff"], res_rl.rwts, yerr=res_rl.rsterr, label="right weights", color="b")
    errorbar(data["xaxis_stimoff"], lwts,        yerr=res_rl.lsterr, label="left weights", color="r")
    if cfg.PLOT_BUPDIFF_KERNEL
        errorbar(data["xaxis_stimoff"], res_d.wts, yerr=res_d.wsterr, label="bupdiff weights", color="gray")
    end
    plot(data["xaxis_stimoff"], zeros(length(data["xaxis_stimoff"])), "k--", alpha=0.5)
    grid() ; legend()
    if cfg.PLOT_KERNEL_MAGNITUDE
        ylabel("logit weights magnitude");
    else
        ylabel("logit weights <-left -- right->")
    end
    xlabel("Time relative to stimulus offset/onset [ms]")
    if !cfg.PLOT_KERNEL_MAGNITUDE
        ylims = ylim() ; ylim([-maximum(abs.(ylims)), maximum(abs.(ylims))])
    end
    xlim([data["xaxis_stimoff"][1]-5 , data["xaxis_stimoff"][end]+5])
    title(L"Stim offset $w_{t}$, β=" * @sprintf("%0.2f ", res_rl.bias) *
        @sprintf("AUC(ROC)=%0.2f+/-%0.2f", mean(auc_off_rl), std(auc_off_rl)))

    ############################################################################
    ## Generate rate 1D psychometric function (click diff), and model as well ##
    ############################################################################
    subplot(233)
    bd = data[irat]["wholetrl"]["X"].wtRtot - data[irat]["wholetrl"]["X"].wtLtot
    gr = data[irat]["wholetrl"]["X"].gr
    df = DataFrame(gr=data[irat]["wholetrl"]["X"].gr, bd=bd)
    logit_bd = glm(@formula(gr ~ bd), df, Binomial(), LogitLink())
    maxbd = maximum(abs.(bd))
    xaxis = -maxbd : 1 : maxbd
    propgr = zeros(length(xaxis))
    for i = 1 : length(propgr)
        matching_trials = bd .== xaxis[i]
        propgr[i] = length(findall(gr[matching_trials] .== 1))/length(gr[matching_trials])
    end
    plot(xaxis, predict(logit_bd, DataFrame(bd=xaxis)), "k", label="model")
    plot(xaxis, propgr, "k", alpha=0.5, label="rat")
    grid() ; legend(); xlabel("Click difference [#R-#L]") ; ylabel("% go right")
    title(@sprintf("1D rat & model. AUC(ROC)=%0.2f+/-%0.2f", mean(auc_wholetrlbd), std(auc_wholetrlbd)))

    ###################################
    ## Performance stacked bar plots ##
    ###################################
    subplot(236)
    corr_right   = sum(Bool.(data[irat]["wholetrl"]["X"].cr) .& Bool.(data[irat]["wholetrl"]["X"].hh))
    corr_left    = sum(Bool.(data[irat]["wholetrl"]["X"].cl) .& Bool.(data[irat]["wholetrl"]["X"].hh))
    incorr_right = sum(Bool.(data[irat]["wholetrl"]["X"].cr) .& .!Bool.(data[irat]["wholetrl"]["X"].hh))
    incorr_left  = sum(Bool.(data[irat]["wholetrl"]["X"].cl) .& .!Bool.(data[irat]["wholetrl"]["X"].hh))
    # lapse_right  = sum(data[irat]["violtrls"]["cr"])
    # lapse_left   = sum(data[irat]["violtrls"]["cl"])

    p1 = bar(1:2, height=[corr_right, corr_left])
    p2 = bar(1:2, height=[incorr_right, incorr_left], bottom=[corr_right, corr_left], tick_label=["right", "left"])
    # p3 = bar(1:2, height=[lapse_right, lapse_left], bottom=[corr_right+incorr_right, corr_left+incorr_left],
    # legend((p1[1],p2[1],p3[1]), ["corr", "incorr", "viol"])
    legend((p1[1],p2[1]), ["corr", "incorr"])

    tight_layout()
    if cfg.SAVE_FIGS
        savefig(exportpath * "/" * data[irat]["fname"][1:end-4] * ".png", dpi=150);
    end
end

if cfg.SAVE_DATA
    filename1 = cfg.EXPORTPATH_DATA * "logitFits_allrats_" * cfg.TITLE * "_" *
               string(data["cfg"].STIM_WINDOW_LEN) * "msLim_" *
               string(data["cfg"].MSPERSEG)        * "msBin_" *
               string(data["cfg"].MSOVERLAP) * "msOverlap.jld2"
    ## Save dataframes for next step in logit analysis
    println("Saving: " * string(filename1))
    save(filename1, "data", data)
end
