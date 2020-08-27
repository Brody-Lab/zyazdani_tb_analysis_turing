# zyazdani_tb_analysis_turing

Turing_Rat_Analysis2.jl is the most recent version of my analysis

utils.jl, data_preprocessing.jl, fit_logits_kfoldcv.jl,  summarize_rats.jl, and summarize_all_rats.jl are Jorge's original code 

    An extension of the analysis done in the Brunton et al experiment. The original experiment had a rat place their nose in a center port and listen to auditory clicks to their left and right. At the end of the trial the rat moves to the side which it heard more clicks. One question is how does the rat make the decision that it does? Does the rat base its decision on early clicks or late clicks, or perhaps a burst of clicks? With regards to the question of early or late clicks, the previous experimenters found that no time bin in particular stood out as being more important in the rat’s decision making. Further analysis by Jorge Yanar confirmed the same result. We sought to check this once more by implementing a slightly different approach to the analysis.
    Instead of using a drift diffusion framework or the generalized linear model (GLM.jl package) which Jorge used, we used the Turing.jl package to do Bayesian logistic regression. Using Jorge’s code as inspiration I rewrote all new code.
    We chose Bayesian logistic regression because it’s simpler than the drift diffusion-based model and we wanted to know whether there were primacy/recency effects. This is why, similar to Jorge’s analysis we divided the trial into smaller bins. 
    We fitted three models all of which used normally distributed click differences as priors and a Bernoulli distributed posterior distribution. The “element” model and the “matrix model” had 10 regressor weights each representing a different time bin that the rat heard clicks. The only difference between the element model and the matrix model was that the matrix model formatted the 10 w's into a multivariate normal distribution (using 0 as the mean and the identity matrix as a covariance matrix) and matrix multiplied the training data with the weights. The element model, on the other hand, used 10 separate normal distributions (with 0 as the mean and 1 as the variance) and summed the element wise multiplication of the training data and the weights. The 3rd model, the “LVR” model, treated clicks from the right and left separately using the same 10 time bins but resulting in a total of 20 weights (10 from the left and 10 from the right). It utilized two multivariate normal distributions similar to the matrix model. 
    Next step would be to extend this model to a higherarchical framework, inferring the weights across rats. 



