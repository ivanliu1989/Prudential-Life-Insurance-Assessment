setwd('/Users/ivanliu/Downloads/Prudential-Life-Insurance-Assessment')
library(Metrics)
library(Hmisc)
library(xgboost)
library(checkmate)
library(mlr) 
library(data.table)
source('mlr_xgboost_reg_func.R')

load("data/cleaned_datasets.RData")
# load("data/cleaned_datasets_imputed.RData")
# load("data/cleaned_datasets_no_encoded.RData")

##################################################################
# 1.create mlr task and convert factors to dummy features ########
##################################################################
allTask = makeRegrTask(data = train[,-1], target = "Response")
trainTask = makeRegrTask(data = train_20[,-1], target = "Response")
validTask = makeRegrTask(data = validation_20[,-1], target = "Response")
testTask = makeRegrTask(data = test[,-1], target = "Response")

##############################
# 2.create mlr learner #######
##############################
set.seed(1989)
lrn = makeLearner("regr.xgboost")
lrn$par.vals = list(
    nthread             = 5,
    nrounds             = 800,
    print.every.n       = 10,
    objective           = "reg:linear"
)
# lrn$par.vals = list(
#     #nthread             = 30,
#     nrounds             = 110,
#     print.every.n       = 5,
#     objective           = "reg:linear",
#     depth = 20,
#     colsample_bytree = 0.66,
#     min_child_weight = 3,
#     subsample = 0.71
# )
# # missing values will be imputed by their median
# lrn = makeImputeWrapper(lrn, classes = list(numeric = imputeMedian(), integer = imputeMedian()))

######################################
# 3.Create Evaluation Function #######
######################################
SQWKfun = function(x = seq(1.5, 7.5, by = 1), pred) {
    preds = pred$data$response
    true = pred$data$truth 
    cuts = c(min(preds), x[1], x[2], x[3], x[4], x[5], x[6], x[7], max(preds))
    preds = as.numeric(Hmisc::cut2(preds, cuts))
    err = Metrics::ScoreQuadraticWeightedKappa(preds, true, 1, 8)
    return(-err)
}
SQWK = makeMeasure(id = "SQWK", minimize = FALSE, properties = c("regr"), best = 1, worst = 0,
                   fun = function(task, model, pred, feats, extra.args) {
                       return(-SQWKfun(x = seq(1.5, 7.5, by = 1), pred))
                   })

###############################################
# 4.Do it in parallel with parallelMap ########
###############################################
library(parallelMap)
parallelStartSocket(3)
parallelExport("SQWK", "SQWKfun", "trainLearner.regr.xgboost", "predictLearner.regr.xgboost" , "makeRLearner.regr.xgboost")

############################
# 5.Train the Model ########
############################
# 1) Define the set of parameters you want to tune (here 'eta')
ps = makeParamSet(
    makeNumericParam("eta", lower = 0.1, upper = 0.3),
    makeNumericParam("colsample_bytree", lower = 1, upper = 2, trafo = function(x) x/2),
    makeNumericParam("subsample", lower = 1, upper = 2, trafo = function(x) x/2)
)
# 2) Use 3-fold Cross-Validation to measure improvements
rdesc = makeResampleDesc("CV", iters = 3, stratify = TRUE)
# 3) Here we use Random Search (with 10 Iterations) to find the optimal hyperparameter
ctrl =  makeTuneControlRandom(maxit = 10)
# 4) now use the learner on the training Task with the 3-fold CV to optimize your set of parameters and evaluate it with SQWK
res = tuneParams(lrn, task = trainTask, resampling = rdesc, par.set = ps, control = ctrl, measures = SQWK)
res
# 5) set the optimal hyperparameter
lrn = setHyperPars(lrn, par.vals = res$x)

# perform crossvalidation in parallel
cv = crossval(lrn, trainTask, iter = 3, measures = SQWK, show.info = TRUE)
parallelStop()
## now try to find the optimal cutpoints that maximises the SQWK measure based on the Cross-Validated predictions
optCuts = optim(seq(1.5, 7.5, by = 1), SQWKfun, pred = cv$pred)
optCuts

## predict validation using the optimal cut-points
pred = predict(tr, validTask)
preds = as.numeric(Hmisc::cut2(pred$data$response, c(-Inf, optCuts$par, Inf)))
Metrics::ScoreQuadraticWeightedKappa(preds, true, 1, 8)

## now train the model on all training data
tr = train(lrn, allTask)

## predict using the optimal cut-points 
pred = predict(tr, testTask)
preds = as.numeric(Hmisc::cut2(pred$data$response, c(-Inf, optCuts$par, Inf)))
table(preds)

###################################
# 5.create submission file ########
###################################
submission = data.frame(Id = testId)
submission$Response = as.integer(preds)
write.csv(submission, "mlr.xgboost.beatbench.csv", row.names = FALSE)
