setwd('/Users/ivanliu/Downloads/Prudential-Life-Insurance-Assessment')
library(readr)
library(xgboost)
library(Metrics)
library(Hmisc)
rm(list=ls());gc()

##############################
          # 1. Read Data #####
          ####################
load('data/fin_train_test_validation.RData')
load('data/fin_train_test.RData')

##############################
          # 2. Eval Func #####
          ####################
evalerror = function(preds, dtrain) {
    x = seq(1.5, 7.5, by = 1)
    labels <- getinfo(dtrain, "label")
    cuts = c(min(preds), x[1], x[2], x[3], x[4], x[5], x[6], x[7], max(preds))
    as.numeric(Hmisc::cut2(preds, cuts))
    err = Metrics::ScoreQuadraticWeightedKappa(as.numeric(labels), preds, 1, 8)
    return(list(metric = "kappa", value = err))
}

evalerror_2 = function(x = seq(1.5, 7.5, by = 1), preds, labels) {
    cuts = c(min(preds), x[1], x[2], x[3], x[4], x[5], x[6], x[7], max(preds))
    preds = as.numeric(Hmisc::cut2(preds, cuts))
    err = Metrics::ScoreQuadraticWeightedKappa(as.numeric(labels), preds, 1, 8)
    return(-err)
}

#####################################
          # 3. Model strategies ##### 
          ###########################
# 1. split the training data into two parts: A and B. 
# Train on part A and then predict on part B. Train on part B and then predict on part A.
# Combine the predictions on A and B. 
# Use optim to get cutpoints based on the true training labels and your predictions
set.seed(19890624)
feature.names <- names(train)[2:ncol(train)-1]
dval          <- xgb.DMatrix(data=data.matrix(validation[,feature.names]),label=validation$Response) 
dtrain        <- xgb.DMatrix(data=data.matrix(train[,feature.names]),label=train$Response) 
dtrain_a      <- xgb.DMatrix(data=data.matrix(train_a[,feature.names]),label=train_a$Response) 
dtrain_b      <- xgb.DMatrix(data=data.matrix(train_b[,feature.names]),label=train_b$Response) 
dtest         <- xgb.DMatrix(data=data.matrix(test[,feature.names]),label=test$Response)
watchlist     <- list(val=dval,train=dtrain)
watchlist_ab  <- list(val=dtrain_b,train=dtrain_a)
watchlist_ba  <- list(val=dtrain_a,train=dtrain_b)

cat("running xgboost...\n")
clf <- xgb.train(data                = dtrain, 
                 nrounds             = 1650, 
                 early.stop.round    = 100,
                 watchlist           = watchlist,
                 # watchlist           = watchlist_ab,
                 # watchlist           = watchlist_ba,
                 # feval               = evalerror,
                 eval_metric         = 'rmse',
                 maximize            = FALSE,
                 verbose             = 1,
                 objective           = "reg:linear",
                 booster             = "gbtree",
                 eta                 = 0.015,
                 # gamma               = 0.05,
                 max_depth           = 7,
                 min_child_weight    = 3,
                 subsample           = 0.7,
                 colsample           = 0.7,
                 print.every.n       = 10
)

# Make predictions
validPreds <- predict(clf, data.matrix(validation[,feature.names])) 
validScore <- ScoreQuadraticWeightedKappa(round(validPreds),validation$Response)
evalerror_2(preds = validPreds, labels = validation$Response)  # 0.7573752
# Find optimal cutoff
library(mlr)
optCuts = optim(seq(1.5, 7.5, by = 1), evalerror_2, preds = validPreds, labels = validation$Response, 
                method = 'Nelder-Mead', control = list(maxit = 30000, trace = TRUE, REPORT = 500))
optCuts
validPredsOptim = as.numeric(Hmisc::cut2(validPreds, c(-Inf, optCuts$par, Inf))); table(validPredsOptim)
evalerror_2(preds = validPredsOptim, labels = validation$Response) # 0.8112775
# VALIDATION-TRAIN: 0.7573752/0.8112775

# 2. Staked generalization
# fit train_a and predict on train_b
# fit train_b and predict on train_a
# fit entire train and predict on test
# fit 2nd stack by train on entire train + predictions and predict on test

# 3. Stacking results (methods/objective functions)

# 4. optim cut-offs after all the stacking, blending and other tricks
# 5. Bayesian Optimization

testPreds <- predict(clf, data.matrix(test[,feature.names])) 
testPredsOptim = as.numeric(Hmisc::cut2(testPreds, c(-Inf, optCuts$par, Inf))); table(testPredsOptim)
head(as.data.frame(cbind(Id = test$Id, Response = testPredsOptim)), 20)
