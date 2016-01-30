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
    # x = c(1.619689, 3.413080, 4.206752, 4.805708, 5.610160, 6.232827, 6.686749)
    labels <- getinfo(dtrain, "label")
    cuts = c(min(preds), x[1], x[2], x[3], x[4], x[5], x[6], x[7], max(preds))
    # preds = as.numeric(cut(preds,8))
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
# 3. Model strategies #####  GOTO ii_modeling.R
###########################
# 1. split the training data into two parts: A and B. 
# Train on part A and then predict on part B. Train on part B and then predict on part A.
# Combine the predictions on A and B. 
# Use optim to get cutpoints based on the true training labels and your predictions
set.seed(19890624)
feature.names <- names(train)[2:ncol(train)-1]
dval       <- xgb.DMatrix(data=data.matrix(validation[,feature.names]),label=validation$Response) # validation_20, validation_10
dtrain     <- xgb.DMatrix(data=data.matrix(train[,feature.names]),label=train$Response) # train_20, train_10
dtrain_a   <- xgb.DMatrix(data=data.matrix(train_a[,feature.names]),label=train_a$Response) # train_20, train_10
dtrain_b   <- xgb.DMatrix(data=data.matrix(train_b[,feature.names]),label=train_b$Response) # train_20, train_10
dtest      <- xgb.DMatrix(data=data.matrix(test[,feature.names]),label=test$Response) # train_20, train_10
watchlist  <- list(val=dval,train=dtrain)

cat("running xgboost...\n")
clf <- xgb.train(data                = dtrain, 
                 nrounds             = 1650, 
                 early.stop.round    = 2000,
                 watchlist           = watchlist,
                 # feval               = evalerror,
                 eval_metric         = 'rmse',
                 maximize            = F,
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

# just for keeping track of how things went...
# run prediction on training set so we can add the value to our output filename
validPreds <- predict(clf, data.matrix(validation_20[,feature.names])) # validation_20, validation_10
validScore <- ScoreQuadraticWeightedKappa(round(validPreds),validation_20$Response) # validation_20, validation_10
evalerror_2(preds = validPreds, labels = validation_20$Response) 

# Find optimal cutoff
library(mlr)
optCuts = optim(seq(1.5, 7.5, by = 1), evalerror_2, preds = validPreds, labels = validation_20$Response, 
                method = 'Nelder-Mead', control = list(maxit = 30000, trace = TRUE, REPORT = 500))
optCuts
validPredsOptim = as.numeric(Hmisc::cut2(validPreds, c(-Inf, optCuts$par, Inf)))
table(validPredsOptim)
evalerror_2(preds = validPredsOptim, labels = validation_20$Response) 


# 2. Staked generalization
# fit train_a and predict on train_b
# fit train_b and predict on train_a
# fit entire train and predict on test
# fit 2nd stack by train on entire train + predictions and predict on test

# 3. Stacking results (methods/objective functions)

# 4. optim cut-offs after all the stacking, blending and other tricks
# 5. Bayesian Optimization

