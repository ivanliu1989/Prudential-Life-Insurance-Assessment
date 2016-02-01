setwd('/Users/ivanliu/Downloads/Prudential-Life-Insurance-Assessment')
library(readr)
library(xgboost)
library(Metrics)
library(Hmisc)
rm(list=ls());gc()
# load('data/fin_train_test_stack_feat.RData')
load('data/fin_train_test_stack_feat_2.RData')

evalerror = function(preds, dtrain) {
    labels <- getinfo(dtrain, "label")
    err <- ScoreQuadraticWeightedKappa(as.numeric(labels),as.numeric(round(preds)))
    return(list(metric = "kappa", value = err))
}
evalerror_2 = function(x = seq(1.5, 7.5, by = 1), preds, labels) {
    cuts = c(min(preds), x[1], x[2], x[3], x[4], x[5], x[6], x[7], max(preds))
    preds = as.numeric(Hmisc::cut2(preds, cuts))
    err = Metrics::ScoreQuadraticWeightedKappa(as.numeric(labels), preds, 1, 8)
    return(-err)
}

set.seed(23)
dropitems <- c('Id','Response', 'XGB_SOFTMAX_MLOG')
feature.names <- names(train_2nd)[!names(train_2nd) %in% dropitems]
### Num stack features
dtrain        <- xgb.DMatrix(data=data.matrix(train_2nd[,feature.names]),label=train_2nd$Response) 
dtrain_a      <- xgb.DMatrix(data=data.matrix(train_a_2nd[,feature.names]),label=train_a_2nd$Response) 
dtrain_b      <- xgb.DMatrix(data=data.matrix(train_b_2nd[,feature.names]),label=train_b_2nd$Response) 
dtest         <- xgb.DMatrix(data=data.matrix(test[,feature.names]),label=test$Response)
watchlist     <- list(val=dtrain,train=dtrain)
watchlist_ab  <- list(val=dtrain_b,train=dtrain_a)
watchlist_ba  <- list(val=dtrain_a,train=dtrain_b)

clf <- xgb.train(data                = dtrain, # dtrain_a, dtrain_b, dtrain
                 nrounds             = 200, 
                 early.stop.round    = 200,
                 watchlist           = watchlist, # watchlist_ab, watchlist_ba, watchlist
                 feval               = evalerror,
                 # eval_metric         = 'rmse',
                 maximize            = T,
                 verbose             = 1,
                 objective           = "reg:linear", 
                 booster             = "gbtree",
                 eta                 = 0.035,
                 max_depth           = 6,
                 min_child_weight    = 3,
                 subsample           = 0.9,
                 colsample           = 0.67,
                 print.every.n       = 10
)

### Ensemble
# pred_b_rmse <- predict(clf, data.matrix(train_b_2nd[,feature.names]))
# pred_a_rmse <- predict(clf, data.matrix(train_a_2nd[,feature.names]))
# pred_b_kap <- predict(clf, data.matrix(train_b_2nd[,feature.names]))
# pred_a_kap <- predict(clf, data.matrix(train_a_2nd[,feature.names]))
validPreds <- (c(pred_a_rmse,pred_b_rmse) + c(pred_a_kap,pred_b_kap))/2

test_kap <- predict(clf, data.matrix(test[,feature.names]))
test_rmse <- predict(clf, data.matrix(test[,feature.names]))
testPreds <- (test_kap + test_rmse)/2

### Find optimal cutoff
library(mlr)
optCuts = optim(seq(1.5, 7.5, by = 1), evalerror_2, preds = validPreds, labels = train_2nd$Response, 
                method = 'Nelder-Mead', control = list(maxit = 30000, trace = TRUE, REPORT = 500))
validPredsOptim = as.numeric(Hmisc::cut2(validPreds, c(-Inf, optCuts$par, Inf))); table(validPredsOptim)
evalerror_2(preds = validPredsOptim, labels = train_2nd$Response)

PredsOptim = as.numeric(Hmisc::cut2(testPreds, c(-Inf, optCuts$par, Inf))); table(PredsOptim)

submission <- data.frame(Id=test$Id, Response = PredsOptim)
write_csv(submission, "submission_xgb_stack_20160201_1.csv")
