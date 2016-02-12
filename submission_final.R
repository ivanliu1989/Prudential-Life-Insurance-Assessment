setwd('/Users/ivanliu/Downloads/Prudential-Life-Insurance-Assessment')
library(readr)
library(xgboost)
library(Metrics)
library(Hmisc)
library(caret)
rm(list=ls());gc()
load('data/xgb_meta_leaf_20150211_dummy.RData')

### Evaluation Func ###
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

### Split Data ###
set.seed(1989)
cv <- 10
folds <- createFolds(as.factor(train$Response), k = cv, list = FALSE)
dropitems <- c('Id','Response')
feature.names <- names(train)[!names(train) %in% dropitems] 
train_sc <- train
test_sc <- test


dval          <- xgb.DMatrix(data=data.matrix(test_sc[,feature.names]),label=test_sc[,'Response'])
dtrain        <- xgb.DMatrix(data=data.matrix(train_sc[,feature.names]),label=train_sc[,'Response']) 
watchlist     <- list(val=dval,train=dtrain)

clf <- xgb.train(data                = dtrain,
                 nrounds             = 300, 
                 early.stop.round    = 800,
                 watchlist           = watchlist,
                 feval               = evalerror,
                 # eval_metric         = 'rmse',
                 maximize            = TRUE,
                 objective           = "reg:linear",
                 booster             = "gbtree",
                 eta                 = 0.035,
                 max_depth           = 6,
                 min_child_weight    = 50,
                 subsample           = 0.8,
                 colsample           = 0.67,
                 print.every.n       = 10
)

### Make predictions
validPreds <- predict(clf, dval)
### Find optimal cutoff
fix_cut <- c(2.951759, 3.653780, 4.402781, 4.911808, 5.543988, 6.135754, 6.716891)
validPredsFix = as.numeric(Hmisc::cut2(validPreds, c(-Inf, fix_cut, Inf)));

submission_kappa <- data.frame(Id = test$Id, Response = validPredsFix)

clf <- xgb.train(data                = dtrain,
                 nrounds             = 300, 
                 early.stop.round    = 800,
                 watchlist           = watchlist,
                 # feval               = evalerror,
                 eval_metric         = 'rmse',
                 maximize            = FALSE,
                 objective           = "reg:linear",
                 booster             = "gbtree",
                 eta                 = 0.035,
                 max_depth           = 6,
                 min_child_weight    = 50,
                 subsample           = 0.8,
                 colsample           = 0.67,
                 print.every.n       = 10
)

### Make predictions
validPreds <- predict(clf, dval)
### Find optimal cutoff
fix_cut <- c(2.951759, 3.653780, 4.402781, 4.911808, 5.543988, 6.135754, 6.716891)
validPredsFix = as.numeric(Hmisc::cut2(validPreds, c(-Inf, fix_cut, Inf)));

submission_rmse <- data.frame(Id = test$Id, Response = validPredsFix)






