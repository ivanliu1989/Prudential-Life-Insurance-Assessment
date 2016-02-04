setwd('/Users/ivanliu/Downloads/Prudential-Life-Insurance-Assessment')
library(readr)
library(xgboost)
library(Metrics)
library(Hmisc)
library(caret)
# library(mlbench)
rm(list=ls());gc()
load('data/fin_train_test_2.RData')

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
set.seed(23)
dropitems <- c('Id','Response', paste0('TSNE_', 1:3), 'kmeans_all', 'Gender_Speci_feat')
feature.names <- names(train)[!names(train) %in% dropitems] 
sc <- preProcess(train[,feature.names],method = c('center', 'scale'))
train_sc <- cbind(Id = train$Id, predict(sc, train[,feature.names]), Response = train$Response)
test_sc <- cbind(Id = test$Id, predict(sc, test[,feature.names]), Response = test$Response)

### Start Training ###
dtest         <- xgb.DMatrix(data=data.matrix(test_sc[,feature.names]),label=test_sc[,'Response'])
dtrain        <- xgb.DMatrix(data=data.matrix(train_sc[,feature.names]),label=train_sc[,'Response'])
watchlist     <- list(val=dtrain,train=dtrain)

clf <- xgb.train(data                = dtrain,
                 nrounds             = 800, 
                 early.stop.round    = 200,
                 watchlist           = watchlist,
                 feval               = evalerror,
                 maximize            = TRUE,
                 objective           = "reg:linear",
                 booster             = "gbtree",
                 eta                 = 0.035,
                 max_depth           = 6,
                 min_child_weight    = 3,
                 subsample           = 0.8,
                 colsample           = 0.67,
                 print.every.n       = 10
)

clf2 <- xgb.train(data                = dtrain,
                 nrounds             = 800, 
                 early.stop.round    = 200,
                 watchlist           = watchlist,
                 eval_metric         = 'rmse',
                 maximize            = FALSE,
                 objective           = "reg:linear",
                 booster             = "gbtree",
                 eta                 = 0.035,
                 max_depth           = 6,
                 min_child_weight    = 3,
                 subsample           = 0.8,
                 colsample           = 0.67,
                 print.every.n       = 10
)

### Make predictions
pred_kappa <- predict(clf, dtest)
pred_rmse <- predict(clf2, dtest)
preds <- (pred_kappa + pred_rmse)/2

optCuts = c(2.5671,	3.4588,	3.8821,	5.0254,	5.5961,	6.1620,	6.9018)
PredsOptim = as.numeric(Hmisc::cut2(preds, c(-Inf, optCuts, Inf))); table(PredsOptim)

submission <- data.frame(Id=test$Id, Response = PredsOptim)
write_csv(submission, "submission_single_xgb_new_feat_20160204.csv")
