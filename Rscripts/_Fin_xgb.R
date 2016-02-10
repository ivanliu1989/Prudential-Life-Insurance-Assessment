setwd('/Users/ivanliu/Downloads/Prudential-Life-Insurance-Assessment')
library(readr)
library(xgboost)
library(Metrics)
library(Hmisc)
library(caret)
rm(list=ls());gc()
load('data/VII_train_test_xgb_leaf.RData')

### Evaluation Func ###
evalerror = function(preds, dtrain) {
    labels <- getinfo(dtrain, "label")
    err <- ScoreQuadraticWeightedKappa(as.numeric(labels),as.numeric(round(preds)))
    return(list(metric = "kappa", value = err))
}

set.seed(1989)
dropitems <- c('Id','Response')
feature.names <- names(train)[!names(train) %in% dropitems] 
dtest         <- xgb.DMatrix(data=data.matrix(test[,feature.names]),label=test[,'Response'])
dtrain        <- xgb.DMatrix(data=data.matrix(train[,feature.names]),label=train[,'Response']) 
watchlist     <- list(val=dtest,train=dtrain)

clf <- xgb.train(data                = dtrain,
                 nrounds             = 800, 
                 early.stop.round    = 800,
                 watchlist           = watchlist,
                 feval               = evalerror,
                 # eval_metric         = 'rmse',
                 maximize            = TRUE,
                 objective           = "reg:linear",
                 booster             = "gbtree",
                 eta                 = 0.035,
                 max_depth           = 6,
                 min_child_weight    = 240,
                 subsample           = 0.8,
                 colsample           = 0.7,
                 print.every.n       = 10
)

clf2 <- xgb.train(data                = dtrain,
                 nrounds             = 800, 
                 early.stop.round    = 800,
                 watchlist           = watchlist,
                 # feval               = evalerror,
                 eval_metric         = 'rmse',
                 maximize            = FALSE,
                 objective           = "reg:linear",
                 booster             = "gbtree",
                 eta                 = 0.035,
                 max_depth           = 6,
                 min_child_weight    = 240,
                 subsample           = 0.8,
                 colsample           = 0.7,
                 print.every.n       = 10
)

### Make predictions
pred_kappa <- predict(clf, dtest)
pred_rmse <- predict(clf2, dtest)

kappa = evalerror_2(preds = validPreds, labels = train_sc[f,'Response'])
### Find optimal cutoff
optCuts = optim(seq(1.5, 7.5, by = 1), evalerror_2, preds = validPreds, labels = train_sc[f,'Response'], 
                method = 'Nelder-Mead', control = list(maxit = 900000, trace = TRUE, REPORT = 500))
validPredsOptim = as.numeric(Hmisc::cut2(validPreds, c(-Inf, optCuts$par, Inf))); table(validPredsOptim)
optimal_kappa = evalerror_2(preds = validPredsOptim, labels = train_sc[f,'Response'])
fix_cut <- c(2.6121,	3.3566,	4.1097,	5.0359,	5.5267,	6.4481,	6.7450)
# avg: 2.7215,	3.5684,	4.0434,	4.8546,	5.4606,	6.2149,	6.7898 
validPredsFix = as.numeric(Hmisc::cut2(validPreds, c(-Inf, fix_cut, Inf)));
fix_kappa = evalerror_2(preds = validPredsFix, labels = train_sc[f,'Response'])

results[i,1:11] <- c(paste0('CV_', i), -kappa, -optimal_kappa, -fix_kappa, optCuts$par)
View(results)