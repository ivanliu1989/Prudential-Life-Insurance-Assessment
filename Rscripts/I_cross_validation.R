setwd('/Users/ivanliu/Downloads/Prudential-Life-Insurance-Assessment')
# setwd('C:/Users/iliu2/Documents/Prudential-Life-Insurance-Assessment-master')
library(readr)
library(xgboost)
library(Metrics)
library(Hmisc)
library(caret)
# library(mlbench)
rm(list=ls());gc()
# load('data/fin_train_test_prod.RData')
# load('data/V_train_test_valid_xgb_meta_NEW.RData')
load('data/VII_train_test_xgb_leaf.RData')

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
folds <- createFolds(as.factor(train$Response), k = cv, list = FALSE,)
dropitems <- c('Id','Response')#, paste0('TSNE_', 1:3), 'kmeans_all', 'Gender_Speci_feat')
feature.names <- names(train)[!names(train) %in% dropitems] 
# sc <- preProcess(train[,feature.names],method = c('center', 'scale'))
# train_sc <- cbind(Id = train$Id, predict(sc, train[,feature.names]), Response = train$Response)
train_sc <- train

### Setup Results Table ###
results <- as.data.frame(matrix(rep(0,11*cv), cv))
names(results) <- c('cv_num', 'kappa', 'optim_kappa', 'fixed_kappa', '1st_cut', '2nd_cut', 
                    '3rd_cut', '4th_cut', '5th_cut', '6th_cut', '7th_cut')

### Start Training ###
for(i in 1:cv){
  f <- folds==i
  dval          <- xgb.DMatrix(data=data.matrix(train_sc[f,feature.names]),label=train_sc[f,'Response'])
  dtrain        <- xgb.DMatrix(data=data.matrix(train_sc[!f,feature.names]),label=train_sc[!f,'Response']) 
  watchlist     <- list(val=dval,train=dtrain)
  
  clf <- xgb.train(data                = dtrain,
                   nrounds             = 850, 
                   early.stop.round    = 50,
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
  
  ### Make predictions
  validPreds <- predict(clf, dval)
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
}