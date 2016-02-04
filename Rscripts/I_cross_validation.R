setwd('/Users/ivanliu/Downloads/Prudential-Life-Insurance-Assessment')
library(readr)
library(xgboost)
library(Metrics)
library(Hmisc)
library(caret)
# library(mlbench)
rm(list=ls());gc()
load('data/fin_train_test_validation_2.RData')

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
train <- rbind(train, validation)
cv <- 5
folds <- createFolds(train$Response, k = cv, list = FALSE,)
dropitems <- c('Id','Response', paste0('TSNE_', 1:3), 'kmeans_all', 'Gender_Speci_feat', 'Medical_History_10', 'Medical_History_24')
feature.names <- names(train)[!names(train) %in% dropitems] 
sc <- preProcess(train[,feature.names],method = c('center', 'scale'))
train_sc <- cbind(Id = train$Id, predict(sc, train[,feature.names]), Response = train$Response)
# train_sc <- train

### Setup Results Table ###
results <- as.data.frame(matrix(rep(0,10*cv), cv))
names(results) <- c('cv_num', 'kappa', 'optim_kappa', '1st_cut', '2nd_cut', 
                    '3rd_cut', '4th_cut', '5th_cut', '6th_cut', '7th_cut')

### Start Training ###
for(i in 1:cv){
  f <- folds==i
  dval          <- xgb.DMatrix(data=data.matrix(train_sc[f,feature.names]),label=train_sc[f,'Response'])
  dtrain        <- xgb.DMatrix(data=data.matrix(train_sc[!f,feature.names]),label=train_sc[!f,'Response']) 
  watchlist     <- list(val=dval,train=dtrain)
  
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
  
  ### Make predictions
  validPreds <- predict(clf, dval)
  kappa = evalerror_2(preds = validPreds, labels = train_sc[f,'Response'])
  ### Find optimal cutoff
  optCuts = optim(seq(1.5, 7.5, by = 1), evalerror_2, preds = validPreds, labels = train_sc[f,'Response'], 
                  method = 'Nelder-Mead', control = list(maxit = 900000, trace = TRUE, REPORT = 500))
  validPredsOptim = as.numeric(Hmisc::cut2(validPreds, c(-Inf, optCuts$par, Inf))); table(validPredsOptim)
  optimal_kappa = evalerror_2(preds = validPredsOptim, labels = train_sc[f,'Response'])
  
  results[i,1:10] <- c(paste0('CV_', i), -kappa, -optimal_kappa, optCuts$par)
  View(results)
}