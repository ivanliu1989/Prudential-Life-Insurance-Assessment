setwd('/Users/ivanliu/Downloads/Prudential-Life-Insurance-Assessment')
setwd('C:/Users/iliu2/Documents/Prudential-Life-Insurance-Assessment-master')
library(readr)
library(xgboost)
library(Metrics)
library(Hmisc)
library(caret)
# library(mlbench)
rm(list=ls());gc()
load('data/fin_train_test_validation_prod.RData')
# load('data/fin_train_test_validation_xgb_meta.RData')

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
cv <- 10
folds <- createFolds(train$Response, k = cv, list = FALSE,)
dropitems <- c('Id','Response', paste0('TSNE_', 1:3), 'kmeans_all', 'Gender_Speci_feat')
feature.names <- names(train)[!names(train) %in% dropitems] 
sc <- preProcess(train[,feature.names],method = c('center', 'scale'))
train_sc <- cbind(Id = train$Id, predict(sc, train[,feature.names]), Response = train$Response)
# train_sc <- train

### Setup Results Table ###
results <- as.data.frame(matrix(rep(0,11*cv), cv))
names(results) <- c('cv_num', 'kappa', 'optim_kappa', 'fixed_kappa', '1st_cut', '2nd_cut', 
                    '3rd_cut', '4th_cut', '5th_cut', '6th_cut', '7th_cut')

### Start Training ###
for(i in 1:cv){
  f <- folds==i
  val <- train_sc[f,c(feature.names, 'Response')]
  tra <- train_sc[!f,c(feature.names, 'Response')]
   
  fitControl <- trainControl(method = "none"
                             # ,classProbs = TRUE
                             )
  fitGrid <-  expand.grid()
  cls <- train(Response ~ ., data = tra,
               method = "blassoAveraged", # awtan, blassoAveraged, bridge, foba, glmboost, leapSeq, ranger
               trControl = fitControl
               # ,tuneGrid = fitGrid,
               # verbose = TRUE
               )
  
  ### Make predictions
  validPreds <- predict(clf, val)
  kappa = evalerror_2(preds = validPreds, labels = val[,'Response'])
  ### Find optimal cutoff
  optCuts = optim(seq(1.5, 7.5, by = 1), evalerror_2, preds = validPreds, labels = val[,'Response'], 
                  method = 'Nelder-Mead', control = list(maxit = 900000, trace = TRUE, REPORT = 500))
  validPredsOptim = as.numeric(Hmisc::cut2(validPreds, c(-Inf, optCuts$par, Inf))); table(validPredsOptim)
  optimal_kappa = evalerror_2(preds = validPredsOptim, labels = val[,'Response'])
  fix_cut <- c(2.6121,	3.3566,	4.1097,	5.0359,	5.5267,	6.4481,	6.7450)
  
  validPredsFix = as.numeric(Hmisc::cut2(validPreds, c(-Inf, fix_cut, Inf)));
  fix_kappa = evalerror_2(preds = validPredsFix, labels = val[,'Response'])
  
  results[i,1:11] <- c(paste0('CV_', i), -kappa, -optimal_kappa, -fix_kappa, optCuts$par)
  View(results)
}