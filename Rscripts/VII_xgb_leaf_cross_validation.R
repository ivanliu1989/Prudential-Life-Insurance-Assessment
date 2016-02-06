setwd('/Users/ivanliu/Downloads/Prudential-Life-Insurance-Assessment')
# setwd('C:/Users/iliu2/Documents/Prudential-Life-Insurance-Assessment-master')
library(readr)
library(xgboost)
library(Metrics)
library(Hmisc)
library(caret)
# library(mlbench)
rm(list=ls());gc()
# load('data/fin_train_test_validation_prod.RData')
load('data/V_train_test_valid_xgb_meta_NEW.RData')

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
folds <- createFolds(as.factor(train$Response), k = cv, list = FALSE,)
dropitems <- c('Id','Response', paste0('xgb_meta_', 1:8))
feature.names <- names(train)[!names(train) %in% dropitems] 
train_sc <- train
test_sc <- test

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
                   nrounds             = 16, 
                   early.stop.round    = 200,
                   watchlist           = watchlist,
                   feval               = evalerror,
                   # eval_metric         = 'rmse',
                   maximize            = TRUE,
                   objective           = "reg:linear",
                   booster             = "gbtree",
                   eta                 = 0.6,
                   max_depth           = 6,
                   min_child_weight    = 200,
                   subsample           = 0.8,
                   colsample           = 0.67,
                   print.every.n       = 1
  )
  
  ### Make predictions
  validPreds <- as.data.frame(predict(clf, dval, predleaf = TRUE))
  names(validPreds) <- c(paste0('xgb_leaf_', 1:16))
  train_sub <- cbind(train_sc[f,-ncol(train_sc)], validPreds, Response = train_sc[f,'Response'])
  if(i==1){
    train_meta <- train_sub
  }else{
    train_meta <- rbind(train_meta, train_sub)
  }
}

dtest          <- xgb.DMatrix(data=data.matrix(test_sc[,feature.names]),label=test_sc[,'Response']-1)
dtrain        <- xgb.DMatrix(data=data.matrix(train_sc[,feature.names]),label=train_sc[,'Response']-1) 
watchlist     <- list(val=dtest,train=dtrain)

clf <- xgb.train(data                = dtrain,
                 nrounds             = 16, 
                 early.stop.round    = 200,
                 watchlist           = watchlist,
                 feval               = evalerror,
                 maximize            = TRUE,
                 objective           = "reg:linear",
                 booster             = "gbtree",
                 eta                 = 0.6,
                 max_depth           = 6,
                 min_child_weight    = 200,
                 subsample           = 0.8,
                 colsample           = 0.67,
                 print.every.n       = 1
)

### Make predictions
validPreds <- as.data.frame(predict(clf, dtest, predleaf = TRUE))
names(validPreds) <- c(paste0('xgb_leaf_', 1:16))
test_meta <- cbind(test_sc[,-ncol(test_sc)], validPreds, Response = test_sc[,'Response'])

######################################
# 4. Split/Output Data #####
############################
total_new <- rbind(train_meta, test_meta)
for (col in paste0('xgb_leaf_',1:16)){
    total_new[,col] <- as.factor(total_new[,col])
}
dummies <- dummyVars(Response ~ ., data = total_new[,c(paste0("xgb_leaf_", 1:16), 'Response')], 
                     sep = "_", levelsOnly = FALSE, fullRank = TRUE)
total1 <- as.data.frame(predict(dummies, newdata = total_new))
total_new <- cbind(total_new[,-c(ncol(total_new))], total1, Response=total_new$Response)

train <- total_new[total_new$Response != 0, ]
test <- total_new[total_new$Response == 0, ]

library(caret)
set.seed(1989)
# No validation
# inTraining <- createDataPartition(train$Response, p = .5, list = FALSE)
# train_a <- train[-inTraining,]
# train_b <- train[inTraining,]
# dim(train_b); dim(train_a); dim(test); dim(train)
save(train, test, file = 'data/VII_train_test_xgb_leaf.RData')
