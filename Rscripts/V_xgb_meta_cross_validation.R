setwd('/Users/ivanliu/Downloads/Prudential-Life-Insurance-Assessment')
library(readr)
library(xgboost)
library(Metrics)
library(Hmisc)
library(caret)
# library(mlbench)
rm(list=ls());gc()
load('data/fin_train_test_validation_prod.RData')

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
dropitems <- c('Id','Response', paste0('TSNE_', 1:3), 'kmeans_all', 'Gender_Speci_feat', 'Medical_History_10', 'Medical_History_24')
feature.names <- names(train)[!names(train) %in% dropitems] 
# sc <- preProcess(train[,feature.names],method = c('center', 'scale'))
# train_sc <- cbind(Id = train$Id, predict(sc, train[,feature.names]), Response = train$Response)
train_sc <- train
test_sc <- test

### Setup Results Table ###
results <- as.data.frame(matrix(rep(0,10*cv), cv))
names(results) <- c('cv_num', 'kappa', 'optim_kappa', '1st_cut', '2nd_cut', 
                    '3rd_cut', '4th_cut', '5th_cut', '6th_cut', '7th_cut')

### Start Training ###
for(i in 1:cv){
    f <- folds==i
    dval          <- xgb.DMatrix(data=data.matrix(train_sc[f,feature.names]),label=train_sc[f,'Response']-1)
    dtrain        <- xgb.DMatrix(data=data.matrix(train_sc[!f,feature.names]),label=train_sc[!f,'Response']-1) 
    watchlist     <- list(val=dval,train=dtrain)
    
    clf <- xgb.train(data                = dtrain,
                     nrounds             = 800, 
                     early.stop.round    = 100,
                     watchlist           = watchlist,
                     eval_metric         = 'mlogloss',
                     maximize            = FALSE,
                     objective           = "multi:softprob",
                     booster             = "gbtree",
                     eta                 = 0.035,
                     max_depth           = 6,
                     min_child_weight    = 3,
                     subsample           = 0.8,
                     colsample           = 0.67,
                     print.every.n       = 10,
                     num_class           = 8
    )
    
    ### Make predictions
    validPreds <- as.data.frame(t(matrix(predict(clf, dval), 8, nrow(train_sc[f,feature.names]))) )
    names(validPreds) <- c(paste0('xgb_meta_', 1:8))
    train_sub <- cbind(Id = train_sc[f,'Id'], train_sc[f,feature.names], validPreds, Response = train_sc[f,'Response'])
    if(i==1){
        train_meta <- train_sub
    }else{
        train_meta <- rbind(train_meta, train_sub)
    }
}

# For test data
dtest          <- xgb.DMatrix(data=data.matrix(test_sc[,feature.names]),label=test_sc[,'Response']-1)
dtrain        <- xgb.DMatrix(data=data.matrix(train_sc[,feature.names]),label=train_sc[,'Response']-1) 
watchlist     <- list(val=dtrain,train=dtrain)

clf <- xgb.train(data                = dtrain,
                 nrounds             = 800, 
                 early.stop.round    = 100,
                 watchlist           = watchlist,
                 eval_metric         = 'mlogloss',
                 maximize            = FALSE,
                 objective           = "multi:softprob",
                 booster             = "gbtree",
                 eta                 = 0.035,
                 max_depth           = 6,
                 min_child_weight    = 3,
                 subsample           = 0.8,
                 colsample           = 0.67,
                 print.every.n       = 10,
                 num_class           = 8
)

### Make predictions
validPreds <- as.data.frame(t(matrix(predict(clf, dtest), 8, nrow(test_sc))) )
names(validPreds) <- c(paste0('xgb_meta_', 1:8))
test_meta <- cbind(Id = test_sc[,'Id'], test_sc[,feature.names], validPreds, Response = test_sc[,'Response'])

######################################
# 4. Split/Output Data #####
############################
train <- train_meta
test <- test_meta
library(caret)
set.seed(1989)
# No validation
inTraining <- createDataPartition(train$Response, p = .5, list = FALSE)
train_a <- train[-inTraining,]
train_b <- train[inTraining,]
dim(train_b); dim(train_a); dim(test); dim(train)
save(train, train_b, train_a, test, file = 'data/V_train_test_xgb_meta_NEW.RData')

# Validation
inTraining <- createDataPartition(train$Response, p = .2, list = FALSE)
validation <- train[inTraining,]
# Train a & b
train <- train[-inTraining,]
inTraining <- createDataPartition(train$Response, p = .5, list = FALSE)
train_a <- train[-inTraining,]
train_b <- train[inTraining,]

dim(train_b); dim(train_a); dim(validation); dim(test); dim(train)
save(train, train_b, train_a, validation, test, file = 'data/V_train_test_valid_xgb_meta_NEW.RData')
