setwd('/Users/ivanliu/Downloads/Prudential-Life-Insurance-Assessment')
library(readr)
library(xgboost)
library(Metrics)
library(Hmisc)
rm(list=ls());gc()

load('data/fin_train_test_prod.RData')

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

### Start Training ###
for(i in 1:cv){
    f <- folds==i
    dval          <- xgb.DMatrix(data=data.matrix(train_sc[f,feature.names]),label=train_sc[f,'Response'])
    dtrain        <- xgb.DMatrix(data=data.matrix(train_sc[!f,feature.names]),label=train_sc[!f,'Response']) 
    watchlist     <- list(val=dval,train=dtrain)
    
    # Feat1
    clf <- xgb.train(data = dtrain, eval_metric = 'rmse',
                     early.stop.round = 200, watchlist = watchlist, maximize = F, 
                     verbose = 1, objective = "reg:linear",  
                     booste = "gbtree", eta = 0.035, max_depth = 6, min_child_weight = 50, subsample = 0.8,
                     nrounds = 700, colsample = 0.67, print.every.n = 10 
    )
    pred_rmse <- predict(clf, dval)
    # Feat2
    clf <- xgb.train(data = dtrain, feval = evalerror,
                     early.stop.round = 200, watchlist = watchlist, maximize = T,
                     verbose = 1, objective = "reg:linear",
                     booste = "gbtree", eta = 0.035, max_depth = 6, min_child_weight = 50, subsample = 0.8,
                     nrounds = 700, colsample = 0.67, print.every.n = 10 
    )
    pred_kappa <- predict(clf, dval)
    # Leafs
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
    validPreds <- as.data.frame(predict(clf, dval, predleaf = TRUE))
    names(validPreds) <- c(paste0('xgb_leaf_', 1:16))
    
    
    dval          <- xgb.DMatrix(data=data.matrix(train_sc[f,feature.names]),label=train_sc[f,'Response']-1)
    dtrain        <- xgb.DMatrix(data=data.matrix(train_sc[!f,feature.names]),label=train_sc[!f,'Response']-1) 
    watchlist     <- list(val=dval,train=dtrain)
    # Feat3
    clf <- xgb.train(data = dtrain, eval_metric = 'merror',
                     early.stop.round = 200, watchlist = watchlist, maximize = F, 
                     verbose = 1, objective = "multi:softmax", 
                     booste = "gbtree", eta = 0.035, max_depth = 6, min_child_weight = 50, subsample = 0.8,
                     nrounds = 500, colsample = 0.7, print.every.n = 10 ,num_class = 8
    )
    pred_softmax <- predict(clf, dval)
    # Feat4
    clf <- xgb.train(data = dtrain, eval_metric = 'merror',
                     early.stop.round = 200, watchlist = watchlist, maximize = F, 
                     verbose = 1, objective = "multi:softprob",  
                     booste = "gbtree", eta = 0.035, max_depth = 6, min_child_weight = 50, subsample = 0.8,
                     nrounds = 500, colsample = 0.7, print.every.n = 10 ,num_class = 8
    )
    pred_softprob <- predict(clf, dval)
    # Feat5
    clf <- xgb.train(data = dtrain, eval_metric = 'mlogloss',
                     early.stop.round = 200, watchlist = watchlist, maximize = F, 
                     verbose = 1, objective = "multi:softmax", 
                     booste = "gbtree", eta = 0.035, max_depth = 6, min_child_weight = 50, subsample = 0.8,
                     nrounds = 800, colsample = 0.7, print.every.n = 10 ,num_class = 8
    )
    pred_softmax_mlog <- predict(clf, dval)
    # Feat6
    clf <- xgb.train(data = dtrain, eval_metric = 'mlogloss',
                     early.stop.round = 200, watchlist = watchlist, maximize = F, 
                     verbose = 1, objective = "multi:softprob",  
                     booste = "gbtree", eta = 0.035, max_depth = 6, min_child_weight = 50, subsample = 0.8,
                     nrounds = 500, colsample = 0.7, print.every.n = 10 ,num_class = 8
    )
    pred_softprob_mlog <- predict(clf, dval)
    
    ### Make new features
    if(i==1){
        train_2nd <- train_sc[f,]
        train_2nd$XGB_RMSE <- pred_rmse
        train_2nd$XGB_KAPPA <- pred_kappa
        train_2nd$XGB_SOFTMAX <- pred_softmax
        train_2nd$XGB_SOFTMAX_MLOG <- pred_softmax_mlog
        
        META_XGB_MULSOFT <- as.data.frame(t(matrix(pred_softprob, 8, nrow(train_2nd))))
        names(META_XGB_MULSOFT) <- paste('META_XGB_MUL_', 1:8, sep = '')
        
        META_XGB_MULSOFT_MLOG <- as.data.frame(t(matrix(pred_softprob_mlog,  8, nrow(train_2nd))))
        names(META_XGB_MULSOFT_MLOG) <- paste('META_XGB_MUL_MLOG_', 1:8, sep = '')
        
        train_2nd <- cbind(train_2nd, META_XGB_MULSOFT, META_XGB_MULSOFT_MLOG, validPreds)
        train_meta <- train_2nd
        
    }else{
        train_2nd <- train_sc[f,]
        train_2nd$XGB_RMSE <- pred_rmse
        train_2nd$XGB_KAPPA <- pred_kappa
        train_2nd$XGB_SOFTMAX <- pred_softmax
        train_2nd$XGB_SOFTMAX_MLOG <- pred_softmax_mlog
        
        META_XGB_MULSOFT <- as.data.frame(t(matrix(pred_softprob, 8, nrow(train_2nd))))
        names(META_XGB_MULSOFT) <- paste('META_XGB_MUL_', 1:8, sep = '')
        
        META_XGB_MULSOFT_MLOG <- as.data.frame(t(matrix(pred_softprob_mlog,  8, nrow(train_2nd))))
        names(META_XGB_MULSOFT_MLOG) <- paste('META_XGB_MUL_MLOG_', 1:8, sep = '')
        
        train_2nd <- cbind(train_2nd, META_XGB_MULSOFT, META_XGB_MULSOFT_MLOG, validPreds)
        train_meta <- rbind(train_meta, train_2nd)
    }
}




# For test data
dtest          <- xgb.DMatrix(data=data.matrix(test_sc[,feature.names]),label=test_sc[,'Response'])
dtrain        <- xgb.DMatrix(data=data.matrix(train_sc[,feature.names]),label=train_sc[,'Response']) 
watchlist     <- list(val=dtest,train=dtrain)
# Feat1
clf <- xgb.train(data = dtrain, eval_metric = 'rmse',
                 early.stop.round = 200, watchlist = watchlist, maximize = F, 
                 verbose = 1, objective = "reg:linear",  
                 booste = "gbtree", eta = 0.035, max_depth = 6, min_child_weight = 50, subsample = 0.8,
                 nrounds = 700, colsample = 0.67, print.every.n = 10 
)
pred_rmse <- predict(clf, dtest)
# Feat2
clf <- xgb.train(data = dtrain, feval = evalerror,
                 early.stop.round = 200, watchlist = watchlist, maximize = T,
                 verbose = 1, objective = "reg:linear",
                 booste = "gbtree", eta = 0.035, max_depth = 6, min_child_weight = 50, subsample = 0.8,
                 nrounds = 700, colsample = 0.67, print.every.n = 10 
)
pred_kappa <- predict(clf, dtest)
# Leafs
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
validPreds <- as.data.frame(predict(clf, dtest, predleaf = TRUE))
names(validPreds) <- c(paste0('xgb_leaf_', 1:16))

dtest          <- xgb.DMatrix(data=data.matrix(test_sc[,feature.names]),label=test_sc[,'Response']-1)
dtrain        <- xgb.DMatrix(data=data.matrix(train_sc[,feature.names]),label=train_sc[,'Response']-1) 
watchlist     <- list(val=dtest,train=dtrain)
# Feat3
clf <- xgb.train(data = dtrain, eval_metric = 'merror',
                 early.stop.round = 200, watchlist = watchlist, maximize = F, 
                 verbose = 1, objective = "multi:softmax", 
                 booste = "gbtree", eta = 0.035, max_depth = 6, min_child_weight = 50, subsample = 0.8,
                 nrounds = 500, colsample = 0.7, print.every.n = 10 ,num_class = 8
)
pred_softmax <- predict(clf, dtest)
# Feat4
clf <- xgb.train(data = dtrain, eval_metric = 'merror',
                 early.stop.round = 200, watchlist = watchlist, maximize = F, 
                 verbose = 1, objective = "multi:softprob",  
                 booste = "gbtree", eta = 0.035, max_depth = 6, min_child_weight = 50, subsample = 0.8,
                 nrounds = 500, colsample = 0.7, print.every.n = 10 ,num_class = 8
)
pred_softprob <- predict(clf, dtest)
# Feat5
clf <- xgb.train(data = dtrain, eval_metric = 'mlogloss',
                 early.stop.round = 200, watchlist = watchlist, maximize = F, 
                 verbose = 1, objective = "multi:softmax", 
                 booste = "gbtree", eta = 0.035, max_depth = 6, min_child_weight = 50, subsample = 0.8,
                 nrounds = 800, colsample = 0.7, print.every.n = 10 ,num_class = 8
)
pred_softmax_mlog <- predict(clf, dtest)
# Feat6
clf <- xgb.train(data = dtrain, eval_metric = 'mlogloss',
                 early.stop.round = 200, watchlist = watchlist, maximize = F, 
                 verbose = 1, objective = "multi:softprob",  
                 booste = "gbtree", eta = 0.035, max_depth = 6, min_child_weight = 50, subsample = 0.8,
                 nrounds = 500, colsample = 0.7, print.every.n = 10 ,num_class = 8
)
pred_softprob_mlog <- predict(clf, dtest)

### Make predictions
train_2nd <- test_sc
train_2nd$XGB_RMSE <- pred_rmse
train_2nd$XGB_KAPPA <- pred_kappa
train_2nd$XGB_SOFTMAX <- pred_softmax
train_2nd$XGB_SOFTMAX_MLOG <- pred_softmax_mlog

META_XGB_MULSOFT <- as.data.frame(t(matrix(pred_softprob, 8, nrow(train_2nd))))
names(META_XGB_MULSOFT) <- paste('META_XGB_MUL_', 1:8, sep = '')

META_XGB_MULSOFT_MLOG <- as.data.frame(t(matrix(pred_softprob_mlog,  8, nrow(train_2nd))))
names(META_XGB_MULSOFT_MLOG) <- paste('META_XGB_MUL_MLOG_', 1:8, sep = '')

train_2nd <- cbind(train_2nd, META_XGB_MULSOFT, META_XGB_MULSOFT_MLOG, validPreds)
test_meta <- train_2nd


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

save(train, test, file = 'data/xgb_meta_leaf_20150210.RData')




