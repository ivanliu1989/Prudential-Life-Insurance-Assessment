setwd('/Users/ivanliu/Downloads/Prudential-Life-Insurance-Assessment')
library(readr)
library(xgboost)
library(Metrics)
library(Hmisc)
rm(list=ls());gc()

load('data/fin_train_test_validation.RData'); test <- validation
# load('data/fin_train_test.RData')

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
dropitems <- c('Id','Response')
feature.names <- names(train)[!names(train) %in% dropitems]
### Num stack features
dtrain        <- xgb.DMatrix(data=data.matrix(train[,feature.names]),label=train$Response) 
dtrain_a      <- xgb.DMatrix(data=data.matrix(train_a[,feature.names]),label=train_a$Response) 
dtrain_b      <- xgb.DMatrix(data=data.matrix(train_b[,feature.names]),label=train_b$Response) 
dtest         <- xgb.DMatrix(data=data.matrix(test[,feature.names]),label=test$Response)
watchlist     <- list(val=dtrain,train=dtrain)
watchlist_ab  <- list(val=dtrain_b,train=dtrain_a)
watchlist_ba  <- list(val=dtrain_a,train=dtrain_b)
# Feat1
clf <- xgb.train(data = dtrain_a, eval_metric = 'rmse',
                 early.stop.round = 200, watchlist = watchlist_ab, maximize = F, 
                 verbose = 1, objective = "reg:linear",  
                 booste = "gbtree", eta = 0.035, max_depth = 6, min_child_weight = 3, subsample = 0.8,
                 nrounds = 700, colsample = 0.67, print.every.n = 10 
)
pred_b_rmse <- predict(clf, data.matrix(train_b[,feature.names]))
clf <- xgb.train(data = dtrain_b, eval_metric = 'rmse',
                 early.stop.round = 200, watchlist = watchlist_ba, maximize = F, 
                 verbose = 1, objective = "reg:linear", 
                 booste = "gbtree", eta = 0.035, max_depth = 6, min_child_weight = 3, subsample = 0.8,
                 nrounds = 700, colsample = 0.67, print.every.n = 10 
)
pred_a_rmse <- predict(clf, data.matrix(train_a[,feature.names]))
clf <- xgb.train(data = dtrain, eval_metric = 'rmse',
                 early.stop.round = 200, watchlist = watchlist, maximize = F, 
                 verbose = 1, objective = "reg:linear",  
                 booste = "gbtree", eta = 0.035, max_depth = 6, min_child_weight = 3, subsample = 0.8,
                 nrounds = 700, colsample = 0.67, print.every.n = 10 
)
pred_test_rmse <- predict(clf, data.matrix(test[,feature.names]))
# Feat2
clf <- xgb.train(data = dtrain_a, feval = evalerror,
                 early.stop.round = 200, watchlist = watchlist_ab, maximize = T,
                 verbose = 1, objective = "reg:linear",
                 booste = "gbtree", eta = 0.035, max_depth = 6, min_child_weight = 3, subsample = 0.8,
                 nrounds = 700, colsample = 0.67, print.every.n = 10 
)
pred_b_kappa <- predict(clf, data.matrix(train_b[,feature.names]))
clf <- xgb.train(data = dtrain_b, feval = evalerror,
                 early.stop.round = 200, watchlist = watchlist_ba, maximize = T,
                 verbose = 1, objective = "reg:linear", 
                 booste = "gbtree", eta = 0.035, max_depth = 6, min_child_weight = 3, subsample = 0.8,
                 nrounds = 700, colsample = 0.67, print.every.n = 10 
)
pred_a_kappa <- predict(clf, data.matrix(train_a[,feature.names]))
clf <- xgb.train(data = dtrain, feval = evalerror,
                 early.stop.round = 200, watchlist = watchlist, maximize = T,
                 verbose = 1, objective = "reg:linear",  
                 booste = "gbtree", eta = 0.035, max_depth = 6, min_child_weight = 3, subsample = 0.8,
                 nrounds = 700, colsample = 0.67, print.every.n = 10 
)
pred_test_kappa <- predict(clf, data.matrix(test[,feature.names]))

### Cate stack features
dtrain        <- xgb.DMatrix(data=data.matrix(train[,feature.names]),label=train$Response-1) 
dtrain_a      <- xgb.DMatrix(data=data.matrix(train_a[,feature.names]),label=train_a$Response-1) 
dtrain_b      <- xgb.DMatrix(data=data.matrix(train_b[,feature.names]),label=train_b$Response-1) 
dtest         <- xgb.DMatrix(data=data.matrix(test[,feature.names]),label=test$Response-1)
watchlist     <- list(val=dtrain,train=dtrain)
watchlist_ab  <- list(val=dtrain_b,train=dtrain_a)
watchlist_ba  <- list(val=dtrain_a,train=dtrain_b)
# Feat3
clf <- xgb.train(data = dtrain_a, eval_metric = 'merror',
                 early.stop.round = 200, watchlist = watchlist_ab, maximize = F, 
                 verbose = 1, objective = "multi:softmax", 
                 booste = "gbtree", eta = 0.035, max_depth = 6, min_child_weight = 3, subsample = 0.8,
                 nrounds = 500, colsample = 0.7, print.every.n = 10 ,num_class = 8
)
pred_b_softmax <- predict(clf, data.matrix(train_b[,feature.names]))
clf <- xgb.train(data = dtrain_b, eval_metric = 'merror',
                 early.stop.round = 200, watchlist = watchlist_ba, maximize = F, 
                 verbose = 1, objective = "multi:softmax",
                 booste = "gbtree", eta = 0.035, max_depth = 6, min_child_weight = 3, subsample = 0.8,
                 nrounds = 500, colsample = 0.7, print.every.n = 10 ,num_class = 8
)
pred_a_softmax <- predict(clf, data.matrix(train_a[,feature.names]))
clf <- xgb.train(data = dtrain, eval_metric = 'merror',
                 early.stop.round = 200, watchlist = watchlist, maximize = F, 
                 verbose = 1, objective = "multi:softmax",  
                 booste = "gbtree", eta = 0.035, max_depth = 6, min_child_weight = 3, subsample = 0.8,
                 nrounds = 500, colsample = 0.7, print.every.n = 10 ,num_class = 8
)
pred_test_softmax <- predict(clf, data.matrix(test[,feature.names]))
# Feat4
clf <- xgb.train(data = dtrain_a, eval_metric = 'merror',
                 early.stop.round = 200, watchlist = watchlist_ab, maximize = F, 
                 verbose = 1, objective = "multi:softprob",  
                 booste = "gbtree", eta = 0.035, max_depth = 6, min_child_weight = 3, subsample = 0.8,
                 nrounds = 500, colsample = 0.7, print.every.n = 10 ,num_class = 8
)
pred_b_softprob <- predict(clf, data.matrix(train_b[,feature.names]))
clf <- xgb.train(data = dtrain_b, eval_metric = 'merror',
                 early.stop.round = 200, watchlist = watchlist_ba, maximize = F, 
                 verbose = 1, objective = "multi:softprob",
                 booste = "gbtree", eta = 0.035, max_depth = 6, min_child_weight = 3, subsample = 0.8,
                 nrounds = 800, colsample = 0.7, print.every.n = 10 ,num_class = 8
)
pred_a_softprob <- predict(clf, data.matrix(train_a[,feature.names]))
clf <- xgb.train(data = dtrain, eval_metric = 'merror',
                 early.stop.round = 200, watchlist = watchlist, maximize = F, 
                 verbose = 1, objective = "multi:softprob",  
                 booste = "gbtree", eta = 0.035, max_depth = 6, min_child_weight = 3, subsample = 0.8,
                 nrounds = 800, colsample = 0.7, print.every.n = 10 ,num_class = 8
)
pred_test_softprob <- predict(clf, data.matrix(test[,feature.names]))

# Feat5
clf <- xgb.train(data = dtrain_a, eval_metric = 'mlogloss',
                 early.stop.round = 200, watchlist = watchlist_ab, maximize = F, 
                 verbose = 1, objective = "multi:softmax", 
                 booste = "gbtree", eta = 0.035, max_depth = 6, min_child_weight = 3, subsample = 0.8,
                 nrounds = 800, colsample = 0.7, print.every.n = 10 ,num_class = 8
)
pred_b_softmax_mlog <- predict(clf, data.matrix(train_b[,feature.names]))
clf <- xgb.train(data = dtrain_b, eval_metric = 'mlogloss',
                 early.stop.round = 200, watchlist = watchlist_ba, maximize = F, 
                 verbose = 1, objective = "multi:softmax",
                 booste = "gbtree", eta = 0.035, max_depth = 6, min_child_weight = 3, subsample = 0.8,
                 nrounds = 500, colsample = 0.7, print.every.n = 10 ,num_class = 8
)
pred_a_softmax_mlog <- predict(clf, data.matrix(train_a[,feature.names]))
clf <- xgb.train(data = dtrain, eval_metric = 'mlogloss',
                 early.stop.round = 200, watchlist = watchlist, maximize = F, 
                 verbose = 1, objective = "multi:softmax",  
                 booste = "gbtree", eta = 0.035, max_depth = 6, min_child_weight = 3, subsample = 0.8,
                 nrounds = 500, colsample = 0.7, print.every.n = 10 ,num_class = 8
)
pred_test_softmax_mlog <- predict(clf, data.matrix(test[,feature.names]))
# Feat6
clf <- xgb.train(data = dtrain_a, eval_metric = 'mlogloss',
                 early.stop.round = 200, watchlist = watchlist_ab, maximize = F, 
                 verbose = 1, objective = "multi:softprob",  
                 booste = "gbtree", eta = 0.035, max_depth = 6, min_child_weight = 3, subsample = 0.8,
                 nrounds = 500, colsample = 0.7, print.every.n = 10 ,num_class = 8
)
pred_b_softprob_mlog <- predict(clf, data.matrix(train_b[,feature.names]))
clf <- xgb.train(data = dtrain_b, eval_metric = 'mlogloss',
                 early.stop.round = 200, watchlist = watchlist_ba, maximize = F, 
                 verbose = 1, objective = "multi:softprob",
                 booste = "gbtree", eta = 0.035, max_depth = 6, min_child_weight = 3, subsample = 0.8,
                 nrounds = 500, colsample = 0.7, print.every.n = 10 ,num_class = 8
)
pred_a_softprob_mlog <- predict(clf, data.matrix(train_a[,feature.names]))
clf <- xgb.train(data = dtrain, eval_metric = 'mlogloss',
                 early.stop.round = 200, watchlist = watchlist, maximize = F, 
                 verbose = 1, objective = "multi:softprob",  
                 booste = "gbtree", eta = 0.035, max_depth = 6, min_child_weight = 3, subsample = 0.8,
                 nrounds = 500, colsample = 0.7, print.every.n = 10 ,num_class = 8
)
pred_test_softprob_mlog <- predict(clf, data.matrix(test[,feature.names]))

### Features
train_2nd <- rbind(train_a, train_b)
train_2nd$XGB_RMSE <- c(pred_a_rmse, pred_b_rmse)
train_2nd$XGB_KAPPA <- c(pred_a_kappa, pred_b_kappa)
train_2nd$XGB_SOFTMAX <- c(pred_a_softmax, pred_b_softmax)
train_2nd$XGB_SOFTMAX_MLOG <- c(pred_a_softmax_mlog, pred_b_softmax_mlog)
META_XGB_MULSOFT <- as.data.frame(rbind(t(matrix(pred_a_softprob, 8, nrow(train_a))), t(matrix(pred_b_softprob, 8, nrow(train_b)))))
names(META_XGB_MULSOFT) <- paste('META_XGB_MUL_', 1:8, sep = '')
META_XGB_MULSOFT_MLOG <- as.data.frame(rbind(t(matrix(pred_a_softprob_mlog,  8, nrow(train_a))), t(matrix(pred_b_softprob_mlog, 8, nrow(train_b)))))
names(META_XGB_MULSOFT_MLOG) <- paste('META_XGB_MUL_MLOG_', 1:8, sep = '')
train_2nd <- cbind(train_2nd, META_XGB_MULSOFT, META_XGB_MULSOFT_MLOG)

test$XGB_RMSE <- pred_test_rmse
test$XGB_KAPPA <- pred_test_kappa
test$XGB_SOFTMAX <- pred_test_softmax
test$XGB_SOFTMAX_MLOG <- pred_test_softmax_mlog
META_XGB_MULSOFT <- as.data.frame(t(matrix(pred_test_softprob, 8, nrow(test))))
names(META_XGB_MULSOFT) <- paste('META_XGB_MUL_', 1:8, sep = '')
META_XGB_MULSOFT_MLOG <- as.data.frame(t(matrix(pred_test_softprob_mlog, 8, nrow(test))))
names(META_XGB_MULSOFT_MLOG) <- paste('META_XGB_MUL_MLOG_', 1:8, sep = '')
test <- cbind(test, META_XGB_MULSOFT, META_XGB_MULSOFT_MLOG)

train_a_2nd <- train_a
train_a_2nd$XGB_RMSE <- pred_a_rmse
train_a_2nd$XGB_KAPPA <- pred_a_kappa
train_a_2nd$XGB_SOFTMAX <- pred_a_softmax
train_a_2nd$XGB_SOFTMAX_MLOG <- pred_a_softmax_mlog
META_XGB_MULSOFT <- as.data.frame(t(matrix(pred_a_softprob, 8, nrow(train_a))))
names(META_XGB_MULSOFT) <- paste('META_XGB_MUL_', 1:8, sep = '')
META_XGB_MULSOFT_MLOG <- as.data.frame(t(matrix(pred_a_softprob_mlog, 8, nrow(train_a))))
names(META_XGB_MULSOFT_MLOG) <- paste('META_XGB_MUL_MLOG_', 1:8, sep = '')
train_a_2nd <- cbind(train_a_2nd, META_XGB_MULSOFT, META_XGB_MULSOFT_MLOG)

train_b_2nd <- train_b
train_b_2nd$XGB_RMSE <- pred_b_rmse
train_b_2nd$XGB_KAPPA <- pred_b_kappa
train_b_2nd$XGB_SOFTMAX <- pred_b_softmax
train_b_2nd$XGB_SOFTMAX_MLOG <- pred_b_softmax_mlog
META_XGB_MULSOFT <- as.data.frame(t(matrix(pred_b_softprob, 8, nrow(train_b))))
names(META_XGB_MULSOFT) <- paste('META_XGB_MUL_', 1:8, sep = '')
META_XGB_MULSOFT_MLOG <- as.data.frame(t(matrix(pred_b_softprob_mlog, 8, nrow(train_b))))
names(META_XGB_MULSOFT_MLOG) <- paste('META_XGB_MUL_MLOG_', 1:8, sep = '')
train_b_2nd <- cbind(train_b_2nd, META_XGB_MULSOFT, META_XGB_MULSOFT_MLOG)

save(train_2nd, train_a_2nd, train_b_2nd, test, file= 'data/fin_train_test_stack_feat_comp.RData')

ScoreQuadraticWeightedKappa(round(pred_b_rmse),train_b$Response)
ScoreQuadraticWeightedKappa(round(pred_a_rmse),train_a$Response)
ScoreQuadraticWeightedKappa(round(pred_b_kappa),train_b$Response)
ScoreQuadraticWeightedKappa(round(pred_a_kappa),train_a$Response)
ScoreQuadraticWeightedKappa(round(pred_b_softmax),train_b$Response)
ScoreQuadraticWeightedKappa(round(pred_a_softmax),train_a$Response)
ScoreQuadraticWeightedKappa(round(pred_b_softmax_mlog),train_b$Response)
ScoreQuadraticWeightedKappa(round(pred_a_softmax_mlog),train_a$Response)