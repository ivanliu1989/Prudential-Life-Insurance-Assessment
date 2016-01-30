setwd('/Users/ivanliu/Downloads/Prudential-Life-Insurance-Assessment')
library(readr)
library(xgboost)
library(Metrics)
library(Hmisc)
rm(list=ls());gc()

##############################
          # 1. Read Data #####
          ####################
# load('data/fin_train_test_validation.RData')
# load('data/fin_train_test.RData')
# load('data/fin_train_test_validation_onehot.RData')
load('data/fin_train_test_validation_pca.RData')
##############################
          # 2. Eval Func #####
          ####################
evalerror = function(preds, dtrain) {
    x = seq(1.5, 7.5, by = 1)
    labels <- getinfo(dtrain, "label")
    cuts = c(min(preds), x[1], x[2], x[3], x[4], x[5], x[6], x[7], max(preds))
    preds = as.numeric(Hmisc::cut2(preds, cuts))
    err = Metrics::ScoreQuadraticWeightedKappa(as.numeric(labels), preds, 1, 8)
    return(list(metric = "kappa", value = err))
    #     labels <- getinfo(dtrain, "label")
    #     err <- ScoreQuadraticWeightedKappa(as.numeric(labels),as.numeric(round(preds)))
    #     return(list(metric = "kappa", value = err))
}

evalerror_2 = function(x = seq(1.5, 7.5, by = 1), preds, labels) {
    cuts = c(min(preds), x[1], x[2], x[3], x[4], x[5], x[6], x[7], max(preds))
    preds = as.numeric(Hmisc::cut2(preds, cuts))
    err = Metrics::ScoreQuadraticWeightedKappa(as.numeric(labels), preds, 1, 8)
    return(-err)
}

#####################################
          # 3. Model strategies ##### 
          ###########################
# 1. split the training data into two parts: A and B. 
# Train on part A and then predict on part B. Train on part B and then predict on part A.
# Combine the predictions on A and B. 
# Use optim to get cutpoints based on the true training labels and your predictions
set.seed(23)
dropitems <- c('Id','Response'#'Medical_History_10','Medical_History_24',
               # 'Cnt_NA_Emp_row','Cnt_NA_Fam_row','Cnt_NA_Medi_row','Cnt_NA_Ins_row'
               # 'Gender_Speci_feat'
               # , 'TSNE_ALL_1', 'TSNE_ALL_2'
               )
feature.names <- names(train)[!names(train) %in% dropitems] 
dval          <- xgb.DMatrix(data=data.matrix(validation[,feature.names]),label=validation$Response) 
dtrain        <- xgb.DMatrix(data=data.matrix(train[,feature.names]),label=train$Response) 
dtrain_a      <- xgb.DMatrix(data=data.matrix(train_a[,feature.names]),label=train_a$Response) 
dtrain_b      <- xgb.DMatrix(data=data.matrix(train_b[,feature.names]),label=train_b$Response) 
dtest         <- xgb.DMatrix(data=data.matrix(test[,feature.names]),label=test$Response)
watchlist     <- list(val=dval,train=dtrain)
watchlist_ab  <- list(val=dtrain_b,train=dtrain_a)
watchlist_ba  <- list(val=dtrain_a,train=dtrain_b)

clf <- xgb.train(data                = dtrain, # dtrain_a, dtrain_b, dtrain, dtrain_stack
                 nrounds             = 800, 
                 early.stop.round    = 200,
                 watchlist           = watchlist, # watchlist_ab, watchlist_ba, watchlist, watchlist_stack
                 feval               = evalerror,
                 # eval_metric         = 'rmse',
                 maximize            = TRUE,
                 verbose             = 1,
                 objective           = "reg:linear",
                 booster             = "gbtree",
                 eta                 = 0.035,
                 # gamma               = 0.05,
                 max_depth           = 6,
                 min_child_weight    = 3,
                 subsample           = 0.9,
                 colsample           = 0.67,
                 print.every.n       = 10
)

### Make predictions
validPreds <- predict(clf, data.matrix(validation[,feature.names])) 
validScore <- ScoreQuadraticWeightedKappa(round(validPreds),validation$Response)
evalerror_2(preds = validPreds, labels = validation$Response)  
### Find optimal cutoff
library(mlr)
optCuts = optim(seq(1.5, 7.5, by = 1), evalerror_2, preds = validPreds, labels = validation$Response, 
                method = 'Nelder-Mead', control = list(maxit = 30000, trace = TRUE, REPORT = 500))
optCuts
validPredsOptim = as.numeric(Hmisc::cut2(validPreds, c(-Inf, optCuts$par, Inf))); table(validPredsOptim)
evalerror_2(preds = validPredsOptim, labels = validation$Response)
# VALIDATION-TRAIN: 0.6143617/0.6602325
# 0.6128556/0.6553111 tsne
# 0.6129287/0.6553886 no tsne

### Feature Importance
names <- dimnames(as.matrix(train[,feature.names]))[[2]]
importance_matrix <- xgb.importance(names, model = clf)
xgb.plot.importance(importance_matrix)

###################################
# 2. Staked generalization ########
###################################
# fit train_a and predict on train_b
pred_b <- predict(clf, data.matrix(train_b[,feature.names]))
evalerror_2(preds = pred_b, labels = train_b$Response)
# fit train_b and predict on train_a
pred_a <- predict(clf, data.matrix(train_a[,feature.names]))
evalerror_2(preds = pred_a, labels = train_a$Response)
# fit entire train and predict on test
trainPred <- c(pred_a, pred_b); trainLabel <- c(train_a$Response, train_b$Response)
optCuts = optim(seq(1.5, 7.5, by = 1), evalerror_2, preds = trainPred, labels = trainLabel, 
                method = 'Nelder-Mead', control = list(maxit = 30000, trace = TRUE, REPORT = 500))
trainPredsOptim = as.numeric(Hmisc::cut2(trainPred, c(-Inf, optCuts$par, Inf))); table(trainPredsOptim)
evalerror_2(preds = trainPredsOptim, labels = trainLabel)

# fit 2nd stack by train on entire train + predictions and predict on test
train_2nd <- rbind(train_a, train_b)
train_2nd$Stack_1 <- trainPredsOptim
feature.names.stack <- names(train_2nd)[!names(train_2nd) %in% dropitems]
dtrain_stack <- xgb.DMatrix(data=data.matrix(train_2nd[,feature.names.stack]),label=train_2nd$Response) 

validation$Stack_1 <- validPredsOptim
dval_stack <- xgb.DMatrix(data=data.matrix(validation[,feature.names.stack]),label=validation$Response) 
watchlist_stack <- list(val=dval_stack,train=dtrain_stack)

# 3. Stacking results (methods/objective functions)
### Make predictions
stackPreds <- predict(clf, data.matrix(validation[,feature.names.stack])) 
evalerror_2(preds = stackPreds, labels = validation$Response)  
### Find optimal cutoff
optCuts = optim(seq(1.5, 7.5, by = 1), evalerror_2, preds = stackPreds, labels = validation$Response, 
                method = 'Nelder-Mead', control = list(maxit = 30000, trace = TRUE, REPORT = 500))
stackPredsOptim = as.numeric(Hmisc::cut2(stackPreds, c(-Inf, optCuts$par, Inf))); table(stackPredsOptim)
evalerror_2(preds = stackPredsOptim, labels = validation$Response)
# 0.6621021 - Optimal rounding results

# 4. optim cut-offs after all the stacking, blending and other tricks
# 5. Bayesian Optimization

#############################
          # 4. Blending ##### 
          ###################
testPreds <- predict(clf, data.matrix(test[,feature.names])) 
testPredsOptim = as.numeric(Hmisc::cut2(testPreds, c(-Inf, optCuts$par, Inf))); table(testPredsOptim)
head(as.data.frame(cbind(Id = test$Id, Response_Optim = testPredsOptim, Response = round(testPreds))), 20)
