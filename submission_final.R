setwd('/Users/ivanliu/Downloads/Prudential-Life-Insurance-Assessment')
library(readr)
library(xgboost)
library(Metrics)
library(Hmisc)
library(caret)
rm(list=ls());gc()
load('data/xgb_meta_leaf_20150211_dummy.RData')

### Evaluation Func ###
evalerror = function(preds, dtrain) {
    labels <- getinfo(dtrain, "label")
    err <- ScoreQuadraticWeightedKappa(as.numeric(labels),as.numeric(round(preds)))
    return(list(metric = "kappa", value = err))
}

### Split Data ###
set.seed(1989)
cv <- 10
folds <- createFolds(as.factor(train$Response), k = cv, list = FALSE)
dropitems <- c('Id','Response')
feature.names <- names(train)[!names(train) %in% dropitems] 
train_sc <- train
test_sc <- test


dval          <- xgb.DMatrix(data=data.matrix(test_sc[,feature.names]),label=test_sc[,'Response'])
dtrain        <- xgb.DMatrix(data=data.matrix(train_sc[,feature.names]),label=train_sc[,'Response']) 
watchlist     <- list(val=dval,train=dtrain)

clf <- xgb.train(data                = dtrain,
                 nrounds             = 300, 
                 early.stop.round    = 800,
                 watchlist           = watchlist,
                 feval               = evalerror,
                 # eval_metric         = 'rmse',
                 maximize            = TRUE,
                 objective           = "reg:linear",
                 booster             = "gbtree",
                 eta                 = 0.035,
                 max_depth           = 6,
                 min_child_weight    = 50,
                 subsample           = 0.8,
                 colsample           = 0.67,
                 print.every.n       = 10
)

### Make predictions
validPreds <- predict(clf, dval)
fix_cut <- c(2.951759, 3.653780, 4.402781, 4.911808, 5.543988, 6.135754, 6.716891)
validPredsFix = as.numeric(Hmisc::cut2(validPreds, c(-Inf, fix_cut, Inf)));
submission_kappa <- data.frame(Id = test$Id, Response = validPredsFix)

clf <- xgb.train(data                = dtrain,
                 nrounds             = 300, 
                 early.stop.round    = 800,
                 watchlist           = watchlist,
                 # feval               = evalerror,
                 eval_metric         = 'rmse',
                 maximize            = FALSE,
                 objective           = "reg:linear",
                 booster             = "gbtree",
                 eta                 = 0.035,
                 max_depth           = 6,
                 min_child_weight    = 50,
                 subsample           = 0.8,
                 colsample           = 0.67,
                 print.every.n       = 10
)

### Make predictions
validPreds <- predict(clf, dval)
fix_cut <- c(2.951759, 3.653780, 4.402781, 4.911808, 5.543988, 6.135754, 6.716891)
validPredsFix = as.numeric(Hmisc::cut2(validPreds, c(-Inf, fix_cut, Inf)));
submission_rmse <- data.frame(Id = test$Id, Response = validPredsFix)


write.csv(submission_kappa, file = 'blending/submit_kappa.csv', row.names = F)
write.csv(submission_rmse, file = 'blending/submit_rmse.csv', row.names = F)



### H2O

dropitems <- c('Id','Response')
independent <- names(train)[!names(train) %in% dropitems] 
dependent <- "Response"
colnames(train) <- c('Id', paste0('var_', 1:length(independent)), dependent)
colnames(test) <- c('Id', paste0('var_', 1:length(independent)))
independent <- paste0('var_', 1:length(independent))

train_df          <- as.h2o(localH2O, train)
validation_df     <- as.h2o(localH2O, test)

fit <-
    h2o.glm(
        y = dependent, x = independent, training_frame = train_df, #train_df | total_df
        max_iterations = 100, beta_epsilon = 1e-4, solver = "L_BFGS", #IRLSM  L_BFGS
        standardize = F, family = 'gaussian', link = 'identity', alpha = 0.1, # 1 lasso 0 ridge
        lambda = 0, lambda_search = T, nlambda = 55, #lambda_min_ratio = 1e-08,
        intercept = T
    )


validPreds <- as.data.frame(h2o.predict(object = fit, newdata = validation_df))
fix_cut <- c(2.951759, 3.653780, 4.402781, 4.911808, 5.543988, 6.135754, 6.716891)
validPredsFix = as.numeric(Hmisc::cut2(validPreds[,1], c(-Inf, fix_cut, Inf)));
submission_glm <- data.frame(Id = test$Id, Response = validPredsFix)


