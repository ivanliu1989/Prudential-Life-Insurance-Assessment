setwd('/Users/ivanliu/Downloads/Prudential-Life-Insurance-Assessment')
library(readr)
library(xgboost)
library(Metrics)
library(Hmisc)
rm(list=ls());gc()
# [Tune] Result: eta=0.277; colsample_bytree=0.954; subsample=0.813 : SQWK.test.mean=0.602
####################################################################################################
# FUNCTION / VARIABLE DECLARTIONS
####################################################################################################

# evaluation function that we'll use for "feval" in xgb.train...
# evalerror <- function(preds, dtrain) {
#     labels <- getinfo(dtrain, "label")
#     err <- ScoreQuadraticWeightedKappa(as.numeric(labels),as.numeric(round(preds)))
#     return(list(metric = "kappa", value = err))
# }

evalerror = function(preds, dtrain) {
    x = seq(1.5, 7.5, by = 1)
    # x = c(1.619689, 3.413080, 4.206752, 4.805708, 5.610160, 6.232827, 6.686749)
    labels <- getinfo(dtrain, "label")
    cuts = c(min(preds), x[1], x[2], x[3], x[4], x[5], x[6], x[7], max(preds))
    # preds = as.numeric(cut(preds,8))
    as.numeric(Hmisc::cut2(preds, cuts))
    err = Metrics::ScoreQuadraticWeightedKappa(as.numeric(labels), preds, 1, 8)
    return(list(metric = "kappa", value = err))
}

evalerror_2 = function(x = seq(1.5, 7.5, by = 1), preds, labels) {
    cuts = c(min(preds), x[1], x[2], x[3], x[4], x[5], x[6], x[7], max(preds))
    preds = as.numeric(Hmisc::cut2(preds, cuts))
    err = Metrics::ScoreQuadraticWeightedKappa(as.numeric(labels), preds, 1, 8)
    return(-err)
}

####################################################################################################
# MAINLINE
####################################################################################################
set.seed(23)

cat("read train and test data...\n")
# load("data/cleaned_datasets.RData")
# load("data/cleaned_datasets_imputed.RData")
load("data/cleaned_datasets_no_encoded.RData")

feature.names <- names(train)[2:132] #132 ncol(train)-1
# remove.names <- c(grep("Medical_History_10",feature.names),grep("Medical_History_24",feature.names))
# feature.names <- feature.names[-remove.names]
# response values are in the range [1:8] ... make it [0:7] for xgb softmax....
# train_20$Response = train_20$Response - 1 # train_20, train_10

cat("create dval/dtrain/watchlist...\n")
dval       <- xgb.DMatrix(data=data.matrix(validation_20[,feature.names]),label=validation_20$Response) # validation_20, validation_10
dtrain     <- xgb.DMatrix(data=data.matrix(train_20[,feature.names]),label=train_20$Response) # train_20, train_10
watchlist  <- list(val=dval,train=dtrain)

cat("running xgboost...\n")
clf <- xgb.train(data                = dtrain, 
                 nrounds             = 1650, 
                 early.stop.round    = 2000,
                 watchlist           = watchlist,
                 # feval               = evalerror,
                 eval_metric         = 'rmse',
                 maximize            = F,
                 verbose             = 1,
                 objective           = "reg:linear",
                 booster             = "gbtree",
                 eta                 = 0.015,
                 # gamma               = 0.05,
                 max_depth           = 7,
                 min_child_weight    = 3,
                 subsample           = 0.7,
                 colsample           = 0.7,
                 print.every.n       = 10
                 # num_class           = 8
)

# just for keeping track of how things went...
# run prediction on training set so we can add the value to our output filename
validPreds <- predict(clf, data.matrix(validation_20[,feature.names])) # validation_20, validation_10
validScore <- ScoreQuadraticWeightedKappa(round(validPreds),validation_20$Response) # validation_20, validation_10
evalerror_2(preds = validPreds, labels = validation_20$Response) 

# Find optimal cutoff
library(mlr)
optCuts = optim(seq(1.5, 7.5, by = 1), evalerror_2, preds = validPreds, labels = validation_20$Response, 
                method = 'Nelder-Mead', control = list(maxit = 30000, trace = TRUE, REPORT = 500))
optCuts
validPredsOptim = as.numeric(Hmisc::cut2(validPreds, c(-Inf, optCuts$par, Inf)))
table(validPredsOptim)
evalerror_2(preds = validPredsOptim, labels = validation_20$Response) 
# cleaned_datasets_imputed.RData - 0.5959
# cleaned_datasets_no_encoded.RData - 0.5969
# cleaned_datasets.RData - 0.5967 | 0.6047 (no tsne) | 0.61219
# optimal cutoff: 
# 1. 0.6562269 min_child_weight=240, eta=0.05, nrounds=700, max_depth=6, subsample=1, colsample=0.67
# 2. 0.6601624 min_child_weight=50, eta=0.015, nrounds=3000, max_depth=6, subsample=0.7, colsample=0.7 ~1650 ITERATIONS

outFileName <- paste("z0.00000 - ",validScore,
                     " - ",clf$bestScore,
                     " - xgb - kappa - softmax",
                     " - ",myBooster,
                     " - ",myValSetPCT,
                     " - ",myEta,
                     " - ",myGamma,
                     " - ",myMaxDepth,
                     " - ",mySubsample,
                     " - ",myColSampleByTree,
                     " - ",myMinChildWeight,
                     " - ",myNRounds,
                     " - ",myEarlyStopRound,
                     " - ",clf$bestInd,".csv",sep = "")

cat("\ngenerate submission...\n")
submission <- data.frame(Id=test$Id)
submission$Response <- as.integer(round(predict(clf, data.matrix(test[,feature.names]))))

# we predicted in the range of [0:7] based on softmax... move back to [1:8]...
submission$Response <- submission$Response + 1

write_csv(submission, outFileName)