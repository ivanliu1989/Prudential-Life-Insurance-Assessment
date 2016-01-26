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
    labels <- getinfo(dtrain, "label")
    cuts = c(min(preds), x[1], x[2], x[3], x[4], x[5], x[6], x[7], max(preds))
    preds = as.numeric(Hmisc::cut2(preds, cuts))
    err = Metrics::ScoreQuadraticWeightedKappa(as.numeric(labels), preds, 1, 8)
    return(list(metric = "kappa", value = err))
}

evalerror2 = function(preds, labels) {
    x = seq(1.5, 7.5, by = 1)
    cuts = c(min(preds), x[1], x[2], x[3], x[4], x[5], x[6], x[7], max(preds))
    preds = as.numeric(Hmisc::cut2(preds, cuts))
    err = Metrics::ScoreQuadraticWeightedKappa(as.numeric(labels), preds, 1, 8)
    return(list(metric = "kappa", value = err))
}

####################################################################################################
# MAINLINE
####################################################################################################
set.seed(1989)

cat("read train and test data...\n")
load("data/cleaned_datasets.RData")
# load("data/cleaned_datasets_imputed.RData")
# load("data/cleaned_datasets_no_encoded.RData")

feature.names <- names(train)[2:909] #132 ncol(train)-1

# response values are in the range [1:8] ... make it [0:7] for xgb softmax....
# train_20$Response = train_20$Response - 1 # train_20, train_10

cat("create dval/dtrain/watchlist...\n")
dval       <- xgb.DMatrix(data=data.matrix(validation_20[,feature.names]),label=validation_20$Response) # validation_20, validation_10
dtrain     <- xgb.DMatrix(data=data.matrix(train_20[,feature.names]),label=train_20$Response) # train_20, train_10
watchlist  <- list(val=dval,train=dtrain)

cat("running xgboost...\n")
clf <- xgb.train(data                = dtrain, 
                 nrounds             = 100, 
                 early.stop.round    = 50,
                 watchlist           = watchlist,
                 feval               = evalerror,
                 maximize            = TRUE,
                 verbose             = 1,
                 objective           = "reg:linear"
)

# just for keeping track of how things went...
# run prediction on training set so we can add the value to our output filename
validPreds <- predict(clf, data.matrix(validation_20[,feature.names])) # validation_20, validation_10
validScore <- ScoreQuadraticWeightedKappa(round(validPreds),validation_20$Response) # validation_20, validation_10
evalerror2(validPreds, validation_20$Response) 
# cleaned_datasets_imputed.RData - 0.5959933
# cleaned_datasets_no_encoded.RData - 0.5969993
# cleaned_datasets.RData - 0.5967163 | 0.6047035 (no tsne)

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