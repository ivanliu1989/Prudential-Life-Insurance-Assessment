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

# declare these as variables: easier to reuse the script; and many are included in the output filename
myObjective       <- "reg:linear" #"multi:softmax"  # xgb parm... objective... multiclass classification
myBooster         <- "gblinear"         # xgb parm... type of booster... gbtree 
# myValSetPCT       <- 20.0              # pct of training set to hold for validation
myEta             <- 0.277             # xgb parm... smaller = more conservative 0.02
myGamma           <- 0.1              # xgb parm... bigger = more conservative 0.3
myMaxDepth        <- 6               # xgb parm... bigger = might overfit 15
mySubsample       <- 0.813              # xgb parm... 0.9 to 0.7 usually good 
myColSampleByTree <- 0.954              # xgb parm... 0.5 to 0.7 usually good
myMinChildWeight  <- 50                # xgb parm... bigger = more conservative 3
myNRounds         <- 500              # xgb parm... bigger = might overfit
myEarlyStopRound  <- 500               # xgb parm... stop learning early if no increase after this many rounds
myNThread         <- 3                # num threads to use

####################################################################################################
# MAINLINE
####################################################################################################
set.seed(1989)

cat("read train and test data...\n")
# load("data/cleaned_datasets.RData")
load("data/cleaned_datasets_imputed.RData")
# load("data/cleaned_datasets_no_encoded.RData")

feature.names <- names(train)[2:ncol(train)-1] #132

# response values are in the range [1:8] ... make it [0:7] for xgb softmax....
train_20$Response = train_20$Response - 1 # train_20, train_10

cat("create dval/dtrain/watchlist...\n")
dval       <- xgb.DMatrix(data=data.matrix(validation_20[,feature.names]),label=validation_20$Response) # validation_20, validation_10
dtrain     <- xgb.DMatrix(data=data.matrix(train_20[,feature.names]),label=train_20$Response) # train_20, train_10
watchlist  <- list(val=dval,train=dtrain)

cat("running xgboost...\n")
param <- list(   objective           = myObjective, 
                 booster             = myBooster,
                 eta                 = myEta,
                 max_depth           = myMaxDepth,
                 subsample           = mySubsample,
                 colsample_bytree    = myColSampleByTree,
                 min_child_weight    = myMinChildWeight,
                 gamma               = myGamma,
                 # num_parallel_tree   = 2
                 # alpha               = 0.0001, 
                 # lambda              = 1,
                 num_class           = 8,
                 nthread             = myNThread
)

clf <- xgb.train(params              = param, 
                 data                = dtrain, 
                 nrounds             = myNRounds, 
                 early.stop.round    = myEarlyStopRound,
                 watchlist           = watchlist,
                 feval               = evalerror,
                 maximize            = TRUE,
                 verbose             = 1
)


# just for keeping track of how things went...
# run prediction on training set so we can add the value to our output filename
validPreds <- predict(clf, data.matrix(validation_20[,feature.names])) # validation_20, validation_10
validScore <- ScoreQuadraticWeightedKappa(round(validPreds),validation_20$Response) # validation_20, validation_10
evalerror2(validPreds, validation_20$Response)

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