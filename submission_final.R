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
library(h2o)
localH2O <- h2o.init(ip = 'localhost', port = 54321, max_mem_size = '12g')
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
        max_iterations = 1000, beta_epsilon = 1e-4, solver = "L_BFGS", #IRLSM  L_BFGS
        standardize = F, family = 'gaussian', link = 'identity', alpha = 0.1, # 1 lasso 0 ridge
        lambda = 0, lambda_search = T, nlambda = 55, #lambda_min_ratio = 1e-08,
        intercept = T
    )


validPreds <- as.data.frame(h2o.predict(object = fit, newdata = validation_df))
# fix_cut <- c(2.951759, 3.653780, 4.402781, 4.911808, 5.543988, 6.135754, 6.716891)
fix_cut <- c(2.3617,	3.5307,	4.3668,	4.6543,	5.3891,	6.2634,	6.8177)
validPredsFix = as.numeric(Hmisc::cut2(validPreds[,1], c(-Inf, fix_cut, Inf)));
submission_glm <- data.frame(Id = test$Id, Response = validPredsFix)
write.csv(submission_glm, file = 'blending/submit_glm.csv', row.names = F)

### H2O DL
library(h2o)
localH2O <- h2o.init(ip = 'localhost', port = 54321, max_mem_size = '12g')
dropitems <- c('Id','Response')
independent <- names(train)[!names(train) %in% dropitems] 
dependent <- "Response"
colnames(train) <- c('Id', paste0('var_', 1:length(independent)), dependent)
colnames(test) <- c('Id', paste0('var_', 1:length(independent)))
independent <- paste0('var_', 1:length(independent))

train_df          <- as.h2o(localH2O, train)
validation_df     <- as.h2o(localH2O, test)

for(i in 21:50){
    set.seed(i*66)
    cat(paste0('Start training Deep Learning...', i))
    fit <-
        h2o.deeplearning(
            y = dependent, x = independent, training_frame = train_df, overwrite_with_best_model = T, #autoencoder
            use_all_factor_levels = T, activation = "RectifierWithDropout",#TanhWithDropout "RectifierWithDropout"
            hidden = c(256,128), epochs = 18, train_samples_per_iteration = -2, adaptive_rate = T, rho = 0.99,  #c(300,150,75)
            epsilon = 1e-6, rate = 0.035, rate_decay = 0.9, momentum_start = 0.9, momentum_stable = 0.99,
            nesterov_accelerated_gradient = T, input_dropout_ratio = 0.5, hidden_dropout_ratios = c(0.5,0.5), 
            l1 = 1e-5, l2 = 3e-5, loss = 'MeanSquare', classification_stop = 0.01,
            diagnostics = T, variable_importances = F, fast_mode = F, ignore_const_cols = T,
            force_load_balance = T, single_node_mode = T, shuffle_training_data = T
        )
    validPreds <- as.data.frame(h2o.predict(object = fit, newdata = validation_df))
    # fix_cut <- c(2.951759, 3.653780, 4.402781, 4.911808, 5.543988, 6.135754, 6.716891)
    fix_cut <- c(2.4798,	3.0854,	4.1037,	4.8560,	5.5441,	6.2637,	6.9075)
    validPredsFix = as.numeric(Hmisc::cut2(validPreds[,1], c(-Inf, fix_cut, Inf)));
    submission_glm <- data.frame(Id = test$Id, Response = validPredsFix)
    write.csv(submission_glm, file = paste0('blending/submit_nnets_',i,'.csv'), row.names = F)
}


### ordinal regression
library(VGAM)
load('../final_regression_pca.RData')
# Logistics regression Using Cumulative Logits
fit <- vglm(as.factor(Response) ~ . , family=cumulative(parallel=TRUE), data=train[,-1]) # Logistics regression Using Cumulative Logits
p1_r <- predict(fit, type = "response", test)
p1_t <- max.col(p1_r)



# dummy
cate.features <- c("Product_Info_1", "Product_Info_3", "Product_Info_5", "Product_Info_6", "Product_Info_7", 
                   "Employment_Info_2", "Employment_Info_3", "Employment_Info_5", 
                   "InsuredInfo_1", "InsuredInfo_2", "InsuredInfo_3", "InsuredInfo_4", "InsuredInfo_5", "InsuredInfo_6", "InsuredInfo_7", 
                   "Insurance_History_1", "Insurance_History_2", "Insurance_History_3", "Insurance_History_4", "Insurance_History_7", 
                   "Insurance_History_8", "Insurance_History_9", 
                   "Family_Hist_1", "Medical_History_2", 
                   "Medical_History_3", "Medical_History_4", "Medical_History_5", "Medical_History_6", "Medical_History_7", 
                   "Medical_History_8", "Medical_History_9", "Medical_History_11", "Medical_History_12",
                   "Medical_History_13", "Medical_History_14", "Medical_History_16", "Medical_History_17", "Medical_History_18", 
                   "Medical_History_19", "Medical_History_20", "Medical_History_21", "Medical_History_22", "Medical_History_23", 
                   "Medical_History_25", "Medical_History_26", "Medical_History_27", "Medical_History_28", "Medical_History_29", 
                   "Medical_History_30", "Medical_History_31", "Medical_History_33", "Medical_History_34", "Medical_History_35", 
                   "Medical_History_36", "Medical_History_37", "Medical_History_38", "Medical_History_39", "Medical_History_40", 
                   "Medical_History_41"
                   ,'Product_Info_2_cate', 'Product_Info_2_num'
)
test$Response <- 0
All_Data <- rbind(train,test) 
for(i in cate.features){
    if(mean(is.na(All_Data[,i]))==0){
        All_Data[,i] <- as.factor(All_Data[,i])
    }
}
dummies <- dummyVars(Response ~ ., data = All_Data, sep = "_", levelsOnly = FALSE, fullRank = TRUE)
All_Data_d <- cbind(as.data.frame(predict(dummies, newdata = All_Data)), Response = All_Data$Response)

library(caret)
total_dum <- All_Data_d
names(total_dum) <- c('Id', paste0('Var_', 1:(ncol(total_dum)-2)), 'Response')
pcafit <- preProcess(as.matrix(total_dum[,2:(ncol(total_dum)-1)]), method = c('center','scale','pca'), thresh = 0.99, na.remove = TRUE)
total_pca <- predict(pcafit, as.matrix(total_dum[,2:(ncol(total_dum)-1)]))
total.pca <- cbind(Id = total_dum$Id, as.data.frame(total_pca), Response = total_dum$Response)
train <- total.pca[total.pca$Response > 0,]
test <- total.pca[total.pca$Response == 0,]
save(train,test, file='../final_regression_pca.RData')


