setwd('/Users/ivanliu/Downloads/Prudential-Life-Insurance-Assessment')
library(data.table)
rm(list=ls());gc()
# 1. load data and define data structure
cat("read train and test data...\n")
train <- fread("data/train.csv", data.table = F)
test  <- fread("data/test.csv", data.table = F)

na.features <- c('Employment_Info_1', 'Employment_Info_4', 'Employment_Info_6',
                 'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5',
                 'Medical_History_1', 'Medical_History_10', 'Medical_History_15', 'Medical_History_24', 'Medical_History_32')
cate.features <- c("Product_Info_1", "Product_Info_2", "Product_Info_3", "Product_Info_5", "Product_Info_6", "Product_Info_7", 
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
medical.keywords <- c(paste('Medical_Keyword_', 1:48, sep = ''))
test$Response <- 0
total <- rbind(train, test); dim(total)

# 1. % of NA / # of NA
Cnt_NA_row <- apply(total, 1, function(x) sum(is.na(x)))
# 2. # of Medical keywords
Cnt_Medi_Key_Word <- rowSums(total[, medical.keywords])
# 3. BMI & Age
BMI_InsAge <- total$BMI * total$Ins_Age
# 4. Split product_info_2
Product_Info_2_cate <- substr(total$Product_Info_2, 1,1)
Product_Info_2_num <- substr(total$Product_Info_2, 2,2)
# 5. Information specific to gender NewFeature1
Gender_Speci_feat <- NA
Gender_Speci_feat <- ifelse(total$Family_Hist_2>=0 & is.na(total$Family_Hist_3), 1, Gender_Speci_feat)
Gender_Speci_feat <- ifelse(total$Family_Hist_3>=0 & is.na(total$Family_Hist_2), 0, Gender_Speci_feat)
Gender_Speci_feat[is.na(Gender_Speci_feat)] <- -1
# 6. ##----Feature engineering-------------------------------------------------------------------------
#Quantile cut-off used to define custom variables 10 and 12 (step functions over BMI and BMI*Ins_Age)
tra <- total
qbmic <- 0.8
qbmic2 <- 0.9
#Hand engineered features. Found by EDA (especially added variable plots), some parameters optimized
#using cross validation. Nonlinear dependence on BMI and its interaction with age make intuitive sense.
custom_var_1 <- as.numeric(tra$Medical_History_15 < 10.0)
custom_var_1[is.na(custom_var_1)] <- 0.0 #impute these NAs with 0s, note that they were not median-imputed
custom_var_3 <- as.numeric(tra$Product_Info_4 < 0.075)
custom_var_4 <- as.numeric(tra$Product_Info_4 == 1)
custom_var_6 <- (tra$BMI + 1.0)**2.0
custom_var_7 <- (tra$BMI)**0.8
custom_var_8 <- tra$Ins_Age**8.5
custom_var_9 <- (tra$BMI*tra$Ins_Age)**2.5
BMI_cutoff <- quantile(tra$BMI, qbmic)
custom_var_10 <- as.numeric(tra$BMI > BMI_cutoff)
custom_var_11 <- (tra$BMI*tra$Product_Info_4)**0.9
ageBMI_cutoff <- quantile(tra$Ins_Age*tra$BMI, qbmic2)
custom_var_12 <- as.numeric(tra$Ins_Age*tra$BMI > ageBMI_cutoff)
custom_var_13 <- (tra$BMI*tra$Medical_Keyword_3 + 0.5)**3.0
##------------------------------------------------------------------------------------------------
load('data/i_cluster_feats_dist.RData')
total <- cbind(total[,1:(ncol(total)-1)], Cnt_NA_row = Cnt_NA_row,
               Cnt_Medi_Key_Word = Cnt_Medi_Key_Word,
               BMI_InsAge = BMI_InsAge,
               Product_Info_2_cate = Product_Info_2_cate,
               Product_Info_2_num = Product_Info_2_num,
               Gender_Speci_feat = Gender_Speci_feat, 
               custom_var_1 = custom_var_1,
               custom_var_3 = custom_var_3,
               custom_var_4 = custom_var_4,
               custom_var_6 = custom_var_6,
               custom_var_7 = custom_var_7,
               custom_var_8 = custom_var_8,
               custom_var_9 = custom_var_9,
               custom_var_10 = custom_var_10,
               custom_var_11 = custom_var_11,
               custom_var_12 = custom_var_12,
               custom_var_13 = custom_var_13,
               distances_all, 
               Response = total$Response)
train <- total[which(total$Response > 0),]
test <- total[which(total$Response == 0),]

# Impute
test$Response <- 0
All_Data <- rbind(train,test) 
cat(sprintf("All_Data has %d rows and %d columns\n", nrow(All_Data), ncol(All_Data)))
for(i in na.features){
    All_Data[is.na(All_Data[,i]),i] <- median(All_Data[,i], na.rm = T)
}
sapply(names(All_Data), function(x){mean(is.na(All_Data[,x]))})
All_Data$Product_Info_2 <- as.factor(All_Data$Product_Info_2)
All_Data$Product_Info_2_cate <- as.factor(All_Data$Product_Info_2_cate)
levels(All_Data$Product_Info_2) <- c(17, 1, 19, 18, 16, 8, 2, 15, 7, 6, 3, 5, 14, 11, 10, 13, 12, 4, 9)
levels(All_Data$Product_Info_2_cate) <- c(1:5)

train <- All_Data[which(All_Data$Response > 0),]
test <- All_Data[which(All_Data$Response == 0),]

# factor
cat("Factor char fields\n")
for (f in cate.features) {
    levels <- unique(c(train[[f]], test[[f]]))
    cat(sprintf("factoring %s with %d levels\n", f, length(levels)))
    train[[f]] <- factor(train[[f]], levels=levels)
    test[[f]] <- factor(test[[f]], levels=levels)
}
total <- rbind(train,test); table(total$Product_Info_2)
sapply(names(total), function(x){mean(is.na(total[,x]))})
train <- total[which(total$Response > 0),]
test <- total[which(total$Response == 0),]

# dummy
test$Response <- 0
#All_Data <- rbind(train,test) 
dummies <- dummyVars(Response ~ ., data = train, sep = "_", levelsOnly = FALSE, fullRank = TRUE)
train1 <- as.data.frame(predict(dummies, newdata = train))
test_dum <- as.data.frame(predict(dummies, newdata = test))
train_dum <- cbind(train1, Response=train$Response)
test_dum$Response <- 0
total_dum <- rbind(train_dum, test_dum)

library(caret)
names(total_dum) <- c('Id', paste0('Var_', 1:(ncol(total_dum)-2)), 'Response')
pcafit <- preProcess(as.matrix(total_dum[,2:(ncol(total_dum)-1)]), method = c('center','scale','pca'), thresh = 0.99, na.remove = TRUE)
total_pca <- predict(pcafit, as.matrix(total_dum[,2:(ncol(total_dum)-1)]))
total.pca <- cbind(Id = total_dum$Id, as.data.frame(total_pca), Response = total_dum$Response)

train <- total.pca[total.pca$Response > 0,]
test <- total.pca[total.pca$Response == 0,]
save(train,test, file='data/final_regression_pca.RData')

##################
### Caret ########
##################
library(caret)
library(Metrics)
evalerror_2 = function(x = seq(1.5, 7.5, by = 1), preds, labels) {
    cuts = c(min(preds), x[1], x[2], x[3], x[4], x[5], x[6], x[7], max(preds))
    preds = as.numeric(Hmisc::cut2(preds, cuts))
    err = Metrics::ScoreQuadraticWeightedKappa(as.numeric(labels), preds, 1, 8)
    return(-err)
}
# numerical
# for (f in names(train)) {
#     train[,f] <- as.numeric(train[,f])
# }
cv <- 10
folds <- createFolds(train$Response, k = cv, list = FALSE,)
feature.names <- names(train[,2:(ncol(train)-1)])
### Setup Results Table ###
results <- as.data.frame(matrix(rep(0,11*cv), cv))
names(results) <- c('cv_num', 'kappa', 'optim_kappa', 'fixed_kappa', '1st_cut', '2nd_cut', 
                    '3rd_cut', '4th_cut', '5th_cut', '6th_cut', '7th_cut')

# ### Start Training ###
# for(i in 1:cv){
#     f <- folds==i
#     val <- train[f,c(feature.names, 'Response')]
#     tra <- train[!f,c(feature.names, 'Response')]
#     # setting 1
#     fitControl <- trainControl(method = "adaptive_cv",
#                                number = 5,
#                                repeats = 2,
#                                classProbs = FALSE,
#                                # summaryFunction = RMSE,
#                                adaptive = list(min = 8,
#                                                alpha = 0.05,
#                                                method = "gls", #gls  BT
#                                                complete = TRUE))
#     # setting 2
# #     fitControl <- trainControl(method = "cv",
# #                                number = 5,
# #                                classProbs = FALSE)
# #     gbmGrid <-  expand.grid(mtry = 18)
#     # training
#     fit <- caret::train(x = tra[,feature.names],
#                         y = as.numeric(tra$Response),
#                         method = "leapSeq",
#                         trControl = fitControl,
#                         preProcess = c("center", "scale"), # pca
#                         tuneLength = 8,
#                         metric = "RMSE", # Rsquared, Kappa
#                         maximize = FALSE
#                         # ,tuneGrid = gbmGrid
#     ) 
#     
#     ### Make predictions
#     validPreds <- predict(fit, val)
#     kappa = evalerror_2(preds = as.numeric(validPreds), labels = val[,'Response'])
#     ### Find optimal cutoff
#     optCuts = optim(seq(1.5, 7.5, by = 1), evalerror_2, preds = as.numeric(validPreds), labels = val[,'Response'], 
#                     method = 'Nelder-Mead', control = list(maxit = 900000, trace = TRUE, REPORT = 500))
#     validPredsOptim = as.numeric(Hmisc::cut2(as.numeric(validPreds), c(-Inf, optCuts$par, Inf))); table(validPredsOptim)
#     optimal_kappa = evalerror_2(preds = validPredsOptim, labels = val[,'Response'])
#     fix_cut <- c(2.6121,	3.3566,	4.1097,	5.0359,	5.5267,	6.4481,	6.7450)
#     
#     validPredsFix = as.numeric(Hmisc::cut2(as.numeric(validPreds), c(-Inf, fix_cut, Inf)));
#     fix_kappa = evalerror_2(preds = validPredsFix, labels = val[,'Response'])
#     
#     results[i,1:11] <- c(paste0('CV_', i), -kappa, -optimal_kappa, -fix_kappa, optCuts$par)
#     View(results)
# }

library(h2o)
localH2O <- h2o.init(ip = 'localhost', port = 54321, max_mem_size = '12g')
independent <- names(train)[2:(ncol(train)-1)]
dependent <- "Response"
mthd = 'GBM'
### Start Training ###
for(i in 1:cv){
    f <- folds==i
    train_df          <- as.h2o(localH2O, train[!f,])
    validation_df     <- as.h2o(localH2O, train[f,]) 
    validation        <- train[f,]
    if(mthd == 'GBM'){
        print('Start training GBM...')
        fit <- h2o.gbm(
            y = dependent, x = independent, training_frame = train_df, 
            ntrees = 800, max_depth = 6, min_rows = 2,
            learn_rate = 0.035, distribution = "gaussian", #multinomial, gaussian
            nbins_cats = 20, importance = F
        )
    }
    else if(mthd == 'DL'){
        print('Start training Deep Learning...')
        fit <-
            h2o.deeplearning(
                y = dependent, x = independent, training_frame = train_df, overwrite_with_best_model = T, #autoencoder
                use_all_factor_levels = T, activation = "RectifierWithDropout",#TanhWithDropout "RectifierWithDropout"
                hidden = c(256,128), epochs = 90, train_samples_per_iteration = -2, adaptive_rate = T, rho = 0.99,  #c(300,150,75)
                epsilon = 1e-6, rate = 0.01, rate_decay = 0.9, momentum_start = 0.9, momentum_stable = 0.99,
                nesterov_accelerated_gradient = T, input_dropout_ratio = 0.5, hidden_dropout_ratios = c(0.5,0.5), 
                l1 = 1e-5, l2 = 3e-5, loss = 'MeanSquare', classification_stop = 0.01,
                diagnostics = T, variable_importances = F, fast_mode = F, ignore_const_cols = T,
                force_load_balance = F, replicate_training_data = T, shuffle_training_data = T
            )
    }
    else if(mthd == 'RF'){
        print('Start training Random Forest...')
        fit <-
            h2o.randomForest(
                y = dependent, x = independent, training_frame = train_df, mtries = -1, 
                ntrees = 800, max_depth = 16, sample.rate = 0.632, min_rows = 1, 
                nbins = 20, nbins_cats = 1024, binomial_double_trees = T
            )
    }
    else if(mthd == 'GLM'){
        print('Start training GLM...')
        fit <-
            h2o.glm(
                y = dependent, x = independent, training_frame = train_df, #train_df | total_df
                max_iterations = 1000, beta_epsilon = 1e-1, solver = "IRLSM", #IRLSM  L_BFGS
                standardize = T, family = 'gaussian', link = 'identity', alpha = 0.5, # 1 lasso 0 ridge
                lambda = 1e-12, lambda_search = T, nlambda = 55, lambda_min_ratio = 1e-08,
                intercept = T
            )
    }
    
    print(fit)
    validPreds <- as.data.frame(h2o.predict(object = fit, newdata = validation_df))
    kappa <- evalerror_2(preds = validPreds$predict, labels = validation$Response)  
    ### Find optimal cutoff
    optCuts = optim(seq(1.5, 7.5, by = 1), evalerror_2, preds = validPreds$predict, labels = validation$Response, 
                    method = 'Nelder-Mead', control = list(maxit = 30000, trace = TRUE, REPORT = 500))
    validPredsOptim = as.numeric(Hmisc::cut2(validPreds$predict, c(-Inf, optCuts$par, Inf))); table(validPredsOptim)
    optimal_kappa <- evalerror_2(preds = validPredsOptim, labels = validation$Response)
    fix_cut <- c(2.6121, 3.3566, 4.1097, 5.0359, 5.5267, 6.4481, 6.7450)
    validPredsFix = as.numeric(Hmisc::cut2(validPreds[,1], c(-Inf, fix_cut, Inf)));
    fix_kappa = evalerror_2(preds = validPredsFix, labels = train[f,'Response'])
    
    results[i,1:11] <- c(paste0('CV_', i), -kappa, -optimal_kappa, -fix_kappa, optCuts$par)
    View(results)
}