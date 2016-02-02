setwd('/Users/ivanliu/Downloads/Prudential-Life-Insurance-Assessment')
library(readr);library(xgboost);library(Hmisc);library(caret);library(e1071);library(data.table);library("Metrics")
rm(list=ls());gc()
set.seed(888)
nbmvars = TRUE
newFeat = TRUE
pcaImpute = TRUE
factorical = TRUE
adddummy = FALSE
clusterFeat = TRUE
splitoutput = TRUE

# 1. read data
cat("read train and test data...\n")
train <- fread("data/train.csv", data.table = F)
test  <- fread("data/test.csv", data.table = F)
feature.names <- names(train)[2:(ncol(train)-1)]
feature.names.dum <- names(train)[2:ncol(train)]
inTraining <- createDataPartition(train$Response, p = .5, list = FALSE)
train_a <- train[-inTraining,]
train_b <- train[inTraining,]
train <- rbind(train_a, train_b)
dim(train_a); dim(train_b); dim(train)

# 2. nb meta data
if(nbmvars) {
    cat("NB meta data\n")
    # Medical
    feature.names1 <- grep('Medical_', feature.names, value=TRUE)
    trainMK <- train[, feature.names1]
    testMK <- test[, feature.names1]
    trainMK_a <- train_a[, feature.names1]
    trainMK_b <- train_b[, feature.names1]
    
    nb <- naiveBayes(x=trainMK, y=factor(train$Response, labels="x"), laplace = 0)
    pred <- predict(nb, newdata=testMK, type="raw")
    test <- cbind(test, NBMedprob1=pred[, 1], NBMedprob2=pred[, 2], NBMedprob3=pred[, 3], NBMedprob4=pred[, 4], NBMedprob5=pred[, 5], NBMedprob6=pred[, 6], NBMedprob7=pred[, 7], NBMedprob8=pred[, 8])
    
    nb <- naiveBayes(x=trainMK_a, y=factor(train_a$Response, labels="x"), laplace = 0)
    pred_b <- predict(nb, newdata=trainMK_b, type="raw")
    nb <- naiveBayes(x=trainMK_b, y=factor(train_b$Response, labels="x"), laplace = 0)
    pred_a <- predict(nb, newdata=trainMK_a, type="raw")
    pred <- rbind(pred_a, pred_b)
    train <- cbind(train[,-ncol(train)], NBMedprob1=pred[, 1], NBMedprob2=pred[, 2], NBMedprob3=pred[, 3], NBMedprob4=pred[, 4], NBMedprob5=pred[, 5], NBMedprob6=pred[, 6], NBMedprob7=pred[, 7], NBMedprob8=pred[, 8], Response=train$Response)
    
    # Insurance
    feature.names1 <- grep('Insur', feature.names, value=TRUE)
    trainMK <- train[, feature.names1]
    testMK <- test[, feature.names1]
    trainMK_a <- train_a[, feature.names1]
    trainMK_b <- train_b[, feature.names1]
    
    nb <- naiveBayes(x=trainMK, y=factor(train$Response, labels="x"), laplace = 0)
    pred <- predict(nb, newdata=testMK, type="raw")
    test <- cbind(test, NBInsprob1=pred[, 1], NBInsprob2=pred[, 2], NBInsprob3=pred[, 3], NBInsprob4=pred[, 4], NBInsprob5=pred[, 5], NBInsprob6=pred[, 6], NBInsprob7=pred[, 7], NBInsprob8=pred[, 8])
    
    nb <- naiveBayes(x=trainMK_a, y=factor(train_a$Response, labels="x"), laplace = 0)
    pred_b <- predict(nb, newdata=trainMK_b, type="raw")
    nb <- naiveBayes(x=trainMK_b, y=factor(train_b$Response, labels="x"), laplace = 0)
    pred_a <- predict(nb, newdata=trainMK_a, type="raw")
    pred <- rbind(pred_a, pred_b)
    train <- cbind(train[,-ncol(train)], NBInsprob1=pred[, 1], NBInsprob2=pred[, 2], NBInsprob3=pred[, 3], NBInsprob4=pred[, 4], NBInsprob5=pred[, 5], NBInsprob6=pred[, 6], NBInsprob7=pred[, 7], NBInsprob8=pred[, 8], Response=train$Response)
}

# 2.2 xgb meta data
if(nbmvars) {
    cat("XGB meta data\n")
    test$Response <- 0
    dtrain        <- xgb.DMatrix(data=data.matrix(train[,feature.names]),label=train$Response-1, missing=NaN) 
    dtrain_a      <- xgb.DMatrix(data=data.matrix(train_a[,feature.names]),label=train_a$Response-1, missing=NaN) 
    dtrain_b      <- xgb.DMatrix(data=data.matrix(train_b[,feature.names]),label=train_b$Response-1, missing=NaN) 
    dtest         <- xgb.DMatrix(data=data.matrix(test[,feature.names]),label=test$Response-1, missing=NA)
    watchlist     <- list(val=dtrain,train=dtrain)
    watchlist_ab  <- list(val=dtrain_b,train=dtrain_a)
    watchlist_ba  <- list(val=dtrain_a,train=dtrain_b)
    
    clf <- xgb.train(data = dtrain_a, eval_metric = 'mlogloss',
                     early.stop.round = 200, watchlist = watchlist_ab, maximize = F, 
                     verbose = 1, objective = "multi:softprob", 
                     booste = "gbtree", eta = 0.035, max_depth = 6, min_child_weight = 3, subsample = 0.8,
                     nrounds = 500, colsample = 0.7, print.every.n = 10 ,num_class = 8
    )
    pred_b <- as.data.frame(t(matrix(predict(clf, dtrain_b), 8, nrow(train_b)))) 
    clf <- xgb.train(data = dtrain_b, eval_metric = 'mlogloss',
                     early.stop.round = 200, watchlist = watchlist_ba, maximize = F, 
                     verbose = 1, objective = "multi:softprob",
                     booste = "gbtree", eta = 0.035, max_depth = 6, min_child_weight = 3, subsample = 0.8,
                     nrounds = 500, colsample = 0.7, print.every.n = 10 ,num_class = 8
    )
    pred_a <- as.data.frame(t(matrix(predict(clf, dtrain_a), 8, nrow(train_a))))
    pred <- rbind(pred_a, pred_b)
    train <- cbind(train[,-ncol(train)], XGBprob1=pred[, 1], XGBprob2=pred[, 2], XGBprob3=pred[, 3], XGBprob4=pred[, 4], XGBprob5=pred[, 5], XGBprob6=pred[, 6], XGBprob7=pred[, 7], XGBprob8=pred[, 8], Response=train$Response)
    
    clf <- xgb.train(data = dtrain, eval_metric = 'mlogloss',
                     early.stop.round = 200, watchlist = watchlist, maximize = F, 
                     verbose = 1, objective = "multi:softprob", 
                     booste = "gbtree", eta = 0.035, max_depth = 6, min_child_weight = 3, subsample = 0.8,
                     nrounds = 500, colsample = 0.7, print.every.n = 10 ,num_class = 8
    )
    pred <- as.data.frame(t(matrix(predict(clf, dtest), 8, nrow(test))))
    test <- cbind(test, XGBprob1=pred[, 1], XGBprob2=pred[, 2], XGBprob3=pred[, 3], XGBprob4=pred[, 4], XGBprob5=pred[, 5], XGBprob6=pred[, 6], XGBprob7=pred[, 7], XGBprob8=pred[, 8])
}
save(train, test, file = 'data/temp_train_test_.RData')
# 3. new features
if(newFeat){
    test$Response <- 0
    total <- rbind(train,test)
    # 1. % of NA / # of NA
    Cnt_NA_row <- apply(total, 1, function(x) sum(is.na(x)))
    Cnt_NA_Emp_row <- apply(total[, c('Employment_Info_1', 'Employment_Info_4', 'Employment_Info_6')], 1, function(x) sum(is.na(x)))
    Cnt_NA_Fam_row <- apply(total[, c('Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5')], 1, function(x) sum(is.na(x)))
    Cnt_NA_Medi_row <- apply(total[, c('Medical_History_1', 'Medical_History_10', 'Medical_History_15', 'Medical_History_24', 'Medical_History_32')], 1, function(x) sum(is.na(x)))
    Cnt_NA_Ins_row <- ifelse(is.na(total[, 'Insurance_History_5']), 1,0)
    
    # 2. # of Medical keywords
    medical.keywords <- c(paste('Medical_Keyword_', 1:48, sep = ''))
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
    
    total <- cbind(total[,1:(ncol(total)-1)], Cnt_NA_row = Cnt_NA_row,
                   Cnt_NA_Emp_row = Cnt_NA_Emp_row,
                   Cnt_NA_Fam_row = Cnt_NA_Fam_row,
                   Cnt_NA_Medi_row = Cnt_NA_Medi_row,
                   Cnt_NA_Ins_row = Cnt_NA_Ins_row,
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
                   Response = total$Response)
    train <- total[which(total$Response > 0),]
    test <- total[which(total$Response == 0),]
}

# 4. pca and impute
if(pcaImpute) {
    test$Response <- 0
    All_Data <- rbind(train,test) 
    na.features <- c('Employment_Info_1', 'Employment_Info_4', 'Employment_Info_6',
                     'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5',
                     'Medical_History_1', 'Medical_History_10', 'Medical_History_15', 'Medical_History_24', 'Medical_History_32')
    
    cat(sprintf("All_Data has %d rows and %d columns\n", nrow(All_Data), ncol(All_Data)))
    
    for(i in na.features){
        All_Data[is.na(All_Data[,i]),i] <- median(All_Data[,i], na.rm = T)
        # total[is.na(total[,i]),i] <- -1
    }
    sapply(names(All_Data), function(x){mean(is.na(All_Data[,x]))})
    
    train <- All_Data[which(All_Data$Response > 0),]
    test <- All_Data[which(All_Data$Response == 0),]
} 

# 5. factors
if(factorical){
    feature.names <- names(train)[2:ncol(train)-1]
    cat("Factor char fields\n")
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
    )
    for (f in cate.features) {
        levels <- unique(c(train[[f]], test[[f]]))
        cat(sprintf("factoring %s with %d levels\n", f, length(levels)))
        train[[f]] <- factor(train[[f]], levels=levels)
        test[[f]] <- factor(test[[f]], levels=levels)
    }
    
}
total <- rbind(train,test); table(total$Product_Info_2)
levels(total$Product_Info_2) <- c(17, 1, 19, 18, 16, 8, 2, 15, 7, 6, 3, 5, 14, 11, 10, 13, 12, 4, 9)
train <- total[which(total$Response > 0),]
test <- total[which(total$Response == 0),]
# save(train, test, file = 'data/temp_train_test_2.RData')

# 6. dummy
if(adddummy) {
    test$Response <- 0
    #All_Data <- rbind(train,test) 
    dummies <- dummyVars(Response ~ ., data = train[,c(1:127,ncol(train))], sep = "_", levelsOnly = FALSE, fullRank = TRUE)
    train1 <- as.data.frame(predict(dummies, newdata = train[,c(1:127,ncol(train))]))
    test_dum <- as.data.frame(predict(dummies, newdata = test[,c(1:127,ncol(train))]))
    train_dum <- cbind(train1, Response=train$Response)
    # head(train_dum[,names(table(names(train_dum))[table(names(train_dum))==2])])
    zv <- names(table(names(train_dum))[table(names(train_dum))==2])
    train_dum <- train_dum[,-which(names(train_dum) %in% zv)]
    test_dum <- test_dum[,-which(names(test_dum) %in% zv)]
}    

# 7. tsne/distance/kmeans
if(clusterFeat){
    if(file.exists('Data/viii_cluster_feats.RData')){
        load('Data/viii_cluster_feats.RData')
        test$Response <- 0
        total_new <- rbind(train,test)
        total <- cbind(total_new[,-ncol(total_new)], 
                       tsne_all, 
                       tsne_medi,
                       tsne_insur,
                       KMEANS = kmeans_all,
                       distances_all,
                       distances_medi,
                       distances_insur,
                       Response = total_new$Response)
        total$KMEANS <- as.factor(total$KMEANS)
        train <- total[which(total$Response > 0),]
        test <- total[which(total$Response == 0),]
    }else{
        test_dum$Response <- 0
        total_new <- rbind(train_dum,test_dum)
        feature.names <- names(total_new)[-c(1, ncol(total_new))]
        # 1. tsne
        library(Rtsne)
        tsne <- Rtsne(as.matrix(total_new[,feature.names]), dims = 3, perplexity=30, check_duplicates = F, pca = F, theta=0.5) #max_iter = 300, 
        embedding <- as.data.frame(tsne$Y)
        tsne_all <- embedding[,1:3]; names(tsne_all) <- c('TSNE_A1','TSNE_A2','TSNE_A3')
        # medical
        feature.names1 <- grep('Medical_', feature.names, value=TRUE)
        tsne <- Rtsne(as.matrix(total_new[,feature.names1]), dims = 3, perplexity=30, check_duplicates = F, pca = F, theta=0.5) #max_iter = 300, 
        embedding <- as.data.frame(tsne$Y)
        tsne_medi <- embedding[,1:3]; names(tsne_medi) <- c('TSNE_M1','TSNE_M2','TSNE_M3')
        # insurance
        feature.names1 <- grep('Insur', feature.names, value=TRUE)
        tsne <- Rtsne(as.matrix(total_new[,feature.names1]), dims = 3, perplexity=30, check_duplicates = F, pca = F, theta=0.5) #max_iter = 300, 
        embedding <- as.data.frame(tsne$Y)
        tsne_insur <- embedding[,1:3]; names(tsne_insur) <- c('TSNE_I1','TSNE_I2','TSNE_I3')
        
        save(tsne_all, tsne_medi, tsne_insur, file = 'data/temp_tsne.RData')
        
        # 2. kmeans
#         library(h2o) 
#         localH2O <- h2o.init(ip = 'localhost', port = 54321, max_mem_size = '12g')
#         kmeans_df <- as.h2o(localH2O, total_new[,feature.names])
#         fit <- h2o.kmeans(kmeans_df, k = 8, max_iterations = 1000, standardize = T, init = 'Random', seed = 1989) #none, PlusPlus, Furthest, Random
#         pred <- as.data.frame(h2o.predict(object = fit, newdata = kmeans_df))
#         kmeans_all <- pred[,1]; table(kmeans_all)
        fit <- kmeans(total_new[,feature.names], centers = 8, iter.max = 10000,
                      nstart = 40)
        
        # 3. correlation distance
        library(caret)
        # all
        centroids <- classDist(total_new[, feature.names], as.factor(total_new[, 'Response']), pca = T, keep = 280) 
        distances <- predict(centroids, total_new[, feature.names])
        distances <- as.data.frame(distances)
        distances_all <- distances[,-1]; names(distances_all) <- paste('DistALL', 1:8, sep = "")
        # medical
        feature.names1 <- grep('Medical_', feature.names, value=TRUE)
        centroids <- classDist(total_new[, feature.names1], as.factor(total_new[, 'Response']), pca = T, keep = 150) 
        distances <- predict(centroids, total_new[, feature.names1])
        distances <- as.data.frame(distances)
        distances_medi <- distances[,-1]; names(distances_medi) <- paste('DistMEDI', 1:8, sep = "")
        # Insurance
        feature.names1 <- grep('Insur', feature.names, value=TRUE)
        centroids <- classDist(total_new[, feature.names1], as.factor(total_new[, 'Response']), pca = T, keep = 22) 
        distances <- predict(centroids, total_new[, feature.names1])
        distances <- as.data.frame(distances)
        distances_insur <- distances[,-1]; names(distances_insur) <- paste('DistINSUR', 1:8, sep = "")
        # nzv <- nearZeroVar(total_new[, 2:(ncol(total_new)-1)], saveMetrics= TRUE)
        # nzv[nzv$nzv,][1:10,]
        
        save(tsne_all, tsne_insur, tsne_medi, kmeans_all, distances_all, distances_medi, distances_insur, file = 'Data/viii_cluster_feats.RData')
        
        # 4. Apply to raw data
        load('Data/viii_cluster_feats.RData')
        test$Response <- 0
        total_new <- rbind(train,test)
        total <- cbind(total_new[,-ncol(total_new)], 
                       tsne_all, 
                       tsne_medi,
                       tsne_insur,
                       KMEANS = kmeans_all,
                       distances_all,
                       distances_medi,
                       distances_insur,
                       Response = total_new$Response)
        total$KMEANS <- as.factor(total$KMEANS)
        train <- total[which(total$Response > 0),]
        test <- total[which(total$Response == 0),]
    }
}

# 8. Split
if(splitoutput){
    total <- rbind(train, test)
    set.seed(1989)
    # No validation
    train <- total[which(total$Response > 0),]
    test <- total[which(total$Response == 0),]
    inTraining <- createDataPartition(train$Response, p = .5, list = FALSE)
    train_a <- train[-inTraining,]
    train_b <- train[inTraining,]
    dim(train_b); dim(train_a); dim(test); dim(train)
    save(train, train_b, train_a, test, file = 'data/viii_train_test.RData')
    
    # Validation
    train <- total[which(total$Response > 0),]
    test <- total[which(total$Response == 0),]
    inTraining <- createDataPartition(train$Response, p = .2, list = FALSE)
    validation <- train[inTraining,]
    train <- train[-inTraining,]
    inTraining <- createDataPartition(train$Response, p = .5, list = FALSE)
    train_a <- train[-inTraining,]
    train_b <- train[inTraining,]
    dim(train_b); dim(train_a); dim(validation); dim(test); dim(train)
    save(train, train_b, train_a, validation, test, file = 'data/viii_train_test_validation.RData')
    
    # Dummy
    train <- total[which(total$Response > 0),]
    test <- total[which(total$Response == 0),]
    test$Response <- NA
    dummies <- dummyVars(Response ~ ., data = train[,], sep = "_", levelsOnly = FALSE, fullRank = TRUE)
    train1 <- as.data.frame(predict(dummies, newdata = train[,]))
    test <- as.data.frame(predict(dummies, newdata = test[,]))
    train <- cbind(train1, Response=train$Response)
    head(test[,names(table(names(test))[table(names(test))==2])])
    
    test$Response <- 0
    total <- rbind(train, test)
    train <- total[which(total$Response > 0),]
    test <- total[which(total$Response == 0),]
    inTraining <- createDataPartition(train$Response, p = .5, list = FALSE)
    train_a <- train[-inTraining,]
    train_b <- train[inTraining,]
    dim(train_b); dim(train_a); dim(test); dim(train)
    save(train, train_b, train_a, test, file = 'data/viii_train_test_dummy.RData')
    
    train <- total[which(total$Response > 0),]
    test <- total[which(total$Response == 0),]
    inTraining <- createDataPartition(train$Response, p = .5, list = FALSE)
    train_a <- train[-inTraining,]
    train_b <- train[inTraining,]
    dim(train_b); dim(train_a); dim(test); dim(train)
    save(train, train_b, train_a, validation, test, file = 'data/viii_train_test_validation_dummy.RData')
}

    