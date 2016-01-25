setwd('/Users/ivanliu/Downloads/Prudential-Life-Insurance-Assessment')
library(readr)
library(xgboost)
library(Metrics)
library(data.table)
rm(list=ls());gc()

####################
# 1. Read Data #####
####################
cat("read train and test data...\n")
train <- fread("data/train.csv", data.table = F)
test  <- fread("data/test.csv", data.table = F)

##########################
# 2. Feature Explore #####
##########################
# ID + 126 FEATURES + RESPONSE = 128 COLUMNS
summary(train)
par(mfcol = c(1,1))
# i. Product Info 1-7
table(train$Product_Info_1); table(test$Product_Info_1) # Categorical - 2
table(train$Product_Info_2); table(test$Product_Info_2) # Categorical - 19
table(train$Product_Info_3); table(test$Product_Info_3) # Categorical - 38
table(train$Product_Info_4); table(test$Product_Info_4) # Numerical (Continuous)
table(train$Product_Info_5); table(test$Product_Info_5) # Categorical - 2
table(train$Product_Info_6); table(test$Product_Info_6) # Categorical - 2
table(train$Product_Info_7); table(test$Product_Info_7) # Categorical - 3

# ii. Applicant Info 4
table(train$Ins_Age); table(test$Ins_Age) # Numerical (Continuous)
table(train$Ht); table(test$Ht) # Numerical (Continuous)
table(train$Wt); table(test$Wt) # Numerical (Continuous)
table(train$BMI); table(test$BMI) # Numerical (Continuous)

# iii. Employment Info 1-6
table(train$Employment_Info_1); table(test$Employment_Info_1) # Numerical (Continuous)
table(train$Employment_Info_2); table(test$Employment_Info_2) # Categorical - 38
table(train$Employment_Info_3); table(test$Employment_Info_3) # Categorical - 2
table(train$Employment_Info_4); table(test$Employment_Info_4) # Numerical (Continuous)
table(train$Employment_Info_5); table(test$Employment_Info_5) # Categorical - 2
table(train$Employment_Info_6); table(test$Employment_Info_6) # Numerical (Continuous)

# iv. Insured Info 1-7
table(train$InsuredInfo_1); table(test$InsuredInfo_1) # Categorical - 3
table(train$InsuredInfo_2); table(test$InsuredInfo_2) # Categorical - 2
table(train$InsuredInfo_3); table(test$InsuredInfo_3) # Categorical - 11
table(train$InsuredInfo_4); table(test$InsuredInfo_4) # Categorical - 2
table(train$InsuredInfo_5); table(test$InsuredInfo_5) # Categorical - 2
table(train$InsuredInfo_6); table(test$InsuredInfo_6) # Categorical - 2
table(train$InsuredInfo_7); table(test$InsuredInfo_7) # Categorical - 2

# v. Insurance History 1-9
table(train$Insurance_History_1); table(test$Insurance_History_1) # Categorical - 2
table(train$Insurance_History_2); table(test$Insurance_History_2) # Categorical - 3
table(train$Insurance_History_3); table(test$Insurance_History_3) # Categorical - 3
table(train$Insurance_History_4); table(test$Insurance_History_4) # Categorical - 3
table(train$Insurance_History_5); table(test$Insurance_History_5) # Numerical (Continuous)
table(train$Insurance_History_7); table(test$Insurance_History_7) # Categorical - 3
table(train$Insurance_History_8); table(test$Insurance_History_8) # Categorical - 3
table(train$Insurance_History_9); table(test$Insurance_History_9) # Categorical - 3

# vi. Family History 1-5
table(train$Family_Hist_1); table(test$Family_Hist_1) # Categorical - 3
table(train$Family_Hist_2); table(test$Family_Hist_2) # Numerical (Continuous)
table(train$Family_Hist_3); table(test$Family_Hist_3) # Numerical (Continuous)
table(train$Family_Hist_4); table(test$Family_Hist_4) # Numerical (Continuous)
table(train$Family_Hist_5); table(test$Family_Hist_5) # Numerical (Continuous)

# vii. Medical History 1-41
table(train$Medical_History_1); table(test$Medical_History_1) # Numerical (Discrete) - 240
table(train$Medical_History_2); table(test$Medical_History_2) # Categorical - 648
table(train$Medical_History_3); table(test$Medical_History_3) # Categorical - 3
table(train$Medical_History_4); table(test$Medical_History_4) # Categorical - 2
table(train$Medical_History_5); table(test$Medical_History_5) # Categorical - 3
table(train$Medical_History_6); table(test$Medical_History_6) # Categorical - 3
table(train$Medical_History_7); table(test$Medical_History_7) # Categorical - 3
table(train$Medical_History_8); table(test$Medical_History_8) # Categorical - 3
table(train$Medical_History_9); table(test$Medical_History_9) # Categorical - 3
table(train$Medical_History_10); table(test$Medical_History_10) # Numerical (Discrete) - 240
table(train$Medical_History_11); table(test$Medical_History_11) # Categorical - 3
table(train$Medical_History_12); table(test$Medical_History_12) # Categorical - 3
table(train$Medical_History_13); table(test$Medical_History_13) # Categorical - 3
table(train$Medical_History_14); table(test$Medical_History_14) # Categorical - 3
table(train$Medical_History_15); table(test$Medical_History_15) # Numerical (Discrete) - 240
table(train$Medical_History_16); table(test$Medical_History_16) # Categorical - 3
table(train$Medical_History_17); table(test$Medical_History_17) # Categorical - 3
table(train$Medical_History_18); table(test$Medical_History_18) # Categorical - 3
table(train$Medical_History_19); table(test$Medical_History_19) # Categorical - 3
table(train$Medical_History_20); table(test$Medical_History_20) # Categorical - 3
table(train$Medical_History_21); table(test$Medical_History_21) # Categorical - 3
table(train$Medical_History_22); table(test$Medical_History_22) # Categorical - 2
table(train$Medical_History_23); table(test$Medical_History_23) # Categorical - 3
table(train$Medical_History_24); table(test$Medical_History_24) # Numerical (Discrete) - 240 
table(train$Medical_History_25); table(test$Medical_History_25) # Categorical - 3
table(train$Medical_History_26); table(test$Medical_History_26) # Categorical - 3
table(train$Medical_History_27); table(test$Medical_History_27) # Categorical - 3
table(train$Medical_History_28); table(test$Medical_History_28) # Categorical - 3
table(train$Medical_History_29); table(test$Medical_History_29) # Categorical - 3
table(train$Medical_History_30); table(test$Medical_History_30) # Categorical - 3
table(train$Medical_History_31); table(test$Medical_History_31) # Categorical - 3
table(train$Medical_History_32); table(test$Medical_History_32) # Numerical (Discrete) - 240
table(train$Medical_History_33); table(test$Medical_History_33) # Categorical - 3
table(train$Medical_History_34); table(test$Medical_History_34) # Categorical - 3
table(train$Medical_History_35); table(test$Medical_History_35) # Categorical - 3
table(train$Medical_History_36); table(test$Medical_History_36) # Categorical - 3
table(train$Medical_History_37); table(test$Medical_History_37) # Categorical - 3
table(train$Medical_History_38); table(test$Medical_History_38) # Categorical - 3
table(train$Medical_History_39); table(test$Medical_History_39) # Categorical - 3
table(train$Medical_History_40); table(test$Medical_History_40) # Categorical - 3
table(train$Medical_History_41); table(test$Medical_History_41) # Categorical - 3

# viii. Medical Keyword 1-48
table(train$Medical_Keyword_1); table(test$Medical_Keyword_1) # Dummary variables - 2
table(train$Medical_Keyword_2); table(test$Medical_Keyword_2) # Dummary variables - 2
table(train$Medical_Keyword_3); table(test$Medical_Keyword_3) # Dummary variables - 2
table(train$Medical_Keyword_4); table(test$Medical_Keyword_4) # Dummary variables - 2
table(train$Medical_Keyword_5); table(test$Medical_Keyword_5) # Dummary variables - 2
table(train$Medical_Keyword_6); table(test$Medical_Keyword_6) # Dummary variables - 2
table(train$Medical_Keyword_7); table(test$Medical_Keyword_7) # Dummary variables - 2
table(train$Medical_Keyword_8); table(test$Medical_Keyword_8) # Dummary variables - 2
table(train$Medical_Keyword_9); table(test$Medical_Keyword_9) # Dummary variables - 2
table(train$Medical_Keyword_10); table(test$Medical_Keyword_10) # Dummary variables - 2
table(train$Medical_Keyword_11); table(test$Medical_Keyword_11) # Dummary variables - 2
table(train$Medical_Keyword_12); table(test$Medical_Keyword_12) # Dummary variables - 2
table(train$Medical_Keyword_13); table(test$Medical_Keyword_13) # Dummary variables - 2
table(train$Medical_Keyword_14); table(test$Medical_Keyword_14) # Dummary variables - 2
table(train$Medical_Keyword_15); table(test$Medical_Keyword_15) # Dummary variables - 2
table(train$Medical_Keyword_16); table(test$Medical_Keyword_16) # Dummary variables - 2
table(train$Medical_Keyword_17); table(test$Medical_Keyword_17) # Dummary variables - 2
table(train$Medical_Keyword_18); table(test$Medical_Keyword_18) # Dummary variables - 2
table(train$Medical_Keyword_19); table(test$Medical_Keyword_19) # Dummary variables - 2
table(train$Medical_Keyword_20); table(test$Medical_Keyword_20) # Dummary variables - 2
table(train$Medical_Keyword_21); table(test$Medical_Keyword_21) # Dummary variables - 2
table(train$Medical_Keyword_22); table(test$Medical_Keyword_22) # Dummary variables - 2
table(train$Medical_Keyword_23); table(test$Medical_Keyword_23) # Dummary variables - 2
table(train$Medical_Keyword_24); table(test$Medical_Keyword_24) # Dummary variables - 2
table(train$Medical_Keyword_25); table(test$Medical_Keyword_25) # Dummary variables - 2
table(train$Medical_Keyword_26); table(test$Medical_Keyword_26) # Dummary variables - 2
table(train$Medical_Keyword_27); table(test$Medical_Keyword_27) # Dummary variables - 2
table(train$Medical_Keyword_28); table(test$Medical_Keyword_28) # Dummary variables - 2
table(train$Medical_Keyword_29); table(test$Medical_Keyword_29) # Dummary variables - 2
table(train$Medical_Keyword_30); table(test$Medical_Keyword_30) # Dummary variables - 2
table(train$Medical_Keyword_31); table(test$Medical_Keyword_31) # Dummary variables - 2
table(train$Medical_Keyword_32); table(test$Medical_Keyword_32) # Dummary variables - 2
table(train$Medical_Keyword_33); table(test$Medical_Keyword_33) # Dummary variables - 2
table(train$Medical_Keyword_34); table(test$Medical_Keyword_34) # Dummary variables - 2
table(train$Medical_Keyword_35); table(test$Medical_Keyword_35) # Dummary variables - 2
table(train$Medical_Keyword_36); table(test$Medical_Keyword_36) # Dummary variables - 2
table(train$Medical_Keyword_37); table(test$Medical_Keyword_37) # Dummary variables - 2
table(train$Medical_Keyword_38); table(test$Medical_Keyword_38) # Dummary variables - 2
table(train$Medical_Keyword_39); table(test$Medical_Keyword_39) # Dummary variables - 2
table(train$Medical_Keyword_40); table(test$Medical_Keyword_40) # Dummary variables - 2
table(train$Medical_Keyword_41); table(test$Medical_Keyword_41) # Dummary variables - 2
table(train$Medical_Keyword_42); table(test$Medical_Keyword_42) # Dummary variables - 2
table(train$Medical_Keyword_43); table(test$Medical_Keyword_43) # Dummary variables - 2
table(train$Medical_Keyword_44); table(test$Medical_Keyword_44) # Dummary variables - 2
table(train$Medical_Keyword_45); table(test$Medical_Keyword_45) # Dummary variables - 2
table(train$Medical_Keyword_46); table(test$Medical_Keyword_46) # Dummary variables - 2
table(train$Medical_Keyword_47); table(test$Medical_Keyword_47) # Dummary variables - 2
table(train$Medical_Keyword_48); table(test$Medical_Keyword_48) # Dummary variables - 2

# ix. Response 1
table(train$Response) # Categorical - 8

################################
# 3. NA value & Imputation #####
################################
sapply(names(train), function(x){mean(is.na(train[,x]))})
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
                   "Medical_History_8", "Medical_History_9", "Medical_History_11", "Medical_History_12", #Medical_History_10
                   "Medical_History_13", "Medical_History_14", "Medical_History_16", "Medical_History_17", "Medical_History_18", 
                   "Medical_History_19", "Medical_History_20", "Medical_History_21", "Medical_History_22", "Medical_History_23", 
                   "Medical_History_25", "Medical_History_26", "Medical_History_27", "Medical_History_28", "Medical_History_29", 
                   "Medical_History_30", "Medical_History_31", "Medical_History_33", "Medical_History_34", "Medical_History_35", 
                   "Medical_History_36", "Medical_History_37", "Medical_History_38", "Medical_History_39", "Medical_History_40", "Medical_History_41")
medical.keywords <- c(paste('Medical_Keyword_', 1:48, sep = ''))

head(train[,na.features]) # all Numerical (Continuous) or Numerical (Discrete) - 240 

test$Response <- 9
total <- rbind(train, test); dim(total)

#############################
# 5. One - hot encoding #####
#############################
for(i in cate.features){
    total[,i] <- as.factor(total[,i])
}
str(total)
dummies <- data.frame(model.matrix(~.-1,total[,cate.features]))
head(dummies)

total_new <- cbind(total[,-which(names(total) %in% c(cate.features,'Response'))], dummies, Response = total$Response)

#######################
# 6. New Features #####
#######################
# i. Count of Medical Keywords
Cnt_Medi_Key_Word <- rowSums(total[, medical.keywords])

# ii. product of BMI and Ins_Age
BMI_InsAge <- total$BMI * total$Ins_Age

# iii. number of NA's
Cnt_NA_row <- apply(total, 1, function(x) sum(is.na(x)))

# iv. get rid of Medical_History_10 and Medical_History_24

#####################
# 7. Imputation #####
#####################
for(i in na.features){
    total[is.na(total[,i]),i] <- median(total[,i], na.rm = T)
}
sapply(names(total), function(x){mean(is.na(total[,x]))})

#########################################
# 8. kmeans & tsne based on sectors #####
#########################################
# i. tsne clustering
total_new <- cbind(total[,-which(names(total) %in% c(cate.features,'Response'))], dummies, Response = total$Response)
library(Rtsne)
# 1) Medical History TSNE
feature.names <- grep("Medical_History",names(total_new))
tsne <- Rtsne(as.matrix(total_new[,feature.names]), dims = 2, perplexity=30, check_duplicates = F, pca = F) # theta=0.5, max_iter = 300, 
embedding <- as.data.frame(tsne$Y)
tsne_MH <- embedding[,1:2]; names(tsne_MH) <- c('TSNE_MH_1','TSNE_MH_2')

# 2) Medical Keyword TSNE
feature.names <- grep("Medical_Keyword",names(total_new))
tsne <- Rtsne(as.matrix(total_new[,feature.names]), dims = 2, perplexity=30, check_duplicates = F, pca = F) # theta=0.5, max_iter = 300, 
embedding <- as.data.frame(tsne$Y)
tsne_MK <- embedding[,1:2]; names(tsne_MK) <- c('TSNE_MK_1','TSNE_MK_2')

# 3) Product Info TSNE
feature.names <- grep("Product_Info",names(total_new))
tsne <- Rtsne(as.matrix(total_new[,feature.names]), dims = 2, perplexity=30, check_duplicates = F, pca = F) # theta=0.5, max_iter = 300, 
embedding <- as.data.frame(tsne$Y)
tsne_PI<- embedding[,1:2]; names(tsne_PI) <- c('TSNE_PI_1','TSNE_PI_2')

# 4) Employment Info TSNE
feature.names <- grep("Employment_Info",names(total_new))
tsne <- Rtsne(as.matrix(total_new[,feature.names]), dims = 2, perplexity=30, check_duplicates = F, pca = F) # theta=0.5, max_iter = 300, 
embedding <- as.data.frame(tsne$Y)
tsne_EI <- embedding[,1:2]; names(tsne_EI) <- c('TSNE_EI_1','TSNE_EI_2')

# 5) Insured Info TSNE
feature.names <- grep("InsuredInfo_",names(total_new))
tsne <- Rtsne(as.matrix(total_new[,feature.names]), dims = 2, perplexity=30, check_duplicates = F, pca = F) # theta=0.5, max_iter = 300, 
embedding <- as.data.frame(tsne$Y)
tsne_II <- embedding[,1:2]; names(tsne_II) <- c('TSNE_II_1','TSNE_II_2')

# 6) Insurance History TSNE
feature.names <- grep("Insurance_History",names(total_new))
tsne <- Rtsne(as.matrix(total_new[,feature.names]), dims = 2, perplexity=30, check_duplicates = F, pca = F) # theta=0.5, max_iter = 300, 
embedding <- as.data.frame(tsne$Y)
tsne_IH <- embedding[,1:2]; names(tsne_IH) <- c('TSNE_IH_1','TSNE_IH_2')

# 7) Family History TSNE
feature.names <- grep("Family_Hist",names(total_new))
tsne <- Rtsne(as.matrix(total_new[,feature.names]), dims = 2, perplexity=30, check_duplicates = F, pca = F) # theta=0.5, max_iter = 300, 
embedding <- as.data.frame(tsne$Y)
tsne_FH <- embedding[,1:2]; names(tsne_FH) <- c('TSNE_FH_1','TSNE_FH_2')

# 8) User Information TSNE
feature.names <- c('Ins_Age', 'Ht', 'Wt', 'BMI')
tsne <- Rtsne(as.matrix(total_new[,feature.names]), dims = 2, perplexity=30, check_duplicates = F, pca = F) # theta=0.5, max_iter = 300, 
embedding <- as.data.frame(tsne$Y)
tsne_CI <- embedding[,1:2]; names(tsne_CI) <- c('TSNE_CI_1','TSNE_CI_2')

# 9) all tsne
feature.names <- names(total_new)[-c(1, ncol(total_new))]
tsne <- Rtsne(as.matrix(total_new[,feature.names]), dims = 2, perplexity=30, check_duplicates = F, pca = F) # theta=0.5, max_iter = 300, 
embedding <- as.data.frame(tsne$Y)
tsne_ALL <- embedding[,1:2]; names(tsne_ALL) <- c('TSNE_ALL_1','TSNE_ALL_2')

# colors = rainbow(8)
# names(colors) = 1:8
# plot(tsne$Y, t='n')
# text(tsne$Y, labels=total_new$Response, col=colors[total_new$Response])

# ii. kmeans clustering
library(h2o)
feature.names <- names(total_new)[-c(1, ncol(total_new))]    
localH2O <- h2o.init(ip = 'localhost', port = 54321, max_mem_size = '12g')
kmeans_df <- as.h2o(localH2O, total_new[,feature.names])
fit <- h2o.kmeans(kmeans_df, k = 4, max_iterations = 50000, standardize = T, init = 'Random', seed = 1989) #none, PlusPlus, Furthest, Random
pred <- as.data.frame(h2o.predict(object = fit, newdata = kmeans_df))
kmeans_all <- pred[,1]; table(kmeans_all)

#####################
# 9. Re-combine #####
#####################
total_new <- cbind(total[,-which(names(total) %in% c(cate.features,'Response'))], dummies, 
                   tsne_ALL, tsne_MH, tsne_MK, tsne_PI, tsne_EI, tsne_II, tsne_IH, tsne_FH, tsne_CI,
                   KMEANS_ALL = kmeans_all, 
                   Response = total$Response)

train <- total_new[total_new$Response != 9, ]
test <- total_new[total_new$Response == 9, ]
library(caret)
set.seed(1989)
inTraining <- createDataPartition(train$Response, p = .9, list = FALSE)
validation_10 <- train[-inTraining,]
train_10 <- train[inTraining,]
inTraining <- createDataPartition(train$Response, p = .8, list = FALSE)
validation_20 <- train[-inTraining,]
train_20 <- train[inTraining,]

save(train, test, validation_10, validation_20, train_10, train_20, file = 'cleaned_datasets.RData')
