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
par(mfcol = c(2,1))
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
table(train$Insurance_History_6); table(test$Insurance_History_6) # All NAs
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
feature.names <- names(train)[2:ncol(train)-1]

################################
# 4. tsne based on sectors #####
################################