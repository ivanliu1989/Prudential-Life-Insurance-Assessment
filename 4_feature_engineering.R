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

# iv. Insured Info 1-6

# v. Insurance History 1-9

# vi. Family History 1-5

# vii. Medical History 1-41

# viii. Medical Keyword 1-48

# ix. Response 1


feature.names <- names(train)[2:ncol(train)-1]

# remove NA values...
train[is.na(train)] <- 0
test[is.na(test)]   <- 0

cat("replace text variables with numerics factors...\n")
for (f in feature.names) {
    if (class(train[[f]])=="character") {
        levels <- unique(c(train[[f]], test[[f]]))
        train[[f]] <- as.integer(factor(train[[f]], levels=levels))
        test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
    }
}

# response values are in the range [1:8] ... make it [0:7] for xgb softmax....
train$Response = train$Response - 1
