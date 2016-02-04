setwd('/Users/ivanliu/Downloads/Prudential-Life-Insurance-Assessment')
load('data/fin_train_test_validation.RData')

train <- rbind(train, validation)
test$Response <- 0

total <- rbind(train, test)

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

total <- cbind(total[,1:(ncol(total)-1)], 
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

######################################
# 4. Split/Output Data #####
############################
train <- total[total$Response != 0, ]
test <- total[total$Response == 0, ]
library(caret)
set.seed(1989)
# No validation
inTraining <- createDataPartition(train$Response, p = .5, list = FALSE)
train_a <- train[-inTraining,]
train_b <- train[inTraining,]
dim(train_b); dim(train_a); dim(test); dim(train)
save(train, train_b, train_a, test, file = 'data/fin_train_test_2.RData')

# Validation
inTraining <- createDataPartition(train$Response, p = .2, list = FALSE)
validation <- train[inTraining,]
# Train a & b
train <- train[-inTraining,]
inTraining <- createDataPartition(train$Response, p = .5, list = FALSE)
train_a <- train[-inTraining,]
train_b <- train[inTraining,]

dim(train_b); dim(train_a); dim(validation); dim(test); dim(train)
save(train, train_b, train_a, validation, test, file = 'data/fin_train_test_validation_2.RData')
