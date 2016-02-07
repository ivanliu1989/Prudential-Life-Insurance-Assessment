setwd('/Users/ivanliu/Downloads/Prudential-Life-Insurance-Assessment')
library(data.table)
rm(list=ls());gc()

# 1. load data and define data structure
cat("read train and test data...\n")
train <- fread("data/train.csv", data.table = F)
test  <- fread("data/test.csv", data.table = F)

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
                   "Medical_History_8", "Medical_History_9", "Medical_History_11", "Medical_History_12",
                   "Medical_History_13", "Medical_History_14", "Medical_History_16", "Medical_History_17", "Medical_History_18", 
                   "Medical_History_19", "Medical_History_20", "Medical_History_21", "Medical_History_22", "Medical_History_23", 
                   "Medical_History_25", "Medical_History_26", "Medical_History_27", "Medical_History_28", "Medical_History_29", 
                   "Medical_History_30", "Medical_History_31", "Medical_History_33", "Medical_History_34", "Medical_History_35", 
                   "Medical_History_36", "Medical_History_37", "Medical_History_38", "Medical_History_39", "Medical_History_40", 
                   "Medical_History_41"
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

# factor
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
total <- rbind(train,test); table(total$Product_Info_2)
levels(total$Product_Info_2) <- c(17, 1, 19, 18, 16, 8, 2, 15, 7, 6, 3, 5, 14, 11, 10, 13, 12, 4, 9)
levels(total$Product_Info_2_cate) <- c(1:5)
train <- total[which(total$Response > 0),]
test <- total[which(total$Response == 0),]

# Split
library(caret)
set.seed(1989)
# No validation
dim(test); dim(train)
save(train, test, file = 'data/fin_train_test_regression.RData')
