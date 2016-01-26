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
                   "Medical_History_8", "Medical_History_9", "Medical_History_11", "Medical_History_12", 
                   "Medical_History_13", "Medical_History_14", "Medical_History_16", "Medical_History_17", "Medical_History_18", 
                   "Medical_History_19", "Medical_History_20", "Medical_History_21", "Medical_History_22", "Medical_History_23", 
                   "Medical_History_25", "Medical_History_26", "Medical_History_27", "Medical_History_28", "Medical_History_29", 
                   "Medical_History_30", "Medical_History_31", "Medical_History_33", "Medical_History_34", "Medical_History_35", 
                   "Medical_History_36", "Medical_History_37", "Medical_History_38", "Medical_History_39", "Medical_History_40", "Medical_History_41")
medical.keywords <- c(paste('Medical_Keyword_', 1:48, sep = ''))

test$Response <- 9
total <- rbind(train, test); dim(total)

#############################
# 5. One - hot encoding #####
#############################
for(i in cate.features){
    total[,i] <- as.factor(total[,i])
}
str(total)

total_new <- total

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

# v. split between Product_Info_2
Product_Info_2_cate <- substr(total$Product_Info_2, 1,1)
Product_Info_2_num <- substr(total$Product_Info_2, 2,2)
Product_Info_2_split <- cbind(Product_Info_2_cate=Product_Info_2_cate, Product_Info_2_num=Product_Info_2_num)
Product_Info_2_split <- data.frame(Product_Info_2_split)
for(i in 1:2){
    Product_Info_2_split[,i] <- as.factor(Product_Info_2_split[,i])
}
head(Product_Info_2_split)

#####################
# 7. Imputation #####
#####################
for(i in na.features){
    # total[is.na(total[,i]),i] <- median(total[,i], na.rm = T)
    total[is.na(total[,i]),i] <- -1
}
sapply(names(total), function(x){mean(is.na(total[,x]))})

#########################################
# 8. kmeans & tsne based on sectors #####
#########################################
load('data/clustering_feats.RData')

#####################
# 9. Re-combine #####
#####################
total_new <- cbind(total[,-which(names(total) %in% c('Response'))], 
                   Cnt_Medi_Key_Word = Cnt_Medi_Key_Word,
                   BMI_InsAge = BMI_InsAge, 
                   Cnt_NA_row = Cnt_NA_row,
                   Product_Info_2_split,
                   tsne_ALL, tsne_MH, tsne_MK, tsne_PI, tsne_EI, tsne_II, tsne_IH, tsne_FH, tsne_CI,
                   KMEANS_ALL = kmeans_all, 
                   Response = total$Response)

train <- total_new[total_new$Response != 9, ]
test <- total_new[total_new$Response == 9, ]
library(caret)
set.seed(1989)
inTraining <- createDataPartition(train$Response, p = .8, list = FALSE)
validation_20 <- train[-inTraining,]
train_20 <- train[inTraining,]

save(train, test, validation_20, train_20, file = 'data/cleaned_datasets_no_encoded.RData')
