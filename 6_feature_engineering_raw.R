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

save(tsne_ALL, tsne_MH, tsne_MK, tsne_PI, tsne_EI, tsne_II, tsne_IH, tsne_FH, tsne_CI, kmeans_all, file = 'data/cleaned_datasets.RData')

#####################
# 9. Re-combine #####
#####################
total_new <- cbind(total[,-which(names(total) %in% c(cate.features,'Response'))], dummies, 
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
inTraining <- createDataPartition(train$Response, p = .9, list = FALSE)
validation_10 <- train[-inTraining,]
train_10 <- train[inTraining,]
inTraining <- createDataPartition(train$Response, p = .8, list = FALSE)
validation_20 <- train[-inTraining,]
train_20 <- train[inTraining,]

save(train, test, validation_10, validation_20, train_10, train_20, file = 'data/cleaned_datasets.RData')
