setwd('/Users/ivanliu/Downloads/Prudential-Life-Insurance-Assessment')
library(data.table)
rm(list=ls());gc()

##############################
          # 1. Read Data #####
          ####################
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
                   "Medical_History_8", "Medical_History_9", "Medical_History_11", "Medical_History_12", #Medical_History_10
                   "Medical_History_13", "Medical_History_14", "Medical_History_16", "Medical_History_17", "Medical_History_18", 
                   "Medical_History_19", "Medical_History_20", "Medical_History_21", "Medical_History_22", "Medical_History_23", 
                   "Medical_History_25", "Medical_History_26", "Medical_History_27", "Medical_History_28", "Medical_History_29", 
                   "Medical_History_30", "Medical_History_31", "Medical_History_33", "Medical_History_34", "Medical_History_35", 
                   "Medical_History_36", "Medical_History_37", "Medical_History_38", "Medical_History_39", "Medical_History_40", 
                   "Medical_History_41",
                   "KMEANS"
)
medical.keywords <- c(paste('Medical_Keyword_', 1:48, sep = ''))

head(train[,na.features]) # all Numerical (Continuous) or Numerical (Discrete) - 240 
test$Response <- 9

# 2. remove low quality records
removal_feat <- c('Product_Info_3', 'Product_Info_7', 'Employment_Info_2', 'InsuredInfo_3', 'Insurance_History_2', 'Insurance_History_3',
                  'Medical_History_3', 'Medical_History_5', 'Medical_History_6', 'Medical_History_9', 'Medical_History_12', 
                  'Medical_History_16', 'Medical_History_17', 'Medical_History_23', 'Medical_History_31', 'Medical_History_37',
                  'Medical_History_41', 'Medical_History_2')
for(f in removal_feat){
    cat(f)
    train[train[,f] %in% as.numeric(names(table(train[,f]))[!names(table(train[,f])) %in% names(table(test[,f]))]),f] <- -1
}
train[train$Product_Info_3 %in% c(1,3,5,12,13,16,18,20,22,27,32,38),]; test[test$Product_Info_3 %in% c(1,3,5,12,13,16,18,20,22,27,32,38),];
train[train$Product_Info_7 %in% c(2),]; test[test$Product_Info_7 %in% c(2),];
train[train$Employment_Info_2 %in% c(17,22,25,28,29,30,31,33,35,38),]; test[test$Employment_Info_2 %in% c(17,22,25,28,29,30,31,33,35,38),];
train[train$InsuredInfo_3 %in% c(5,7),]; test[test$InsuredInfo_3 %in% c(5,7),]; nrow(train[train$InsuredInfo_3 %in% c(5,7),])
train[train$Insurance_History_2 %in% c(2),]; test[test$Insurance_History_2 %in% c(2),]; nrow(train[train$Insurance_History_2 %in% c(2),])
train[train$Insurance_History_3 %in% c(2),]; test[test$Insurance_History_3 %in% c(2),]; nrow(train[train$Insurance_History_3 %in% c(2),])
train[train$Medical_History_3 %in% c(1),]; test[test$Medical_History_3 %in% c(1),]; nrow(train[train$Medical_History_3 %in% c(1),])
train[train$Medical_History_5 %in% c(3),]; test[test$Medical_History_5 %in% c(3),]; nrow(train[train$Medical_History_5 %in% c(3),])
train[train$Medical_History_6 %in% c(2),]; test[test$Medical_History_6 %in% c(2),]; nrow(train[train$Medical_History_6 %in% c(2),])
train[train$Medical_History_9 %in% c(3),]; test[test$Medical_History_9 %in% c(3),]; nrow(train[train$Medical_History_9 %in% c(3),])
train[train$Medical_History_12 %in% c(1),]; test[test$Medical_History_12 %in% c(1),]; nrow(train[train$Medical_History_12 %in% c(1),])
train[train$Medical_History_16 %in% c(2),]; test[test$Medical_History_16 %in% c(2),]; nrow(train[train$Medical_History_16 %in% c(2),])
train[train$Medical_History_17 %in% c(1),]; test[test$Medical_History_17 %in% c(1),]; nrow(train[train$Medical_History_17 %in% c(1),])
train[train$Medical_History_23 %in% c(2),]; test[test$Medical_History_23 %in% c(2),]; nrow(train[train$Medical_History_23 %in% c(2),])
train[train$Medical_History_31 %in% c(2),]; test[test$Medical_History_31 %in% c(2),]; nrow(train[train$Medical_History_31 %in% c(2),])
train[train$Medical_History_37 %in% c(3),]; test[test$Medical_History_37 %in% c(3),]; nrow(train[train$Medical_History_37 %in% c(3),])
train[train$Medical_History_41 %in% c(2),]; test[test$Medical_History_41 %in% c(2),]; nrow(train[train$Medical_History_41 %in% c(2),])
Medical_History_2_feat <- as.numeric(names(table(train$Medical_History_2))[!names(table(train$Medical_History_2)) %in% names(table(test$Medical_History_2))])

# 3. combine train and test
total <- rbind(train, test); dim(total)

#####################################
          # 2. Feature engineer #####
          ###########################
# 1. % of NA / # of NA
Cnt_NA_row <- apply(total, 1, function(x) sum(is.na(x)))
Cnt_NA_Emp_row <- apply(total[, c('Employment_Info_1', 'Employment_Info_4', 'Employment_Info_6')], 1, function(x) sum(is.na(x)))
Cnt_NA_Fam_row <- apply(total[, c('Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5')], 1, function(x) sum(is.na(x)))
Cnt_NA_Medi_row <- apply(total[, c('Medical_History_1', 'Medical_History_10', 'Medical_History_15', 'Medical_History_24', 'Medical_History_32')], 1, function(x) sum(is.na(x)))
Cnt_NA_Ins_row <- ifelse(is.na(total[, 'Insurance_History_5']), 1,0)

# 2. # of Medical keywords
Cnt_Medi_Key_Word <- rowSums(total[, medical.keywords])

# 3. BMI & Age
BMI_InsAge <- total$BMI * total$Ins_Age

# 4. Split product_info_2
Product_Info_2_cate <- substr(total$Product_Info_2, 1,1)
Product_Info_2_num <- substr(total$Product_Info_2, 2,2)

# 5. Split by gender

# 6. Information specific to gender NewFeature1
Gender_Speci_feat <- NA
Gender_Speci_feat <- ifelse(total$Family_Hist_2>=0 & is.na(total$Family_Hist_3), 1, Gender_Speci_feat)
Gender_Speci_feat <- ifelse(total$Family_Hist_3>=0 & is.na(total$Family_Hist_2), 0, Gender_Speci_feat)
Gender_Speci_feat[is.na(Gender_Speci_feat)] <- -1

# 7. imputation
for(i in na.features){
    total[is.na(total[,i]),i] <- median(total[,i], na.rm = T)
    # total[is.na(total[,i]),i] <- -1
}
sapply(names(total), function(x){mean(is.na(total[,x]))})

# 8. High correlation
# 9. as.factor/one-hot
cate.features <- c(cate.features, 'Product_Info_2_cate', 'Product_Info_2_num', 'Gender_Speci_feat')
num.features <- names(total)[!names(total) %in% c(cate.features, 'Id', 'Response', medical.keywords)]

load('data/i_cluster_feats.RData')
total_new <- cbind(total[,-which(names(total) %in% c('Response'))],  
                   Cnt_NA_row = Cnt_NA_row,
                   Cnt_NA_Emp_row = Cnt_NA_Emp_row,
                   Cnt_NA_Fam_row = Cnt_NA_Fam_row,
                   Cnt_NA_Medi_row = Cnt_NA_Medi_row,
                   Cnt_NA_Ins_row = Cnt_NA_Ins_row,
                   Cnt_Medi_Key_Word = Cnt_Medi_Key_Word,
                   BMI_InsAge = BMI_InsAge,
                   Product_Info_2_cate = Product_Info_2_cate,
                   Product_Info_2_num = Product_Info_2_num,
                   Gender_Speci_feat = Gender_Speci_feat,
                   tsne_all, KMEANS = kmeans_all,
                   Response = total$Response)
for(c in cate.features){
    total_new[,c] <- as.factor(total_new[,c])
}
str(total_new[,cate.features])
for(n in num.features){
    total_new[,n] <- as.numeric(total_new[,n])
}
str(total_new[,num.features])
sapply(names(total_new), function(x){mean(is.na(total_new[,x]))})

levels(total_new$Product_Info_2) <- 1:length(levels(total_new$Product_Info_2))
table(total_new$Product_Info_2)

######################################
          # 4. Split/Output Data #####
          ############################
train <- total_new[total_new$Response != 9, ]
test <- total_new[total_new$Response == 9, ]
library(caret)
set.seed(1989)
# No validation
inTraining <- createDataPartition(train$Response, p = .5, list = FALSE)
train_a <- train[-inTraining,]
train_b <- train[inTraining,]
dim(train_b); dim(train_a); dim(test); dim(train)
save(train, train_b, train_a, test, file = 'data/fin_train_test.RData')
str(train[,num.features]); str(train[,cate.features])

# Validation
inTraining <- createDataPartition(train$Response, p = .2, list = FALSE)
validation <- train[inTraining,]
# Train a & b
train <- train[-inTraining,]
inTraining <- createDataPartition(train$Response, p = .5, list = FALSE)
train_a <- train[-inTraining,]
train_b <- train[inTraining,]

dim(train_b); dim(train_a); dim(validation); dim(test); dim(train)
save(train, train_b, train_a, validation, test, file = 'data/fin_train_test_validation.RData')
str(train[,num.features]); str(train[,cate.features])

#######################################
          # 5. One - hot encoding #####
          #############################
total_new <- cbind(data.frame(model.matrix(Response~.-1,total_new)), Response = total_new$Response)
head(total_new);dim(total_new)

# train <- total_new[total_new$Response != 9, ]
# test <- total_new[total_new$Response == 9, ]
# library(caret)
# set.seed(1989)
# # Validation
# inTraining <- createDataPartition(train$Response, p = .2, list = FALSE)
# validation <- train[inTraining,]
# # Train a & b
# train <- train[-inTraining,]
# inTraining <- createDataPartition(train$Response, p = .5, list = FALSE)
# train_a <- train[-inTraining,]
# train_b <- train[inTraining,]
# 
# dim(train_b); dim(train_a); dim(validation); dim(test); dim(train)
# save(train, train_b, train_a, validation, test, file = 'data/fin_train_test_validation_onehot.RData')

#################################################
          # 6. tsne/kmeans/distance feature #####
          #######################################
library(Rtsne)
feature.names <- names(total_new)[-c(1, ncol(total_new))]
tsne <- Rtsne(as.matrix(total_new[,feature.names]), dims = 3, perplexity=30, check_duplicates = F, pca = F, theta=0.5) #max_iter = 300, 
embedding <- as.data.frame(tsne$Y)
tsne_all <- embedding[,1:3]; names(tsne_all) <- c('TSNE_1','TSNE_2','TSNE_3')

library(h2o) 
localH2O <- h2o.init(ip = 'localhost', port = 54321, max_mem_size = '12g')
kmeans_df <- as.h2o(localH2O, total_new[,feature.names])
fit <- h2o.kmeans(kmeans_df, k = 8, max_iterations = 1000, standardize = T, init = 'Random', seed = 1989) #none, PlusPlus, Furthest, Random
pred <- as.data.frame(h2o.predict(object = fit, newdata = kmeans_df))
kmeans_all <- pred[,1]; table(kmeans_all)

library(caret)
centroids <- classDist(as.data.frame(total_new[, 2:672]), total_new[, 'Response'])
# nzv <- nearZeroVar(total_new[, 2:672], saveMetrics= TRUE)
# nzv[nzv$nzv,][1:10,]
# comboInfo <- findLinearCombos(total_new[, 2:672])
# comboInfo

library(pdist)
d <- pdist(total_new[, 2:672])

save(tsne_ALL,kmeans_all,distance_all, file = 'data/i_cluster_feats.RData')

#################################
         # 7. pca transform #####
         ########################
preProcValues <- preProcess(total_new[,-c(1,ncol(total_new))], method = c("pca"))
total_pca <- predict(preProcValues, total_new[,-c(1,ncol(total_new))])
total_pca <- cbind(Id = total_new$Id, total_pca, Response = total_new$Response)

train <- total_pca[total_pca$Response != 9, ]
test <- total_pca[total_pca$Response == 9, ]
library(caret)
set.seed(1989)
# Validation
inTraining <- createDataPartition(train$Response, p = .2, list = FALSE)
validation <- train[inTraining,]
# Train a & b
train <- train[-inTraining,]
inTraining <- createDataPartition(train$Response, p = .5, list = FALSE)
train_a <- train[-inTraining,]
train_b <- train[inTraining,]

dim(train_b); dim(train_a); dim(validation); dim(test); dim(train)
save(train, train_b, train_a, validation, test, file = 'data/fin_train_test_validation_pca.RData')
