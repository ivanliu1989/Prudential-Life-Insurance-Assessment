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
# 6. Information specific to gender NewFeature1
Gender_Speci_feat <- NA
Gender_Speci_feat <- ifelse(total$Family_Hist_2>=0 & is.na(total$Family_Hist_3), 1, Gender_Speci_feat)
Gender_Speci_feat <- ifelse(total$Family_Hist_3>=0 & is.na(total$Family_Hist_2), 0, Gender_Speci_feat)
Gender_Speci_feat[is.na(Gender_Speci_feat)] <- -1
# 7. imputation
for(i in na.features){
  total[is.na(total[,i]),i] <- median(total[,i], na.rm = T)
}
sapply(names(total), function(x){mean(is.na(total[,x]))})

# Categorical
load('data/i_cluster_feats_dist.RData')
cate.features <- c('Product_Info_2_cate', 'Product_Info_2')
total_new <- cbind(total[,-which(names(total) %in% c('Response'))],  
                   Cnt_NA_row = Cnt_NA_row,
                   Cnt_Medi_Key_Word = Cnt_Medi_Key_Word,
                   BMI_InsAge = BMI_InsAge,
                   Product_Info_2_cate = Product_Info_2_cate,
                   Product_Info_2_num = Product_Info_2_num,
                   # Gender_Speci_feat = Gender_Speci_feat, tsne_all, kmeans_all = kmeans_all, 
                   distances_all,
                   Response = total$Response)
for(c in cate.features){
  total_new[,c] <- as.factor(total_new[,c])
}
# levels(total_new$Product_Info_2) <- c(17, 1, 19, 18, 16, 8, 2, 15, 7, 6, 3, 5, 14, 11, 10, 13, 12, 4, 9)#1:length(levels(total_new$Product_Info_2))
dummies <- dummyVars(Response ~ ., data = total_new[,c('Product_Info_2', 'Response')], sep = "_", levelsOnly = FALSE, fullRank = TRUE)
total1 <- as.data.frame(predict(dummies, newdata = total_new))
total_new <- cbind(total_new[,-3], total1, Response=total_new$Response)

levels(total_new$Product_Info_2_cate) <- c(1:5)
for(c in cate.features){
  total_new[,c] <- as.numeric(total_new[,c])
}
for(c in names(total_new)){
    total_new[,c] <- as.numeric(total_new[,c])
}

######################################
# 4. Split/Output Data #####
############################
train <- total_new[total_new$Response != 0, ]
test <- total_new[total_new$Response == 0, ]
library(caret)
set.seed(1989)
# No validation
inTraining <- createDataPartition(train$Response, p = .5, list = FALSE)
train_a <- train[-inTraining,]
train_b <- train[inTraining,]
dim(train_b); dim(train_a); dim(test); dim(train)
save(train, train_b, train_a, test, file = 'data/fin_train_test_prod.RData')

# Validation
inTraining <- createDataPartition(train$Response, p = .2, list = FALSE)
validation <- train[inTraining,]
# Train a & b
train <- train[-inTraining,]
inTraining <- createDataPartition(train$Response, p = .5, list = FALSE)
train_a <- train[-inTraining,]
train_b <- train[inTraining,]

dim(train_b); dim(train_a); dim(validation); dim(test); dim(train)
save(train, train_b, train_a, validation, test, file = 'data/fin_train_test_validation_prod.RData')


# Clustering
library(Rtsne)
feature.names <- names(total_new)[-c(1, ncol(total_new))]
for(c in feature.names){
  total_new[,c] <- as.numeric(total_new[,c])
}
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
centroids <- classDist(as.matrix(total_new[, feature.names]), as.factor(total_new[, 'Response']), pca = T, keep = 149) #672
distances <- predict(centroids, as.matrix(total_new[, feature.names]))
distances <- as.data.frame(distances)
distances_all <- distances[,-1]; names(distances_all) <- paste('Dist_', 1:8, sep = "")

save(distances_all, file = 'data/i_cluster_feats_dist.RData') #kmeans_all,
