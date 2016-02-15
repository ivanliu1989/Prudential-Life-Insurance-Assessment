setwd('/Users/ivanliu/Downloads/Prudential-Life-Insurance-Assessment')
rm(list=ls());gc()
# load('data/xgb_meta_leaf_20150211_dummy.RData')
load('../final_regression_pca.RData')

options(scipen=999);set.seed(19890624)

head(train)

ffmlib_convert <- function(dt){
    # dropitems <- c('Id','Response')
    # feature.names <- names(train)[!names(train) %in% dropitems] 
    # dt <- dt[,feature.names]
    for(i in 2:(ncol(dt)-1)){
        print(i)
        if(i==2){
            dt$ffm <- paste0(dt$Response,' ',i-1,':', i-1,':',dt[,i])
        }else{
            dt$ffm <- paste0(dt$ffm,' ',i-1,':', i-1,':',dt[,i])
        }   
    }
    return(dt$ffm)
}


train_ffm <- ffmlib_convert(train)
test_ffm <- ffmlib_convert(test)
total_ffm <- ffmlib_convert(rbind(train,test))

write.table(train_ffm, file='../train_ffm_pca.txt', quote = F, row.names = F, col.names = F)
write.table(test_ffm, file='../test_ffm_pca.txt', quote = F, row.names = F, col.names = F)
write.table(total_ffm, file='../total_ffm_pca.txt', quote = F, row.names = F, col.names = F)


svmlib_convert <- function(dt){
    # dt <- dt[,c(58, 3:57)]
    # dt <- dt[,c(37, 3:35)]
    # dt$flag_class <- ifelse(dt$flag_class == 'Y', 1, 0)
    for(i in 2:(ncol(dt)-1)){
        print(i)
        if(i==2){
            dt$ffm <- paste0(dt$Response,' ',i-1,':',dt[,i])
        }else{
            dt$ffm <- paste0(dt$ffm,' ',i-1,':',dt[,i])
        }   
    }
    return(dt$ffm)
}

train_svm <- svmlib_convert(train)
test_svm <- svmlib_convert(test)
total_svm <- svmlib_convert(rbind(train,test))

write.table(train_svm, file='../train_svm_pca.txt', quote = F, row.names = F, col.names = F)
write.table(test_svm, file='../test_svm_pca.txt', quote = F, row.names = F, col.names = F)
write.table(total_svm, file='../total_svm_pca.txt', quote = F, row.names = F, col.names = F)
