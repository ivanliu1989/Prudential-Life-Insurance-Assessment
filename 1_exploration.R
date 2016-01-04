setwd('/Users/ivanliu/Downloads/Prudential-Life-Insurance-Assessment')
rm(list=ls()); gc()
library(data.table)
train <- fread('data/train.csv')
test <- fread('data/test.csv')
submission <- fread('data/sample_submission.csv')

# 1. basic 
dim(train);dim(test)
View(train)
