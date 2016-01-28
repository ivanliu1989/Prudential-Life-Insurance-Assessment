# 1. % of NA / # of NA
# 2. % of non NA / # of non NA
# 3. # of Medical keywords
# 4. BMI & Age
# 5. Split product_info_2
# 6. Remove small counts

# 7. Split by gender
# 8. imputation
# 9. as.factor/one-hot

# 10. Information specific to gender NewFeature1
nrow(train[which(train$Family_Hist_2>=0),])/nrow(train)
nrow(train[which(train$Family_Hist_3>=0),])/nrow(train)
nrow(train[which(train$Family_Hist_4>=0),])/nrow(train)
nrow(train[which(train$Family_Hist_5>=0),])/nrow(train)
nrow(train[which(train$Family_Hist_2>=0 & train$Family_Hist_3>=0),]) # check 1 
nrow(train[which(train$Family_Hist_4>=0 & train$Family_Hist_5>=0),]) # check 2

train$NewFeature1 <- NA
train$NewFeature1 <- ifelse(train$Family_Hist_2>=0 & is.na(train$Family_Hist_3), 1, train$NewFeature1)
train$NewFeature1 <- ifelse(train$Family_Hist_3>=0 & is.na(train$Family_Hist_2), 0, train$NewFeature1)

# 11. High correlation

# 12. Stacking results (methods/objective functions)