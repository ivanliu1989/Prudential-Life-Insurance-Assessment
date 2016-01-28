setwd('/Users/ivanliu/Downloads/Prudential-Life-Insurance-Assessment')
rm(list=ls()); gc()
library(data.table); library(dplyr); library(ggplot2)
train <- fread('data/train.csv')
test <- fread('data/test.csv')
submission <- fread('data/sample_submission.csv')

# 1. basic 
dim(train);dim(test)
View(train)

#Those considered as no binary:
train=dplyr::filter(train, Product_Info_7!=2)
train=dplyr::filter(train, Insurance_History_3!=2)
train=dplyr::filter(train, Medical_History_5!=3)
train=dplyr::filter(train, Medical_History_6!=2)
train=dplyr::filter(train, Medical_History_9!=3)
train=dplyr::filter(train, Medical_History_12!=1)
train=dplyr::filter(train, Medical_History_16!=2)
train=dplyr::filter(train, Medical_History_17!=1)
train=dplyr::filter(train, Medical_History_23!=2)
train=dplyr::filter(train, Medical_History_31!=2)
train=dplyr::filter(train, Medical_History_37!=3)
train=dplyr::filter(train, Medical_History_41!=2)

# Separate categorical, binary, and other numeric variables:
feat.cat <- c("Product_Info_2", "Employment_Info_2", "InsuredInfo_1", "InsuredInfo_3", "Insurance_History_4",
              "Insurance_History_7", "Insurance_History_8", "Insurance_History_9", "Family_Hist_1", 
              "Medical_History_3", "Medical_History_7", "Medical_History_8", "Medical_History_11", "Medical_History_13", 
              "Medical_History_14", "Medical_History_18", "Medical_History_19", "Medical_History_20", "Medical_History_21",
              "Medical_History_25",
              "Medical_History_26", "Medical_History_27", "Medical_History_28", "Medical_History_29", "Medical_History_30",
              "Medical_History_33", "Medical_History_34", "Medical_History_35", "Medical_History_36", "Medical_History_38", 
              "Medical_History_39", "Medical_History_40") 


feat.bin <- c("Product_Info_7", "Product_Info_6", "Product_Info_5", "Product_Info_1", "InsuredInfo_2", 
              "InsuredInfo_4", "InsuredInfo_5", "InsuredInfo_6", "InsuredInfo_7", "Employment_Info_3",
              "Employment_Info_5", "Insurance_History_1", "Insurance_History_2", "Insurance_History_3", 
              "Medical_History_4", "Medical_History_5", "Medical_History_6", "Medical_History_9", 
              "Medical_History_12", "Medical_History_16", 
              "Medical_History_17", "Medical_History_22", "Medical_History_23", "Medical_History_31", 
              "Medical_History_37",
              "Medical_History_41", 
              paste("Medical_Keyword_", 1:48, sep=""))

feat.numeric <- setdiff(names(test), c(feat.cat, feat.bin))

# We will convert to 1 and zero the binary features:
for (f in feat.bin) {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))-1
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))-1
}

# Now, lets explore predictability for each feature:
response=train$Response
scores.grid <- expand.grid(columns = 1:ncol(test), score = NA)

for (i in 1:ncol(test)) {
    feat=names(test)[i]
    temp=data.frame(train[, feat])
    names(temp)=feat
    #eGrid = expand.grid(cp=0.002)
    nf  <- trainControl(method="cv", number=5, classProbs = FALSE, summaryFunction = defaultSummary)    
    a= train(x=temp, y=response, method = "lm", metric="RMSE",  trControl=nf)
    #p=predict(a, train)
    scores.grid[i, 2]=mean(a$results[,"RMSE"])
    cat(mean(a$results[,"RMSE"]), "") 
    
}

# Now that we have calculated the rmse score for each feature, we plot it:
scores.grid$featName=names(test)
ggplot(data=scores.grid, aes(x=reorder(featName, -score), y=score)) +
    geom_line(colour="darkblue", fill="blue", aes(group="name")) +
    theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
    coord_flip() +
    ggtitle("Features by rmse Score") 
    
# These are the best features according a regression:
head(scores.grid[order(scores.grid$score),], 10)
summary(train$Medical_History_32)
tail(scores.grid[order(scores.grid$score),], 10)

# Lets see our independent variable and how it goes with the top 8 features. We will split the discrete or continuos variables using their mean to obtain a binary version of them.
data=(train[,c(head(scores.grid[order(scores.grid$score),"featName"], 8), "Response")])

# Creating binary columns using the mean
data$Medical_History_32=ifelse(data$Medical_History_32 >mean(data$Medical_History_32, na.rm=T), 1, 0)
data$BMI=ifelse(data$BMI >mean(data$BMI, na.rm=T), 1, 0)
data$Wt =ifelse(data$Wt  >mean(data$Wt , na.rm=T), 1, 0)
data$Family_Hist_4=ifelse(data$Family_Hist_4 >mean(data$Family_Hist_4, na.rm=T), 1, 0)
data$Family_Hist_2=ifelse(data$Family_Hist_2 >mean(data$Family_Hist_2, na.rm=T), 1, 0)
data$Medical_History_15=ifelse(data$Medical_History_15 >mean(data$Medical_History_15, na.rm=T), 1, 0)
data$Medical_History_24=ifelse(data$Medical_History_24 >mean(data$Medical_History_24, na.rm=T), 1, 0)

data <- melt(data, id.vars="Response")
colnames(data)[2:3]=c("Feature", "Value")

data= data %>% dplyr::group_by(Response, Feature) %>%
    dplyr::summarise(probability=mean(Value, na.rm=T))


ggplot(data=data, aes(x=Response, y=probability)) +
    geom_bar(colour="darkblue", fill="blue", stat="identity") +
    facet_grid(Feature~., as.table = TRUE)+
    theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
    #coord_flip() +
    ggtitle("Top 8 features and their binary probability for each Response value") 

# And the baseline was:
plot(table(train$Response)/nrow(train))

# Lets see quickly if the NA’s in Medical_History_32 are predictive
tr=dplyr::filter(train, is.na(Medical_History_32))
plot(table(tr$Response)/nrow(tr))

# We can see that those NA behave very similarly to the baseline probability that we can estimate from the train set. We have a lot of those NA values in this features.
# So.. lets see the ones that are not NA: Lets see quickly if the NA’s in Medical_History_32 are predictive
tr=dplyr::filter(train, !is.na(Medical_History_32))
plot(table(tr$Response)/nrow(tr))
