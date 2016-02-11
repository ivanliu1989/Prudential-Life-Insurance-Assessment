setwd('/Users/ivanliu/Downloads/Prudential-Life-Insurance-Assessment')
load('../final_regression_pca.RData')
library(caret)
library(VGAM)
evalerror_2 = function(x = seq(1.5, 7.5, by = 1), preds, labels) {
    cuts = c(min(preds), x[1], x[2], x[3], x[4], x[5], x[6], x[7], max(preds))
    preds = as.numeric(Hmisc::cut2(preds, cuts))
    err = Metrics::ScoreQuadraticWeightedKappa(as.numeric(labels), preds, 1, 8)
    return(-err)
}

### Split Data ###
set.seed(1989)
cv <- 10
folds <- createFolds(as.factor(train$Response), k = cv, list = FALSE)
dropitems <- c('Id','Response')
feature.names <- names(train)[!names(train) %in% dropitems] 
train_sc <- train
test_sc <- test

fit <- vglm(as.factor(Response) ~ . , family=cumulative(parallel=TRUE), data=train_sc[,-1]) # Logistics regression Using Cumulative Logits


### Start Training ###
for(i in 1:cv){
    f <- folds==i
    dval          <- train_sc[f,]
    dtrain        <- train_sc[!f,]
    
    # fitted(fit)
    fit <- vglm(as.factor(Response) ~ . , family=cumulative(parallel=TRUE), data=dtrain[,-1]) # Logistics regression Using Cumulative Logits
    p1_r <- predict(fit, type = "response", dval)
    p1_t <- max.col(p1_r)
    print(evalerror_2(preds = p1_t, labels = dval[,'Response']))
    
    fit2 <- vglm(factor(Response) ~ . , family=cumulative, data=dtrain[,-1]) # without proportional odds
    p2_r <- predict(fit2, type = "response", dval)
    p2_t <- max.col(p2_r)
    print(evalerror_2(preds = p2_t, labels = dval[,'Response']))
    
    fit3 <- vglm(factor(Response) ~ . , family=acat(reverse = TRUE, parallel = TRUE), data=dtrain[,-1]) # ADJACENT CATEGORIES LOGIT MODEL
    p3_r <- predict(fit3, type = "response", dval)
    p3_t <- max.col(p3_r)
    print(evalerror_2(preds = p3_t, labels = dval[,'Response']))
    
    fit4 <- vglm(factor(Response) ~ . , family=acat(reverse = TRUE, parallel = FALSE), data=dtrain[,-1]) # ADJACENT CATEGORIES LOGIT MODEL WITH NONPROPORTIONAL ODDS
    p4_r <- predict(fit4, type = "response", dval)
    p4_t <- max.col(p4_r)
    print(evalerror_2(preds = p4_t, labels = dval[,'Response']))
    
    fit5 <- vglm(factor(Response) ~ . , family=cratio(reverse = FALSE, parallel = TRUE), data=dtrain[,-1]) # CONTINUATION RATIO LOGIT
    p5_r <- predict(fit5, type = "response", dval)
    p5_t <- max.col(p5_r)
    print(evalerror_2(preds = p5_t, labels = dval[,'Response']))
    
    fit6 <- vglm(factor(Response) ~ . , family=cumulative(link = probit, parallel = TRUE), data=dtrain[,-1]) # CUMULATIVE PROBIT MODEL
    p6_r <- predict(fit6, type = "response", dval)
    p6_t <- max.col(p6_r)
    print(evalerror_2(preds = p6_t, labels = dval[,'Response']))
    
    fit7 <- vglm(factor(Response) ~ . , family=cumulative(link = logit, parallel = TRUE), data=dtrain[,-1]) 
    p7_r <- predict(fit7, type = "response", dval)
    p7_t <- max.col(p7_r)
    print(evalerror_2(preds = p7_t, labels = dval[,'Response']))
    
    fit8 <- vglm(factor(Response) ~ . , family=cumulative(link = cloglog, parallel = TRUE), data=dtrain[,-1]) # CUMULATIVE complementary log-log MODEL
    p8_r <- predict(fit8, type = "response", dval)
    p8_t <- max.col(p8_r)
    print(evalerror_2(preds = p8_t, labels = dval[,'Response']))
    
    ### Make new features
    if(i==1){
        p1_r <- as.data.frame(p1_r); names(p1_r) <- paste0('p1_r_', 1:8)
        p2_r <- as.data.frame(p2_r); names(p2_r) <- paste0('p2_r_', 1:8)
        p3_r <- as.data.frame(p3_r); names(p3_r) <- paste0('p3_r_', 1:8)
        p4_r <- as.data.frame(p4_r); names(p4_r) <- paste0('p4_r_', 1:8)
        p5_r <- as.data.frame(p5_r); names(p5_r) <- paste0('p5_r_', 1:8)
        p6_r <- as.data.frame(p6_r); names(p6_r) <- paste0('p6_r_', 1:8)
        p7_r <- as.data.frame(p7_r); names(p7_r) <- paste0('p7_r_', 1:8)
        p8_r <- as.data.frame(p8_r); names(p8_r) <- paste0('p8_r_', 1:8)
        
        train_2nd <- train_sc[f,]
        train_2nd <- cbind(train_2nd, p1_r, p2_r, p3_r,p4_r,p5_r,p6_r,p7_r,p8_r)
        train_meta <- train_2nd
        
    }else{
        p1_r <- as.data.frame(p1_r); names(p1_r) <- paste0('p1_r_', 1:8)
        p2_r <- as.data.frame(p2_r); names(p2_r) <- paste0('p2_r_', 1:8)
        p3_r <- as.data.frame(p3_r); names(p3_r) <- paste0('p3_r_', 1:8)
        p4_r <- as.data.frame(p4_r); names(p4_r) <- paste0('p4_r_', 1:8)
        p5_r <- as.data.frame(p5_r); names(p5_r) <- paste0('p5_r_', 1:8)
        p6_r <- as.data.frame(p6_r); names(p6_r) <- paste0('p6_r_', 1:8)
        p7_r <- as.data.frame(p7_r); names(p7_r) <- paste0('p7_r_', 1:8)
        p8_r <- as.data.frame(p8_r); names(p8_r) <- paste0('p8_r_', 1:8)
        
        train_2nd <- train_sc[f,]
        train_2nd <- cbind(train_2nd, p1_r, p2_r, p3_r,p4_r,p5_r,p6_r,p7_r,p8_r)
        train_meta <- rbind(train_meta, train_2nd)
    }
}


# test
dval          <- test
dtrain        <- train

fit <- vglm(factor(Response) ~ . , family=cumulative(parallel=TRUE), data=dtrain[,-1]) # Logistics regression Using Cumulative Logits
p1_r <- predict(fit, type = "response", dval)
p1_t <- max.col(p1_r)

fit2 <- vglm(factor(Response) ~ . , family=cumulative, data=dtrain[,-1]) # without proportional odds
p2_r <- predict(fit2, type = "response", dval)
p2_t <- max.col(p2_r)


fit3 <- vglm(factor(Response) ~ . , family=acat(reverse = TRUE, parallel = TRUE), data=dtrain[,-1]) # ADJACENT CATEGORIES LOGIT MODEL
p3_r <- predict(fit3, type = "response", dval)
p3_t <- max.col(p3_r)

fit4 <- vglm(factor(Response) ~ . , family=acat(reverse = TRUE, parallel = FALSE), data=dtrain[,-1]) # ADJACENT CATEGORIES LOGIT MODEL WITH NONPROPORTIONAL ODDS
p4_r <- predict(fit4, type = "response", dval)
p4_t <- max.col(p4_r)

fit5 <- vglm(factor(Response) ~ . , family=cratio(reverse = FALSE, parallel = TRUE), data=dtrain[,-1]) # CONTINUATION RATIO LOGIT
p5_r <- predict(fit5, type = "response", dval)
p5_t <- max.col(p5_r)

fit6 <- vglm(factor(Response) ~ . , family=cumulative(link = probit, parallel = TRUE), data=dtrain[,-1]) # CUMULATIVE PROBIT MODEL
p6_r <- predict(fit6, type = "response", dval)
p6_t <- max.col(p6_r)

fit7 <- vglm(factor(Response) ~ . , family=cumulative(link = logit, parallel = TRUE), data=dtrain[,-1]) 
p7_r <- predict(fit7, type = "response", dval)
p7_t <- max.col(p7_r)

fit8 <- vglm(factor(Response) ~ . , family=cumulative(link = cloglog, parallel = TRUE), data=dtrain[,-1]) # CUMULATIVE complementary log-log MODEL
p8_r <- predict(fit8, type = "response", dval)
p8_t <- max.col(p8_r)

p1_r <- as.data.frame(p1_r); names(p1_r) <- paste0('p1_r_', 1:8)
p2_r <- as.data.frame(p2_r); names(p2_r) <- paste0('p2_r_', 1:8)
p3_r <- as.data.frame(p3_r); names(p3_r) <- paste0('p3_r_', 1:8)
p4_r <- as.data.frame(p4_r); names(p4_r) <- paste0('p4_r_', 1:8)
p5_r <- as.data.frame(p5_r); names(p5_r) <- paste0('p5_r_', 1:8)
p6_r <- as.data.frame(p6_r); names(p6_r) <- paste0('p6_r_', 1:8)
p7_r <- as.data.frame(p7_r); names(p7_r) <- paste0('p7_r_', 1:8)
p8_r <- as.data.frame(p8_r); names(p8_r) <- paste0('p8_r_', 1:8)

train_2nd <- test
train_2nd <- cbind(train_2nd, p1_r, p2_r, p3_r,p4_r,p5_r,p6_r,p7_r,p8_r)
test_meta <- train_2nd




# library(MASS)
# fit <- polr(as.factor(Response) ~ . , data=train[,-1])
# 
# library(ordinal)
# fit <- clm(as.factor(Response) ~ . , data = train[,-1])
# fit <- clm2(as.factor(Response) ~ . , data = train[,-1])
# fit <- clmm2(as.factor(Response) ~ . , data = train[,-1])

