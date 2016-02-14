setwd('/Users/ivanliu/Downloads/Prudential-Life-Insurance-Assessment')
rm(list=ls()); gc()
library(h2o)
library(readr)
library(Metrics)
library(Hmisc)
library(mlr)
library(caret)
options(scipen=999);set.seed(19890624)
localH2O <- h2o.init(ip = 'localhost', port = 54321, max_mem_size = '12g')

# 1. Read Data #####
load('data/xgb_meta_leaf_20150211_dummy.RData')
mthd <- 'GLM' # GBM, DL, RF, GLM
# 2. Eval Func #####
evalerror_2 = function(x = seq(1.5, 7.5, by = 1), preds, labels) {
    cuts = c(min(preds), x[1], x[2], x[3], x[4], x[5], x[6], x[7], max(preds))
    preds = as.numeric(Hmisc::cut2(preds, cuts))
    err = Metrics::ScoreQuadraticWeightedKappa(as.numeric(labels), preds, 1, 8)
    return(-err)
}

# 3. Model strategies ##### 
cv <- 2
folds <- createFolds(as.factor(train$Response), k = cv, list = FALSE)
dropitems <- c('Id','Response')
independent <- names(train)[!names(train) %in% dropitems] 
dependent <- "Response"
colnames(train) <- c('Id', paste0('var_', 1:length(independent)), dependent)
colnames(test) <- c('Id', paste0('var_', 1:length(independent)))
independent <- paste0('var_', 1:length(independent))

### Setup Results Table ###
results <- as.data.frame(matrix(rep(0,11*cv), cv))
names(results) <- c('cv_num', 'kappa', 'optim_kappa', 'fixed_kappa', '1st_cut', '2nd_cut', 
                    '3rd_cut', '4th_cut', '5th_cut', '6th_cut', '7th_cut')

for(i in 1:cv){
    f <- folds==i
    train_df          <- as.h2o(localH2O, train[!f,])
    validation_df     <- as.h2o(localH2O, train[f,]) 
    validation        <- train[f,]
    #     if(mthd == 'GBM'){
    #         print('Start training GBM...')
    #         fit <- h2o.gbm(
    #             y = dependent, x = independent, training_frame = train_df, 
    #             ntrees = 800, max_depth = 6, min_rows = 2,
    #             learn_rate = 0.035, distribution = "gaussian", #multinomial, gaussian
    #             nbins_cats = 20, importance = F
    #         )
    #     }
    #     else if(mthd == 'DL'){
    print('Start training Deep Learning...')
    fit <-
        h2o.deeplearning(
            y = dependent, x = independent, training_frame = train_df, overwrite_with_best_model = T, #autoencoder
            use_all_factor_levels = T, activation = "RectifierWithDropout",#TanhWithDropout "RectifierWithDropout"
            hidden = c(256,128), epochs = 18, train_samples_per_iteration = -2, adaptive_rate = T, rho = 0.99,  #c(300,150,75)
            epsilon = 1e-6, rate = 0.035, rate_decay = 0.9, momentum_start = 0.9, momentum_stable = 0.99,
            nesterov_accelerated_gradient = T, input_dropout_ratio = 0.5, hidden_dropout_ratios = c(0.5,0.5), 
            l1 = 1e-5, l2 = 3e-5, loss = 'MeanSquare', classification_stop = 0.01,
            diagnostics = T, variable_importances = F, fast_mode = F, ignore_const_cols = T,
            force_load_balance = T, replicate_training_data = T, shuffle_training_data = T
        )
    #     }
    #     else if(mthd == 'RF'){
    #         print('Start training Random Forest...')
    #         fit <-
    #             h2o.randomForest(
    #                 y = dependent, x = independent, training_frame = train_df, mtries = -1, 
    #                 ntrees = 800, max_depth = 16, sample.rate = 0.632, min_rows = 1, 
    #                 nbins = 20, nbins_cats = 1024, binomial_double_trees = T
    #             )
    #     }
    #     else if(mthd == 'GLM'){
    #         print('Start training GLM...')
    #         fit <-
    #             h2o.glm(
    #                 y = dependent, x = independent, training_frame = train_df, #train_df | total_df
    #                 max_iterations = 100, beta_epsilon = 1e-4, solver = "L_BFGS", #IRLSM  L_BFGS
    #                 standardize = F, family = 'gaussian', link = 'identity', alpha = 0.1, # 1 lasso 0 ridge
    #                 lambda = 0, lambda_search = T, nlambda = 55, #lambda_min_ratio = 1e-08,
    #                 intercept = T
    #             )
    #     "gaussian": "identity", "log"
    #     "tweedie": "tweedie"
    # }
    
    print(fit)
    validPreds <- as.data.frame(h2o.predict(object = fit, newdata = validation_df))
    kappa <- evalerror_2(preds = validPreds$predict, labels = train[f,'Response'])  
    ### Find optimal cutoff
    optCuts = optim(seq(1.5, 7.5, by = 1), evalerror_2, preds = validPreds$predict, labels = train[f,'Response'], 
                    method = 'Nelder-Mead', control = list(maxit = 30000, trace = TRUE, REPORT = 500))
    validPredsOptim = as.numeric(Hmisc::cut2(validPreds$predict, c(-Inf, optCuts$par, Inf))); table(validPredsOptim)
    optimal_kappa <- evalerror_2(preds = validPredsOptim, labels = validation$Response)
    fix_cut <- c(2.6121, 3.3566, 4.1097, 5.0359, 5.5267, 6.4481, 6.7450)
    # c(2.951759, 3.653780, 4.402781, 4.911808, 5.543988, 6.135754, 6.716891)
    validPredsFix = as.numeric(Hmisc::cut2(validPreds[,1], c(-Inf, fix_cut, Inf)));
    fix_kappa = evalerror_2(preds = validPredsFix, labels = train[f,'Response'])
    
    results[i,1:11] <- c(paste0('CV_', i), -kappa, -optimal_kappa, -fix_kappa, optCuts$par)
    View(results)
}