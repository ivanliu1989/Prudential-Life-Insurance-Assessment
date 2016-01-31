setwd('/Users/ivanliu/Downloads/Prudential-Life-Insurance-Assessment')
rm(list=ls()); gc()
library(h2o)
library(readr)
library(Metrics)
library(Hmisc)
library(mlr)
options(scipen=999);set.seed(19890624)
localH2O <- h2o.init(ip = 'localhost', port = 54321, max_mem_size = '12g')

##############################
          # 1. Read Data #####
          ####################
    # load('data/fin_train_test_validation.RData')
    # load('data/fin_train_test.RData')
    load('data/fin_train_test_validation_onehot.RData')

##############################
          # 2. Eval Func #####
          ####################
    evalerror_2 = function(x = seq(1.5, 7.5, by = 1), preds, labels) {
        cuts = c(min(preds), x[1], x[2], x[3], x[4], x[5], x[6], x[7], max(preds))
        preds = as.numeric(Hmisc::cut2(preds, cuts))
        err = Metrics::ScoreQuadraticWeightedKappa(as.numeric(labels), preds, 1, 8)
        return(-err)
    }

#####################################
          # 3. Model strategies ##### 
          ###########################
    independent <- colnames(train)[c(2:(ncol(train)-1))]
    dependent <- "Response"
    train_df <- as.h2o(localH2O, train) # train, train_a, train_b
    validation_df <- as.h2o(localH2O, validation)

### 1) gbm
    fit <- h2o.gbm(
        y = dependent, x = independent, training_frame = train_df, 
        ntrees = 800, max_depth = 6, min_rows = 2,
        learn_rate = 0.035, distribution = "gaussian", #multinomial, gaussian
        nbins_cats = 20, importance = F
    )

### 2) deeplearning
    fit <-
        h2o.deeplearning(
            y = dependent, x = independent, training_frame = train_df, overwrite_with_best_model = T, #autoencoder
            use_all_factor_levels = T, activation = "RectifierWithDropout",#TanhWithDropout "RectifierWithDropout"
            hidden = c(300,150,75), epochs = 9, train_samples_per_iteration = -2, adaptive_rate = T, rho = 0.99,  #c(300,150,75)
            epsilon = 1e-6, rate = 0.01, rate_decay = 0.9, momentum_start = 0.9, momentum_stable = 0.99,
            nesterov_accelerated_gradient = T, input_dropout_ratio = 0.25, hidden_dropout_ratios = c(0.25,0.25,0.25), 
            l1 = NULL, l2 = 3e-5, loss = 'MeanSquare', classification_stop = 0.01,
            diagnostics = T, variable_importances = F, fast_mode = F, ignore_const_cols = T,
            force_load_balance = T, replicate_training_data = T, shuffle_training_data = T
        )
    
### 3) random forest
    fit <-
        h2o.randomForest(
            y = dependent, x = independent, training_frame = train_df, mtries = -1, 
            ntrees = 800, max_depth = 16, sample.rate = 0.632, min_rows = 1, 
            nbins = 20, nbins_cats = 1024, binomial_double_trees = T
        )
    
### 4) glm   
    fit <-
        h2o.glm(
            y = dependent, x = independent, training_frame = train_df, #train_df | total_df
            max_iterations = 100, beta_epsilon = 1e-4, solver = "L_BFGS", #IRLSM  L_BFGS
            standardize = F, family = 'tweedie', link = 'tweedie', alpha = 0, # 1 lasso 0 ridge
            lambda = 0, lambda_search = T, nlambda = 55, #lambda_min_ratio = 1e-08,
            intercept = F
        )
#     "gaussian": "identity", "log"
#     "tweedie": "tweedie"

####################################
          # 4. Validation ##########
          ##########################
    validPreds <- as.data.frame(h2o.predict(object = fit, newdata = validation_df))
    ScoreQuadraticWeightedKappa(round(validPreds$predict),validation$Response)
    evalerror_2(preds = validPreds$predict, labels = validation$Response)  
    ### Find optimal cutoff
    optCuts = optim(seq(1.5, 7.5, by = 1), evalerror_2, preds = validPreds$predict, labels = validation$Response, 
                    method = 'Nelder-Mead', control = list(maxit = 30000, trace = TRUE, REPORT = 500))
    validPredsOptim = as.numeric(Hmisc::cut2(validPreds$predict, c(-Inf, optCuts$par, Inf))); table(validPredsOptim)
    evalerror_2(preds = validPredsOptim, labels = validation$Response)
    
    
    # pred_h2o_gbm <- validPreds$predict
    # pred_h2o_glm <- validPreds$predict
    # pred_h2o_rf <- validPreds$predict
    # pred_h2o_dl <- validPreds$predict
    validPreds$predict <- (pred_h2o_gbm + pred_h2o_rf + pred_h2o_glm + pred_h2o_dl)/4
    