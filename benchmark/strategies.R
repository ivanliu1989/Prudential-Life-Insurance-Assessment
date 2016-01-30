# A, B
# A->B
# B->A
# ALL_PRED -> NEW FEAT
# 
# 1. TSNE & features X
# 2. One-hot/PCA X
# 3. OBJECTIVES (RMSE + KAPPA)
# 4. H2O + XGBOOST
# 5. META FEAT (XGBOOST + H2O)

###########################
### Ensemble strategies ###
###########################
##############
### STEP 2 ###
##############
# 3. Stacked generalization
# Split the train set in 2 parts: train_a and train_b
# Fit a first-stage model on train_a and create predictions for train_b
# Fit the same model on train_b and create predictions for train_a
# Finally fit the model on the entire train set and create predictions for the test set.
# Now train a second-stage stacker model on the probabilities from the first-stage model(s).

# 4. (Qauadratic) Linear stacking
# This can easily be combined with feature-weighted linear stacking: 
# -q fs -q ss, possibly improving on both.

# *5. Stacking classifiers with regressors and vice versa
# The predicted probabilities for these classes can help a stacking regressor make better predictions.

##############
### STEP 1 ###
##############
# 6. Stacking unsupervised learned features
# K-Means clustering is a popular technique that makes sense here. 
# Another more recent interesting addition is to use t-SNE: 
# Reduce the dataset to 2 or 3 dimensions and stack this with a non-linear stacker. 
# Using a holdout set for stacking/blending feels like the safest choice here.

# 8. Feature 
# Correlation distance
    # For each row, I calculated the distance between that row and all the rows in Class 1, Class 2, etc.
    # That gave me a distribution of distances for that row.
    # Then I calculated the 10, 25, 50, 75, 90th percentile of those distributions for each class.
# previous best model predictions as features.
# Genetic Algorithm: DEAP

##############
### STEP 3 ###
##############
# 7. Model Selection
# Greedy forward model selection
# Genetic model selection

# 1. Correlation
# Calculate Pearson correlation for all submissions
# Gather wel-performing models which were less correlated.
# Significant increase by average those models

# 2. Weighing (Voting ensemble)
# Count the vote by the best model 3 times
# The other models for one vote each
