# Prudential-Life-Insurance-Assessment

### Data fields
#### Variable	Description
Id	A unique identifier associated with an application. <br>
Product_Info_1-7	A set of normalized variables relating to the product applied for <br>
Ins_Age	Normalized age of applicant <br>
Ht	Normalized height of applicant <br>
Wt	Normalized weight of applicant <br>
BMI	Normalized BMI of applicant <br>
Employment_Info_1-6	A set of normalized variables relating to the employment history of the applicant. <br>
InsuredInfo_1-6	A set of normalized variables providing information about the applicant. <br>
Insurance_History_1-9	A set of normalized variables relating to the insurance history of the applicant.<br>
Family_Hist_1-5	A set of normalized variables relating to the family history of the applicant.<br>
Medical_History_1-41	A set of normalized variables relating to the medical history of the applicant.<br>
Medical_Keyword_1-48	A set of dummy variables relating to the presence of/absence of a medical keyword being associated with the application.<br>
Response	This is the target variable, an ordinal variable relating to the final decision associated with an application<br>

#### The following variables are all categorical (nominal):
Product_Info_1, Product_Info_2, Product_Info_3, Product_Info_5, Product_Info_6, Product_Info_7, Employment_Info_2, Employment_Info_3, Employment_Info_5, InsuredInfo_1, InsuredInfo_2, InsuredInfo_3, InsuredInfo_4, InsuredInfo_5, InsuredInfo_6, InsuredInfo_7, Insurance_History_1, Insurance_History_2, Insurance_History_3, Insurance_History_4, Insurance_History_7, Insurance_History_8, Insurance_History_9, Family_Hist_1, Medical_History_2, Medical_History_3, Medical_History_4, Medical_History_5, Medical_History_6, Medical_History_7, Medical_History_8, Medical_History_9, Medical_History_11, Medical_History_12, Medical_History_13, Medical_History_14, Medical_History_16, Medical_History_17, Medical_History_18, Medical_History_19, Medical_History_20, Medical_History_21, Medical_History_22, Medical_History_23, Medical_History_25, Medical_History_26, Medical_History_27, Medical_History_28, Medical_History_29, Medical_History_30, Medical_History_31, Medical_History_33, Medical_History_34, Medical_History_35, Medical_History_36, Medical_History_37, Medical_History_38, Medical_History_39, Medical_History_40, Medical_History_41

#### The following variables are continuous:
Product_Info_4, Ins_Age, Ht, Wt, BMI, Employment_Info_1, Employment_Info_4, Employment_Info_6, Insurance_History_5, Family_Hist_2, Family_Hist_3, Family_Hist_4, Family_Hist_5

#### The following variables are discrete:
Medical_History_1, Medical_History_10, Medical_History_15, Medical_History_24, Medical_History_32

Medical_Keyword_1-48 are dummy variables.

#### Method
1. One-hot encode all the variables, 
2. dropping dummies with very low counts (i.e. 1's less than 10) - in total ~300 variables.

#### Tips from Kaggle Forum
1. One feature that seemed to help was a count of the number of medical keywords
2. The product of BMI and Ins_Age also seemed helpful whereas the ratio did not, which is not the result I expected.
3. Count the number of NA's row-wise.
4. Get rid of Medical_History_10 and Medical_History_24.
5. One shot encoding for Product_Info_2.
6. Count the number of Medical_keywords_* row-wise.
7. BMI * Ins_Age, row-wise.
8. Clusterize the whole data (join train & test, impute the missing values with the mean in these data, apply kmeans) and provide the clusters row-wise.
9. Optimize the cut-offs using 'optim'.

10. 0.002 improvement by optimizing cut-points using a genetic algorithm, compared to using this offset function. (Python code)

#### Training Methods
1. Genetic programming
2. Xgboost
3. nnet
4. mlr
5. linear regression
6. 

#### Results
1. Raw + tsne + kmean + -1 null - 0.545304983686086/0.565425847292581
2. Raw + -1 null - 0.547655889464964/0.546402606859081
3. Encoded + tsne + kmean + median null - 0.552005910930349/0.554265566239907
4. Encoded + tsne + kmean + -1 null - 0.552686232725092/0.556587256317558
5. Benchmark - 0.525480640019559/0.671169859915566 