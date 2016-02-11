setwd('/Users/ivanliu/Downloads/Prudential-Life-Insurance-Assessment')
library(data.table)

a <- fread('submission/submission_blend_20160204_0.67022.csv', data.table = F)
b <- fread('submission/submission_blend_20160204_0.67628.csv', data.table = F)
c <- fread('submission/submission_single_xgb_new_feat_20160204_0.66717.csv', data.table = F)

d <- fread('submission/submission_xgb_benchmark_20160203_1_0.67616.csv', data.table = F)
e <- fread('submission/submission_xgb_meta_leaf_rsme_kappa_20160210_0.66811.csv', data.table = F)
f <- fread('submission/submission_xgb_stack_20160201_1_0.67348.csv', data.table = F)
g <- fread('submission/submission_xgb_stack_20160202_1_0.66462.csv', data.table = F)
h <- fread('submission/submission_xgb_stack_20160210_1_0.66570.csv', data.table = F)
i <- fread('submission/submission_xgb_stack_20160210_2_meta_0.66449.csv', data.table = F)
j <- fread('submission/xgb_offset_submission_2_0.67459.csv', data.table = F)

head(a)
head(b)
head(c)
head(d)
head(e)
head(f)
head(g)
head(h)
head(i)
head(j)
submit <- a

submit$Response <- round((a$Response * 2 + 
                              b$Response * 2 + 
                              c$Response * 1 + 
                              d$Response * 2 + 
                              e$Response * 1 + 
                              f$Response * 2 + 
                              g$Response * 1 + 
                              h$Response * 1 + 
                              i$Response * 1 + 
                              j$Response * 2
)/15)

range(submit$Response)
table(a$Response)
table(b$Response)
table(c$Response)
table(submit$Response)

write.csv(submit, 'submission_blend_20160210_10model_2t1.csv', row.names = F)
