setwd('/Users/ivanliu/Downloads/Prudential-Life-Insurance-Assessment')
library(data.table)

a <- fread('submission/submission_xgb_stack_20160201_1.csv', data.table = F)
b <- fread('submission/xgb_offset_submission_2.csv', data.table = F)

head(a)
head(b)
c <- a
c$Response <- round((a$Response + b$Response)/2)

range(c$Response)
table(a$Response)
table(b$Response)
table(c$Response)

write.csv(c, 'submission/submission_xgb_benchmark_20160203_1.csv', row.names = F)
