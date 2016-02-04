setwd('/Users/ivanliu/Downloads/Prudential-Life-Insurance-Assessment')
library(data.table)

a <- fread('submission/submission_xgb_stack_20160201_1.csv', data.table = F)
b <- fread('submission/xgb_offset_submission_2.csv', data.table = F)
c <- fread('submission/submission_single_xgb_20160204.csv', data.table = F)

head(a)
head(b)
head(c)
submit <- a
submit$Response <- round((3*a$Response + 3*b$Response + c$Response)/7)

range(submit$Response)
table(a$Response)
table(b$Response)
table(c$Response)
table(submit$Response)

write.csv(submit, 'submission/submission_blend_20160204.csv', row.names = F)
