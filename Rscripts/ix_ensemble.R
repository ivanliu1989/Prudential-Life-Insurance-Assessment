setwd('/Users/ivanliu/Downloads/Prudential-Life-Insurance-Assessment')
library(data.table)
rm(list=ls());gc()

a <- fread('blending/linear_0.65002.csv', data.table = F)
b <- fread('blending/submission_20160215_nnets_0.62519.csv', data.table = F)
c <- fread('blending/submission_blend_20160216_weighted.csv', data.table = F)

# d <- fread('blending/', data.table = F)
e <- fread('blending/submission_xgb_stack_20160201_1_0.67348.csv', data.table = F)
f <- fread('blending/submit_glm_0.65782.csv', data.table = F)
# g <- fread('blending/', data.table = F)
# h <- fread('blending/', data.table = F)
i <- fread('blending/xgb_offset_submission_2_0.67459.csv', data.table = F)
j <- fread('blending/submission_20160215_libsvm_epsilon_rounded_0.60496.csv', data.table = F)

# head(a)
# head(b)
# head(c)
# head(d)
# head(e)
# head(f)
# head(g)
# head(h)
# head(i)
submit <- a

submit$Response <- round((a$Response * 1 + 
                              b$Response * 1 + 
                              c$Response * 2 + 
                              # d$Response * 1 + 
                              e$Response * 2 + 
                              f$Response * 1 + 
                              # g$Response * 0.5 + 
                              # h$Response * 0.5 + 
                              i$Response * 2 +
                              j$Response * 1 
)/10)

range(submit$Response)
table(a$Response)
table(b$Response)
table(e$Response)
table(submit$Response)

write.csv(submit, 'submission_blend_20160216_less_weighted.csv', row.names = F)

# nnets blending
files <- list.files('blending/nnets', full.names = T)
for (f in 1:length(files)){
    if(f == 1){
        submit <- read.csv(files[f])
    }else{
        submit$Response <- submit$Response + read.csv(files[f])[,2]
    }
}
submit$Response <- round(submit$Response/length(files))
write.csv(submit, 'submission_20160215_nnets.csv', row.names = F)
