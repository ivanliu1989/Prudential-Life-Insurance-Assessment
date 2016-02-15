setwd('/Users/ivanliu/Downloads/Prudential-Life-Insurance-Assessment')
library(data.table)

a <- fread('blending/linear_0.65002.csv', data.table = F)
b <- fread('blending/submission_20160215_nnets_0.62519.csv', data.table = F)
c <- fread('blending/submission_single_xgb_new_feat_20160204_0.66717.csv', data.table = F)

d <- fread('blending/submission_xgb_all_meta_20160211_0.66614.csv', data.table = F)
e <- fread('blending/submission_xgb_stack_20160201_1_0.67348.csv', data.table = F)
f <- fread('blending/submit_glm_0.65782.csv', data.table = F)
g <- fread('blending/submit_kappa_0.66614.csv', data.table = F)
h <- fread('blending/submit_rmse_0.66614.csv', data.table = F)
i <- fread('blending/xgb_offset_submission_2_0.67459.csv', data.table = F)

j <- fread('blending/submission_20160215_svm.csv', data.table = F)

head(a)
head(b)
head(c)
head(d)
head(e)
head(f)
head(g)
head(h)
head(i)
submit <- a

submit$Response <- round((a$Response * 1 + 
                              b$Response * 1 + 
                              c$Response * 1 + 
                              d$Response * 1 + 
                              e$Response * 2 + 
                              f$Response * 1 + 
                              g$Response * 1 + 
                              h$Response * 1 + 
                              i$Response * 2 +
                              i$Response * 2 
)/13)

range(submit$Response)
table(a$Response)
table(b$Response)
table(c$Response)
table(submit$Response)

write.csv(submit, 'submission_blend_20160215_2.csv', row.names = F)

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
