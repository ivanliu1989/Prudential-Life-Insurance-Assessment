library(readr)
library(Rtsne)
set.seed(42) 

# The competition datafiles are in the directory ../input
# Read competition data files:
train <- read_csv("../input/train.csv")

#select a subset of the responses.  Here, 1&2 are mostly separable from 7,8 but not from each other
showResponses = c(1,2,7,8)
train = train[train$Response %in% showResponses,]

feature.names <- names(train)[2:ncol(train)-1]
train[is.na(train)] = 0

for (f in feature.names) {
    if (class(train[[f]])=="character") {
        levels <- unique(train[[f]])
        train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    }
}

# subset the data`
h = sample(1:nrow(train),2000)
train <- train[h,]

# use t-sne to visualize the data
tsne <- Rtsne(as.matrix(train[,-1]), dims = 2, perplexity=30, max_iter = 300, check_duplicates = F)

colors = rainbow(8)
names(colors) = 1:8
plot(tsne$Y, t='n')
text(tsne$Y, labels=train$Response, col=colors[train$Response])