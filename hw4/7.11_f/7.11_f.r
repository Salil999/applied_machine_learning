dataset<-read.csv('/home/jjames34/7.11_f/abalone.data', header=FALSE)
install.packages("glmnet", dependencies=TRUE)
library(glmnet)
library(plyr)

# X and Y for the dataset in Part A.
partA_X = dataset[2:8]
partA_y = dataset[9]

# Construct a New Dataset with the gender made numeric.
bgender_dataset  = dataset
bgender_dataset$scode <- mapvalues(bgender_dataset$V1, from = c("M", "F", "I"), to = c(1, -1, 0))
bgender_dataset[1] = bgender_dataset$scode
bgender_dataset$scode <- NULL

#X and Y for Part B:
partB_X = bgender_dataset[1:8]
partB_y = bgender_dataset[9]

#X and Y for Part C:
partC_X = dataset[2:8]
partC_y = log(dataset[9])

#X and Y for Part D:
partD_X = bgender_dataset[1:8]
partD_y = log(bgender_dataset[9])





