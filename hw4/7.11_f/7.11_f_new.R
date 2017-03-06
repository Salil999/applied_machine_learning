dataset<-read.csv('/home/jjames34/7.11_f/abalone.data', header=FALSE)
#install.packages("glmnet", dependencies=TRUE)
library(glmnet)
library(plyr)

# X and Y for the dataset in Part A.
partA_X = as.matrix(dataset[2:8])
partA_y = as.matrix(dataset[9])

# Construct a New Dataset with the gender made numeric.
bgender_dataset  = dataset
bgender_dataset$scode <- mapvalues(bgender_dataset$V1, from = c("M", "F", "I"), to = c(1, -1, 0))
bgender_dataset[1] = as.numeric(bgender_dataset$scode)
bgender_dataset$scode <- NULL

#X and Y for Part B:
partB_X = as.matrix(bgender_dataset[1:8])
partB_y = as.matrix(bgender_dataset[9])

#X and Y for Part C:
partC_X = as.matrix(dataset[2:8])
partC_y = as.matrix(log(dataset[9]))

#X and Y for Part D:
partD_X = as.matrix(bgender_dataset[1:8])
partD_y = as.matrix(log(bgender_dataset[9]))


# Perform Cross validation and plot results.
cvob1 = cv.glmnet(partA_X, partA_y)
plot(cvob1)
print(cvob1$lambda.min)

cvob2 = cv.glmnet(partB_X, partB_y)
plot(cvob2)
print(cvob2$lambda.min)

cvob3 = cv.glmnet(partC_X, partC_y)
plot(cvob3)
print(cvob3$lambda.min)

cvob4 = cv.glmnet(partD_X, partD_y)
plot(cvob4)
print(cvob4$lambda.min)






