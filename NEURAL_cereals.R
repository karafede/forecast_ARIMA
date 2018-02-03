
## Creating index variable 
# https://www.analyticsvidhya.com/blog/2017/09/creating-visualizing-neural-network-in-r/

# other example 
# https://www.r-bloggers.com/fitting-a-neural-network-in-r-neuralnet-package/

# Read the Data
setwd("C:/Forecast_R")
data = read.csv("cereals.csv", header=T)

# Random sampling
samplesize = 0.60 * nrow(data)
set.seed(80)
index = sample(seq_len(nrow(data)), size = samplesize)   # 45 is the length of the traing dataset
index

# Create training and test set
datatrain = data[index, ]     # 45 training data
datatest = data[-index, ]     # 30 test sample (to use later and for the validation)


## Scale data for neural network

max = apply(data, 2, max) # 2 indicate columns
min = apply(data, 2, min)
scaled = as.data.frame(scale(data, center = min, scale = max - min))


## Fit neural network 

# install library
# install.packages("neuralnet")

# load library
library(neuralnet)

# creating training and test set
trainNN = scaled[index , ]
testNN = scaled[-index , ]

# fit neural network
set.seed(2)
NN = neuralnet(rating ~ calories + protein + fat + sodium + fiber, 
               trainNN, hidden = 3 , linear.output = T )

# plot neural network
plot(NN)


## Prediction using neural network

predict_testNN = compute(NN, testNN[,c(1:5)])
predict_testNN = (predict_testNN$net.result * (max(data$rating) - min(data$rating))) + min(data$rating)

plot(datatest$rating, predict_testNN, col='blue', pch=16, ylab = "predicted rating NN", xlab = "real rating")

abline(0,1)

# Calculate Root Mean Square Error (RMSE)
RMSE.NN = (sum((datatest$rating - predict_testNN)^2) / nrow(datatest)) ^ 0.5
# [1] 6.049574835



#################################
##### CROSS VALIDATION ##########
#################################

###  THIS IS TO CHECK HOW LONG SHOULD BE THE TRAINING SET OF DATA #####
# AFTER THIS, WE CAN RUN again the Noural Network above with 
# the right length of training data  ##################################


# k-fold cross-validation on the neural network

library(boot)
library(plyr)
library(neuralnet)

# Initialize variables
set.seed(50)
k = 100
RMSE.NN = NULL

# 65 -10 = 55 dimension of the training data
# k = 100 ---> dimension of the 
# 55 * 100 = 5500

List = list()

j <- 50
i <- 1

# Fit neural network model within nested for loop
for(j in 10:65){  # j = number of training data (sample size)
  for (i in 1:k) {
    index = sample(1:nrow(data),j)   # this is a random output (repat this 100 times)
    index   # random number
    
    trainNN = scaled[index,]
    testNN = scaled[-index,]
    datatest = data[-index,]
    
    NN = neuralnet(rating ~ calories + protein + fat + sodium + fiber, trainNN, hidden = 3, linear.output= T)
    predict_testNN = compute(NN,testNN[,c(1:5)])
    predict_testNN = (predict_testNN$net.result*(max(data$rating)-min(data$rating)))+min(data$rating)
    
    RMSE.NN[i] <- (sum((datatest$rating - predict_testNN)^2)/nrow(datatest))^0.5
  }
  List[[j]] = RMSE.NN
}

Matrix.RMSE = do.call(cbind, List)


##############################
##### RMSE ###################
#############################

# boxplot

boxplot(Matrix.RMSE[,56], ylab = "RMSE", main = "RMSE BoxPlot (length of traning set = 65)")
# boxplot(Matrix.RMSE[,30], ylab = "RMSE", main = "RMSE BoxPlot (length of traning set = 65)")

summary(Matrix.RMSE[,56])
summary(Matrix.RMSE[,30])
# median RMSE ----> 6.395486


# variation of RMSE with the length of training set

# install.packages("matrixStats")
library(matrixStats)

med = colMedians(Matrix.RMSE)

X = seq(10,65)
# X = seq(10,44)
plot (med ~ X, type = "l", xlab = "length of training set", ylab = "median RMSE", main = "Variation of RMSE with length of training set")


######### IMPORTANT #################
#####################################

# the median RMSE of our model decreases as the length of the training the set.
# This is an important result. The reader must remember that the model accuracy is 
# dependent on the length of training set. The performance of neural network model is
# sensitive to training-test split.