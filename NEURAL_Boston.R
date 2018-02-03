

# https://www.r-bloggers.com/fitting-a-neural-network-in-r-neuralnet-package/
  
set.seed(500)
library(MASS)
data <- Boston

# The Boston dataset is a collection of data about housing values in the suburbs
# of Boston. Our goal is to predict the median value of owner-occupied homes (medv)
# using all the other continuous variables available.

index <- sample(1:nrow(data),round(0.75*nrow(data)))
# traininh data (75% of the original data)
train <- data[index,]

# test data (25% of the original data)
test <- data[-index,]

### make a regular multilienar regression with the test data
lm.fit <- glm(medv~., data=train)
summary(lm.fit)

# calculate predicted values
pr.lm <- predict(lm.fit,test)
# calculate the Mean Squared Error
MSE.lm <- sum((pr.lm - test$medv)^2)/nrow(test)

#######################
# neural network ######
#######################

# scale data

maxs <- apply(data, 2, max) 
mins <- apply(data, 2, min)

# normalize the data
scaled <- as.data.frame(scale(data, center = mins, scale = maxs - mins))

train_ <- scaled[index,]
test_ <- scaled[-index,]

# As far as I know there is no fixed rule as to how many layers and neurons
# to use although there are several more or less accepted rules of thumb. 
# Usually, if at all necessary, one hidden layer is enough for a vast numbers 
# of applications. As far as the number of neurons is concerned, it should be between
# the input layer size (input variables) and the output layer size,
# usually 2/3 of the input size.

# we are going to use 2 hidden layers with this configuration: 13:5:3:1. 
# The input layer has 13 inputs (variables), 
# the two hidden layers have 5 and 3 neurons and 
# the output layer has, of course, a single output since we are doing regression.

library(neuralnet)
n <- names(train_)
# write a multilinear regression formula
f <- as.formula(paste("medv ~", paste(n[!n %in% "medv"], collapse = " + ")))
# use normalized training data
nn <- neuralnet(f,data=train_,hidden=c(5,3),linear.output=T)
# nn <- neuralnet(f,data=train_,hidden=c(8),linear.output=T)
plot(nn)

# The black lines show the connections between each layer and the weights
# on each connection while the blue lines show the bias term added in each step. 
# The bias can be thought as the intercept of a linear model.

# The net is essentially a black box so we cannot say that much about the fitting, 
# the weights and the model. Suffice to say that the training algorithm has converged and 
# therefore the model is ready to be used.


###################################################
### prediction with test data #####################
###################################################

# used normalized test data
pr.nn <- compute(nn,test_[,1:13])

# Remember that the net will output a normalized prediction, 
# so we need to scale it back in order to make a meaningful comparison
# (or just a simple prediction).

# only use the net result for the prediction analyis
# undo the normalization
pr.nn_ <- pr.nn$net.result*(max(data$medv)-min(data$medv))+min(data$medv)

# original data not normalized
test.r <- (test_$medv)*(max(data$medv)-min(data$medv))+min(data$medv)

# mean squared error
MSE.nn <- sum((test.r - pr.nn_)^2)/nrow(test_)
# [1] 13.55698822
#### .....to be compared with the MSE obtained from the multilienar regression
MSE.lm <- sum((pr.lm - test$medv)^2)/nrow(test)
# [1] 21.62975935

print(paste(MSE.lm,MSE.nn))


# PLOTS #######

par(mfrow=c(1,2))

plot(test$medv,pr.nn_,col='red',main='Real vs predicted NN',pch=18,cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend='NN',pch=18,col='red', bty='n')

plot(test$medv,pr.lm,col='blue',main='Real vs predicted lm',pch=18, cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend='LM',pch=18,col='blue', bty='n', cex=.95)

dev.off()

plot(test$medv,pr.nn_,col='red',main='Real vs predicted NN',pch=18,cex=0.7)
points(test$medv,pr.lm,col='blue',pch=18,cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend=c('NN','LM'),pch=18,col=c('red','blue'))

####################################
#### CROSS VALIDATION ##############
####################################

library(boot)
set.seed(200)

# generalized linear model (multilinear regression)
lm.fit <- glm(medv~.,data=data)
summary(lm.fit)
# Calculate the estimated K-fold cross-validation prediction error 
# for generalized linear models
# Here is the 10 fold cross validated MSE for the linear model:
cv.glm(data,lm.fit,K=10)$delta[1]


###  THIS IS TO CHECK HOW LONG SHOULD BE THE TRAINING SET OF DATA #####
#### AFTER THIS, WE CAN RUN THE ABOVE SCRIPT ##########################


# k-fold cross-validation on the neural network

set.seed(450)
# cv.error <- NULL
RMSE.NN = NULL
k <- 10


library(plyr) 
pbar <- create_progress_bar('text')
pbar$init(k)


j <- 50
i <- 1

for(i in 1:k){
  index <- sample(1:nrow(data),round(0.9*nrow(data)))   # RANDOM 90% of the data
  index
  train.cv <- scaled[index,]
  test.cv <- scaled[-index,]
  
  nn <- neuralnet(f,data=train.cv,hidden=c(5,2),linear.output=T)
  
  pr.nn <- compute(nn,test.cv[,1:13])
  pr.nn <- pr.nn$net.result*(max(data$medv)-min(data$medv))+min(data$medv)
  
  test.cv.r <- (test.cv$medv)*(max(data$medv)-min(data$medv))+min(data$medv)
  
  RMSE.NN[i] <- sum((test.cv.r - pr.nn)^2)/nrow(test.cv)
  
  pbar$step()
}


mean(RMSE.NN)

boxplot(RMSE.NN,xlab='MSE CV',col='cyan',
        border='blue',names='CV error (MSE)',
        main='CV error (MSE) for NN',horizontal=TRUE)

