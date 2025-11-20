library(nnet)
# Data 
# here I just simulate data by hand to check our BBSE functions
# We have our source data with 10,000 obs and target data with 10,000
# Source data is splited to training and validation, where training use to train model and validation here use to compute confusion matrix
# Each sub data set has Y as labels from 1 to 10, and X as 2-D data X1 and X2
# For target data set, I am planning to use tweak-one shift, making label 5 occupies 55% and rest occupies 45%/9 = 5%. 

set.seed(1834)
n <- 10000

# Class 1 with mean (0,0)
X1 <- cbind(rnorm(n/2, mean = 0),rnorm(n/2, mean = 0))
Y1 <- rep(1, n/2)

# Class 2 with mean (2,2), which is not that seperate, but enough for a good model
X2 <- cbind(rnorm(n/2, mean = 2),rnorm(n/2, mean = 2))
Y2 <- rep(2, n/2)

# Combine into one dataset
X <- rbind(X1, X2)
Y <- factor(c(Y1, Y2))

source <- data.frame(Y = Y, x1 = X[,1], x2 = X[,2])
head(source)

plot(X)

# Prepare Target data set from Source data set
target_probs <- rep(0.2, 2)  # start with 0.2 for both
target_probs[1] <- 0.8# label 1 gets 80%, label 2 stays 20%
target_counts <- round(target_probs * n)
target_counts

#For label 1: sample with replacement
rows_1 <- which(source$Y == 1)
index_1 <- sample(
  rows_1,
  size    = target_counts[1],
  replace = TRUE 
)

# For label 2 sample without replacement
rows_2 <- which(source$Y == 2)
index_2 <- sample(
  rows_2,
  size    = target_counts[2],
  replace = FALSE
)


# Combine indices and build target dataset
target_index <- c(index_1, index_2)
target <- source[target_index, ]
target <- target[sample(nrow(target)), ]

plot(source$Y)
plot(target$Y)
################################################################################
# BBSE

# Helper functions

# Data split
data_split <- function(source, size, train_prop = 0.5) { #input source data set, size of source and the proportion we want goes to train
  sampled_index <- sample(seq_len(nrow(source)), size = size, replace = FALSE) # sample rows from source with n rows
  
  train_size <- floor(train_prop * size) # training size based on prop
  train_index <- sample(sampled_index, size = train_size, replace = FALSE) # sample training rows
  
  validation_index <- setdiff(sampled_index, train_index) # left outs being validation
  
  train      <- source[train_index, ]
  validation <- source[validation_index, ]
  
  return(list(
    train = train,  # Output train and validation
    validation = validation
  ))
}

# check
test1 <- data_split(source, 10000)
train <- test1$train
validation <-test1$validation


# Confusion Matrix
confusion_matrix <- function(validation, L, model, y_col = "Y") { # validation = validation set, L = number of labels, model is trained model aribitary, y_col is the name of the label colmn

  true_y <- validation[[y_col]] # Separate out true labels
  true_y <- factor(true_y, levels = 1:L) # make sure true y is categorical
  
  feature_cols <- setdiff(colnames(validation), y_col) # left out being features
  
  pred_y <- predict(model, newdata = validation[, feature_cols, drop = FALSE], type = "class") # predict_yhat
  pred_y <- factor(pred_y, levels = 1:L) # same as above
  
  C <- matrix(0, nrow = L, ncol = L) # let C be l*l matrix
  rownames(C) <- 1:L  # predicted i
  colnames(C) <- 1:L  # true j
  
  for (i in 1:L) {
    for (j in 1:L) {
      C[i, j] <- sum(pred_y == i & true_y == j) / nrow(validation) # compute every Cij
    }
  }
  return(C) # output confusion matrix
}

# check 
train$Y      <- as.factor(train$Y) # make all labels factor, since we need all response categorical
validation$Y <- as.factor(validation$Y)
mutilog <- multinom( Y ~ x1 + x2, data = train)
test2 <- confusion_matrix(validation, 2, mutilog)
print(test2)

# Steps for main function:
# Input: Source data set P and target data set Q, model class F, and a hyperparameter delta between 0 and 1/L, where L is the number of labels we have
#1. use data_split to split source data into training and validation data
#2. Use training data to train a classifier f that is from model class F
#3. use the validation data and trained model f to get the confusion matrix C, using helper function confusion_matrix
#4. if the eigenvalue of C is less than or eqauls to delta, set w = 1; if not, estimate w = solve(C)*mu_hat, where mu_hat is a vector that mu_hat[i] = (number of predicted labels = i on target data set using model f)/nrows(target data set).
#5. Use the weight to re-weight training model(if we use multinomial logistic regression, use the weight = w inside the function)
#6. Output weighted model f_corrected

# Main function

BBSE <- function(source, # source data set
                 target, # target data set
                 delta, # must between 0 and 1/L
                 L,# Number of labels
                 y_col      = "Y", # label column name for source
                 train_prop = 0.5) { #Proportion for train/validation split

# Check delta
  if (delta < 0 || delta > 1 / L) {warning("delta needs to be in [0, 1/k].")}
  
  # data split
  split_out <- data_split(source, size = nrow(source), train_prop = train_prop)
  train      <- split_out$train
  validation <- split_out$validation
  train[[y_col]]      <- factor(train[[y_col]],      levels = 1:L) # make sure labels are factors
  validation[[y_col]] <- factor(validation[[y_col]], levels = 1:L)
  
  # Train model f
  f <- multinom( Y ~ x1 + x2, data = train)
  
  # Compute confusion matrix C
  C <- confusion_matrix(validation, L = L, model = f, y_col = y_col)
  
  # Decide whether use 1 or w
  eig_vals <- eigen(C, only.values = TRUE)$values # only get eigenvalues
  lambda_min <- min(Re(eig_vals))  # Take the minimum, Re() just in case it's not real
  if (lambda_min <= delta) {
    w <- rep(1, L)
    names(w) <- as.character(1:L)
  } else {
    feature_cols <- setdiff(colnames(target), y_col) # only use features
    y_hat_target <- predict(f, newdata = target[, feature_cols, drop = FALSE], type = "class") # predict labels for target
    y_hat_target <- factor(y_hat_target, levels = 1:L) 
    mu_counts <- table(y_hat_target) # count number of each label l
    
    mu_hat <- numeric(L) # build a vector with L 0s
    names(mu_hat) <- as.character(1:L)
    mu_hat[names(mu_counts)] <- as.numeric(mu_counts) / length(y_hat_target) # fill in mu_hat with mu_hat[i] = (number of predicted labels = i on target data set using model f)/nrows(target data set).
    
    # Solve out w by C^-1*mu_hat
    w <- solve(C, mu_hat)
    names(w) <- as.character(1:L)
  }

  #With w, now we can perform Bayes Rule calibration
  
  #Get P(Y|X)
  prob_S <- predict(f, newdata = target, type = "probs") # Predict P(Y=2|X)
  prob_mat <- cbind(1-prob_S, prob_S)
  colnames(prob_mat) <- levels(source$Y)
  
  #Multiply P(Y=i|X) by w[i]
  mat2 <- sweep(prob_mat, 2, w, "*")
  
  # Get constant C(X)
  C_X <- 1 / rowSums(mat2)
  
  # estimate Q(Y|X)
  prob_T <- mat2*C_X
  
  # Outout
  return(list(weights_class = w,   # w[i] is weight for class i
              C = C, #confusion matrix
              prob_S = prob_mat,
              prob_T = prob_T,
              f = f
  ))
}
###################################################################################
# test
test3 <- BBSE(source, # source data set
              target, # target data set
              0.1, # must between 0 and 1/L
              2,# Number of labels
              y_col      = "Y", # label column name for source
              train_prop = 0.5)

w <- test3$weights_class
f <- test3$f
prob_T = test3$prob_T


# Calibration and Brier score
# Raw probabilities on target
prob_raw <- predict(f, newdata = target, type = "probs")

# Take P(Y=1|X) for raw and adjusted
p_raw <- prob_raw_mat[, "1"]
p_adj <- prob_T[, "1"]

# True labels as 0 and 1
y_true <- as.numeric(target$Y == "1")

#Calibration points
bins_raw <- cut(p_raw, breaks = seq(0, 1, length.out = 11), include.lowest = TRUE)
pred_mean_raw <- tapply(p_raw,  bins_raw, mean)
obs_mean_raw  <- tapply(y_true, bins_raw, mean)

bins_adj <- cut(p_adj, breaks = seq(0, 1, length.out = 11), include.lowest = TRUE)
pred_mean_adj <- tapply(p_adj,  bins_adj, mean)
obs_mean_adj  <- tapply(y_true, bins_adj, mean)

# Raw
plot(pred_mean_raw, obs_mean_raw,
     type = "b", pch = 19,
     xlim = c(0, 1), ylim = c(0, 1),
     xlab = "Predicted P(Y=1)", ylab = "Observed P(Y=1)",
     main = "Calibration (Raw)")
abline(0, 1, lty = 2)

# Adjusted
plot(pred_mean_adj, obs_mean_adj,
     type = "b", pch = 19,
     xlim = c(0, 1), ylim = c(0, 1),
     xlab = "Predicted P(Y=1)", ylab = "Observed P(Y=1)",
     main = "Calibration (Adjusted)")
abline(0, 1, lty = 2)


# Brier scores
brier_raw <- mean((p_raw - y_true)^2)
brier_adj <- mean((p_adj - y_true)^2)

brier_raw
brier_adj
