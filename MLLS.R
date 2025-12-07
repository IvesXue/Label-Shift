# Same data and set up

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


# MLLS

# Prepare everything needs for EM
MLLS_prepare <- function(source, target) {
  # p(y = i)
  p_source <- prop.table(table(source$Y))
  
  # train classifier on source
  f <- multinom(Y ~ x1 + x2, data = source)
  
  # predicted P(y=i|x) on target
  p2 <- as.numeric(predict(f, newdata = target, type = "probs"))
  p1 <- 1 - p2
  
  # make it matrix: col1 = p(Y=1), col2 = P(Y=2)
  pred_target <- cbind(p1, p2)
  colnames(pred_target) <- levels(source$Y)
  
  list(
    p_source    = p_source,# p(y = i )
    pred_target = pred_target, # fi(x)
    f = f
  )
}

# test
p_source <- MLLS_prepare(source, target)$p_source
pred_target <- MLLS_prepare(source, target)$pred_target
test1 <- pred_target[,1]+pred_target[,2]
test1

# EM Algorithm
MLLS_EM <- function(p_source, pred_target, tol = 1e-6) {
  p_s <- as.numeric(p_source)   # source prior
  q   <- p_s # initial guess q^0_i = p_s(y = i)
  
  repeat {
    num <- sweep(pred_target, 2, q / p_s, "*")   # N x K
    denom <- rowSums(num) 
    r     <- num / denom   

    q_new <- colMeans(r) 
    
    # stopping
    if (max(abs(q_new - q)) < tol) {
      q <- q_new
      break
    }
    
    q <- q_new
  }
  
  w <- q / p_s
  
  list(
    q = q,   # estimated q(y = i)
    w = w    #  w_i
  )
}

# Calibration and Brier

prep <- MLLS_prepare(source, target)
res  <- MLLS_EM(prep$p_source, prep$pred_target)
f <- prep$f

# Raw prob on target
prob_raw <- predict(f, newdata = target, type = "probs")

# turn into martix
p2_raw <- as.numeric(prob_raw)
p1_raw <- 1 - p2_raw
prob_raw_mat <- cbind(p1_raw, p2_raw)
colnames(prob_raw_mat) <- levels(source$Y)

# weights
w <- res$w

# Source prob distribution
prob_source_target <- prep$pred_target

# Apply weights and normalize
num   <- sweep(prob_source_target, 2, w, "*")
denom <- rowSums(num)
prob_MLLS_mat <- num / denom
prob_T <- prob_MLLS_mat


# P_raw and after MLLS
p_raw  <- prob_raw_mat[, "1"]
p_MLLS <- prob_T[, "1"]

# True labels as 0&1
y_true <- as.numeric(target$Y =="1")

# Calibration
bins_raw      <- cut(p_raw, breaks = seq(0, 1, length.out = 11), include.lowest = TRUE)
pred_mean_raw <- tapply(p_raw,  bins_raw, mean)
obs_mean_raw  <- tapply(y_true, bins_raw, mean)
bins_MLLS      <- cut(p_MLLS, breaks = seq(0, 1, length.out = 11), include.lowest = TRUE)
pred_mean_MLLS <- tapply(p_MLLS,  bins_MLLS, mean)
obs_mean_MLLS  <- tapply(y_true, bins_MLLS, mean)

# Plot
plot(pred_mean_raw, obs_mean_raw,
     type = "b", pch = 19,
     xlim = c(0, 1), ylim = c(0, 1),
     xlab = "Predicted P(Y=1)", ylab = "Observed P(Y=1)",
     main = "Calibration (Raw)")
abline(0, 1, lty = 2)

plot(pred_mean_MLLS, obs_mean_MLLS,
     type = "b", pch = 19,
     xlim = c(0, 1), ylim = c(0, 1),
     xlab = "Predicted P(Y=1)", ylab = "Observed P(Y=1)",
     main = "Calibration (MLLS Adjusted)")
abline(0, 1, lty = 2)

# Brier
brier_raw  <- mean((p_raw  - y_true)^2)
brier_MLLS <- mean((p_MLLS - y_true)^2)

brier_raw
brier_MLLS

