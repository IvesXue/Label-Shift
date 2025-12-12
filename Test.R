library(nnet)

# Get functions from files we did before
source("BBSE.R")
source("MLLS.R")

# Function to make the source data
make_source <- function(n, b) {
  X1 <- cbind(rnorm(n/2, mean = 0), rnorm(n/2, mean = 0))
  Y1 <- rep(1, n/2)
  X2 <- cbind(rnorm(n/2, mean = b), rnorm(n/2, mean = b))
  Y2 <- rep(2, n/2)
  X <- rbind(X1, X2)
  Y <- factor(c(Y1, Y2))
  data.frame(Y = Y, x1 = X[,1], x2 = X[,2])
}

# create target
make_target_from_source <- function(source, n, q) {
  # q = P_T(Y=1)
  target_counts <- round(c(q, 1-q) * n) # count # obs for each target class
  rows_1 <- which(source$Y == 1)
  rows_2 <- which(source$Y == 2)
  # replacement/without
  if (target_counts[1] >= target_counts[2]) { # when 1>2, pick 1 with replacement
    index_1 <- sample(rows_1, size = target_counts[1], replace = TRUE)
    index_2 <- sample(rows_2, size = target_counts[2], replace = FALSE)
  } else { # when 2 >1, pick 2 with replacement
    index_1 <- sample(rows_1, size = target_counts[1], replace = FALSE)
    index_2 <- sample(rows_2, size = target_counts[2], replace = TRUE)
  }
  target_index <- c(index_1, index_2)
  target <- source[target_index, ]
  target[sample(nrow(target)), ]
}

# MSE vectror function
mse_vec <- function(w_hat, w_true) {
  mean((as.numeric(w_hat) - as.numeric(w_true))^2)
}

# One run that returns MSEs for BBSE and MLLS, for given (n, b, q)
one_run_mse <- function(n, b, q, train_prop = 0.9, delta = 0.1, L = 2, seed = 1) {
  set.seed(seed)
  source_data <- make_source(n, b)
  target_data <- make_target_from_source(source_data, n, q)
  
  # True weights: w = q/p, p = 0.5 here
  p <- c(0.5, 0.5)
  w_true <- c(q/p[1], (1-q)/p[2])
  
  # BBSE
  set.seed(seed)
  bbse_out <- BBSE(
    source = source_data,
    target = target_data,
    delta  = delta,
    L = L,
    y_col  = "Y",
    train_prop = train_prop
  )
  w_hat_bbse <- bbse_out$weights_class
  
  # MLLS(used the same training set as source with BBSE)
  set.seed(seed)
  split_out <- data_split(source_data, size = nrow(source_data), train_prop = train_prop)
  train_set <- split_out$train
  prep <- MLLS_prepare(train_set, target_data)
  res  <- MLLS_EM(prep$p_source, prep$pred_target)
  w_hat_mlls <- res$w
  list(
    mse_bbse = mse_vec(w_hat_bbse, w_true),
    mse_mlls = mse_vec(w_hat_mlls, w_true)
  )
}

# Repeat & Take average
avg_mse <- function(n, b, q, train_prop, delta, L, R, seed0 = 1000) {
  mse_bbse <- numeric(R)
  mse_mlls <- numeric(R) 
  for (r in 1:R) {
    out <- one_run_mse(
      n = n, b = b, q =q,
      train_prop = train_prop, delta = delta, L= L,
      seed = seed0 + r
    )
    mse_bbse[r] <- out$mse_bbse
    mse_mlls[r] <- out$mse_mlls
  }
  
  c(mean(mse_bbse), mean(mse_mlls))
}

# global settings that not change
train_prop <- 0.9
delta <- 0.1
L <- 2

# Replications
R1 <- 30
R2 <- 30
R3 <- 10 # increase running time dramatically

# Baselines for tests, whcih is the default used in BBSE and MLLS
n_base <- 10000
b_base <- 2.0
q_base <- 0.8


# TEST 1: Separation b from 0.1 to 5 by 0.1
b_seq <- seq(0.1, 5.0, by = 0.1)

tab_sep <- data.frame(b = b_seq,  MSE_BBSE = NA_real_, #fill with NA in numeric types 
                      MSE_MLLS = NA_real_
)

for (i in seq_along(b_seq)) { # i in 1 to length(b-seq)
  b <- b_seq[i]
  mses <- avg_mse(n = n_base, b = b, q = q_base, train_prop = train_prop, delta = delta, L = L, R = R1, seed0 = 20000)
  tab_sep$MSE_BBSE[i] <- mses[1]
  tab_sep$MSE_MLLS[i] <- mses[2]
}

# Plot
plot(tab_sep$b, tab_sep$MSE_BBSE,
     type = "b", pch = 19, col = "blue",
     xlab = "Class separation b (means at (0,0) and (b,b))",
     ylab = "MSE(w_hat, w_true)",
     main = "Test 1: Weight MSE vs Class Separation")
lines(tab_sep$b, tab_sep$MSE_MLLS, # line for MLLS
      type = "b", pch = 17, col = "red", lty = 2)
legend("topleft", # put lengend
       legend = c("BBSE", "MLLS"),
       col = c("blue", "red"),
       lty = c(1, 2),
       pch = c(19, 17),
       bty = "n")


# TEST 2: Extent of shifting
# fix source p = 0.5 and vary target q from 0 to 1 by 0.02.
# Report x-axis as |p-q| scaled to [0,1] via shift = 2*|0.5-q|.
q_seq <- seq(0, 1, by = 0.02)
shift_extent <- 2 * abs(0.5 - q_seq)  # in [0,1]

tab_shift <- data.frame(shift = shift_extent,q = q_seq,MSE_BBSE = NA_real_,MSE_MLLS = NA_real_)
for (i in seq_along(q_seq)) {
  q <- q_seq[i]
  mses <- avg_mse(
    n = n_base, b = b_base, q = q,
    train_prop = train_prop, delta = delta, L = L,
    R = R2, seed0 = 30000
  )
  tab_shift$MSE_BBSE[i] <- mses[1]
  tab_shift$MSE_MLLS[i] <- mses[2]
}

# Plot
tab_shift_plot <- tab_shift[order(tab_shift$shift), ]

# force y-range to include both curves
y_all <- c(tab_shift_plot$MSE_BBSE, tab_shift_plot$MSE_MLLS)

plot(tab_shift_plot$shift, tab_shift_plot$MSE_BBSE,
     type = "b", pch = 19, col = "blue",
     ylim = range(y_all, na.rm = TRUE),
     xlab = "Shift extent = 2|0.5 − q|",
     ylab = "MSE(w_hat, w_true)",
     main = "Test 2: Weight MSE vs Label-Shift Extent")

lines(tab_shift_plot$shift, tab_shift_plot$MSE_MLLS,type = "b", pch = 17, col = "red", lty = 2)
legend("topleft",
       legend = c("BBSE", "MLLS"),
       col = c("blue", "red"),
       lty = c(1, 2),
       pch = c(19, 17),
       bty = "n")


# TEST 3: Sample size n from 1000 to 50,000 by 1000
# Here n is total obs in source and target (both size n).
# Keep train_prop fixed (0.9/0.1) throughout.

n_seq <- seq(1000, 50000, by = 1000)

tab_n <- data.frame(
  n = n_seq,
  MSE_BBSE = NA_real_,
  MSE_MLLS = NA_real_
)

for (i in seq_along(n_seq)) {
  n <- n_seq[i]
  mses <- avg_mse(
    n = n, b = b_base, q = q_base,
    train_prop = train_prop, delta = delta, L = L,
    R = R3, seed0 = 40000
  )
  tab_n$MSE_BBSE[i] <- mses[1]
  tab_n$MSE_MLLS[i] <- mses[2]
}

# Plot
plot(tab_n$n, tab_n$MSE_BBSE,
     type = "b", pch = 19, col = "blue",
     xlab = "Sample size n (source size = n, target size = n)",
     ylab = "MSE(w_hat, w_true)",
     main = "Test 3: Weight MSE vs Sample Size")

lines(tab_n$n, tab_n$MSE_MLLS, type = "b", pch = 17, col = "red", lty = 2)
legend("topright",
       legend = c("BBSE", "MLLS"),
       col = c("blue", "red"),
       lty = c(1, 2),
       pch = c(19, 17),
       bty = "n")



# Output: 3 tables
tab_sep
tab_shift
tab_n
