################################################################################
set.seed(1)

## --- 1) Simulate base data: 20,000 points, binary labels, Gaussian X|Y ---
make_data <- function(N = 20000, d = 2, sep = 2.5, pi1 = 0.5) {
  y <- rbinom(N, 1, pi1)                       # labels in {0,1}
  mu0 <- rep(-sep/2, d); mu1 <- rep(sep/2, d)  # class means
  X <- matrix(NA_real_, N, d)
  X[y == 0, ] <- sweep(matrix(rnorm(sum(y == 0) * d), ncol = d), 2, mu0, `+`)
  X[y == 1, ] <- sweep(matrix(rnorm(sum(y == 1) * d), ncol = d), 2, mu1, `+`)
  data.frame(X, y = y)
}

## --- 2) Helpers: tweak-one prior and classwise resampling ---
# Tweak-one prior: class 1 gets rho, class 0 gets (1 - rho)
tweak_one_prior <- function(rho) c(`0` = 1 - rho, `1` = rho)

# Sample with replacement inside each class to keep p(X|Y) unchanged
sample_by_prior <- function(df, n, prior) {
  stopifnot(abs(sum(prior) - 1) < 1e-8)
  n0 <- rbinom(1, n, prob = prior["1"])        # number of class-1 draws
  n1 <- n0; n0 <- n - n1                       # rename: n0 for class 0, n1 for class 1
  i0 <- sample(which(df$y == 0), n0, replace = TRUE)
  i1 <- sample(which(df$y == 1), n1, replace = TRUE)
  df[c(i0, i1), , drop = FALSE]
}

## --- 3) Generate everything (minimal) ---
base <- make_data(N = 20000, d = 2, sep = 2.8, pi1 = 0.5)   # pool with base prior 0.5

# Choose sizes (change if you like)
n_train <- 10000
n_test  <- 5000
rho     <- 0.75  # tweak-one: target P(Y*=1) = rho

# Train set with base prior (resample by class from the pool)
train <- sample_by_prior(base, n = n_train, prior = c(`0`=0.5, `1`=0.5))

# Tweak-one shifted test set
pi_tweak <- tweak_one_prior(rho)
test_tweak <- sample_by_prior(base, n = n_test, prior = pi_tweak)

## --- Quick checks ---
emp_pi <- function(y) c(`0` = mean(y == 0), `1` = mean(y == 1))
cat("Empirical priors (train):   ", round(emp_pi(train$y), 3), "\n")
cat("Target tweak-one prior:     ", round(pi_tweak, 3), "\n")
cat("Empirical (tweak-one test): ", round(emp_pi(test_tweak$y), 3), "\n")

# Peek
head(train)
head(test_tweak)
################################################################################

# ===============================
# Classification tree on tweak-one shift
# ===============================
set.seed(1)
suppressPackageStartupMessages({
  library(rpart)
  library(ggplot2)
  library(dplyr)
})

# Make sure labels are factors
train$y      <- factor(train$y, levels = c(0,1))
test_tweak$y <- factor(test_tweak$y, levels = c(0,1))

# --- Train the decision tree ---
tree <- rpart(
  y ~ X1 + X2,
  data = train,
  method = "class",
  control = rpart.control(cp = 0.001, minbucket = 20, maxdepth = 10)
)

# --- Predict probabilities on tweak-one test ---
p_test <- predict(tree, newdata = test_tweak, type = "prob")[, "1"]

# Quick metrics
acc   <- mean((p_test >= 0.5) == (test_tweak$y == "1"))
brier <- mean((p_test - as.numeric(test_tweak$y))^2)
cat(sprintf("Test accuracy (0.5 threshold): %.3f | Brier: %.3f\n", acc, brier))

# --- Calibration plot ---
calibration_df <- function(y_true_factor, p_hat, nbins = 10) {
  tibble(y = as.integer(as.character(y_true_factor)),
         p = p_hat) |>
    mutate(bin = cut(p, breaks = seq(0,1,length.out = nbins+1), include.lowest = TRUE)) |>
    group_by(bin) |>
    summarise(avg_p = mean(p),
              frac_pos = mean(y),
              n = n(),
              .groups = "drop")
}

cal_df <- calibration_df(test_tweak$y, p_test, nbins = 12)

ggplot(cal_df, aes(avg_p, frac_pos)) +
  geom_abline(slope = 1, intercept = 0, linetype = 2) +
  geom_point(size = 2) +
  geom_line() +
  coord_equal(xlim = c(0,1), ylim = c(0,1)) +
  labs(title = "Calibration plot (Decision Tree on tweak-one test)",
       x = "Average predicted probability",
       y = "Empirical positive rate") +
  theme_minimal()

#################################################################################################################################

# ===============================
# Black Box Shift Estimation (BBSE)
# ===============================

# 1) Get tree outputs on train/test
p_train <- predict(tree, newdata = train,      type = "prob")[, "1"]  # P_hat(Y=1|X) on train
p_test  <- predict(tree, newdata = test_tweak, type = "prob")[, "1"]  # P_hat(Y=1|X) on test

y_tr <- as.integer(as.character(train$y))       # 0/1
y_te <- as.integer(as.character(test_tweak$y))  # 0/1

# ---------- BBSE (hard) ----------
# Build confusion matrix C[i,j] = P(ŷ=i | y=j) on source (train),
# where ŷ = 1(p >= 0.5). Then solve C * pi*_hat = mu_hat, with mu_hat = P(ŷ=i) on target (test).
bbse_hard <- function(y_true, p_train, p_test, thr = 0.5) {
  yhat_tr <- as.integer(p_train >= thr)  # 0/1
  yhat_te <- as.integer(p_test  >= thr)
  
  C <- matrix(0, nrow = 2, ncol = 2)     # rows: yhat in {0,1}, cols: y in {0,1}
  for (j in 0:1) {
    idx <- which(y_true == j)
    C[1, j+1] <- mean(yhat_tr[idx] == 0)
    C[2, j+1] <- mean(yhat_tr[idx] == 1)
  }
  mu_hat <- c(mean(yhat_te == 0), mean(yhat_te == 1))  # target ŷ marginals
  
  # Solve; clip to simplex
  pi_hat <- as.numeric(solve(C, mu_hat))
  pi_hat[pi_hat < 0] <- 0
  if (sum(pi_hat) == 0) pi_hat <- c(0.5, 0.5) else pi_hat <- pi_hat / sum(pi_hat)
  names(pi_hat) <- c("P(Y*=0)", "P(Y*=1)")
  list(pi_hat = pi_hat, C = C, mu_hat = mu_hat, detC = det(C))
}

# ---------- BBSE (soft) ----------
# Use soft confusion: C_soft[i,j] = E[ p_hat(i|X) | y=j ], and mu_hat_soft = E[ p_hat(i|X) ] on target.
# Often more stable than hard thresholding.
bbse_soft <- function(y_true, p_train, p_test) {
  # For binary, define probs for class 0 and 1
  s1_tr <- p_train; s0_tr <- 1 - p_train
  s1_te <- p_test;  s0_te <- 1 - p_test
  
  C <- matrix(0, nrow = 2, ncol = 2)
  for (j in 0:1) {
    idx <- which(y_true == j)
    C[1, j+1] <- mean(s0_tr[idx])  # E[p(0|X) | y=j]
    C[2, j+1] <- mean(s1_tr[idx])  # E[p(1|X) | y=j]
  }
  mu_hat <- c(mean(s0_te), mean(s1_te))  # E_target[p(0|X)], E_target[p(1|X)]
  
  pi_hat <- as.numeric(solve(C, mu_hat))
  pi_hat[pi_hat < 0] <- 0
  if (sum(pi_hat) == 0) pi_hat <- c(0.5, 0.5) else pi_hat <- pi_hat / sum(pi_hat)
  names(pi_hat) <- c("P(Y*=0)", "P(Y*=1)")
  list(pi_hat = pi_hat, C = C, mu_hat = mu_hat, detC = det(C))
}

# Run both estimators
bbseH <- bbse_hard(y_tr, p_train, p_test, thr = 0.5)
bbseS <- bbse_soft(y_tr, p_train, p_test)

# Ground-truth prevalence on the (labeled) test for reference
true_prev_test <- mean(y_te == 1)

cat("\n=== BBSE results ===\n")
cat(sprintf("True  P(Y*=1) on test:       %.4f\n", true_prev_test))
cat(sprintf("BBSE-hard  estimate:          %.4f   (det(C)=%.4g)\n",
            bbseH$pi_hat["P(Y*=1)"], bbseH$detC))
cat(sprintf("BBSE-soft  estimate:          %.4f   (det(C_soft)=%.4g)\n",
            bbseS$pi_hat["P(Y*=1)"], bbseS$detC))

#####################################################################################################################################
# ===============================
# Prior correction using BBSE estimate
# ===============================

# Training prior (source distribution)
pi1_train <- mean(y_tr == 1)

# Take the BBSE-soft estimate (usually more stable)
pi1_hat_test <- bbseS$pi_hat["P(Y*=1)"]

# Function: adjust probabilities using prior shift formula
adjust_posteriors <- function(p, pi1_train, pi1_hat_test) {
  num <- p * (pi1_hat_test / pi1_train)
  den <- num + (1 - p) * ((1 - pi1_hat_test) / (1 - pi1_train))
  pmin(pmax(num / den, 1e-8), 1 - 1e-8)
}

p_test_adj <- adjust_posteriors(p_test, pi1_train, pi1_hat_test)

# --- Calibration plots before & after ---
cal_df_before <- calibration_df(test_tweak$y, p_test, nbins = 12)
cal_df_after  <- calibration_df(test_tweak$y, p_test_adj, nbins = 12)

p1 <- ggplot(cal_df_before, aes(avg_p, frac_pos)) +
  geom_abline(slope=1, intercept=0, linetype=2) +
  geom_point(size=2) + geom_line() +
  coord_equal(xlim=c(0,1), ylim=c(0,1)) +
  labs(title="Before correction (tree on tweak-one test)",
       x="Average predicted probability", y="Empirical positive rate") +
  theme_minimal()

p2 <- ggplot(cal_df_after, aes(avg_p, frac_pos)) +
  geom_abline(slope=1, intercept=0, linetype=2) +
  geom_point(size=2) + geom_line() +
  coord_equal(xlim=c(0,1), ylim=c(0,1)) +
  labs(title="After BBSE prior correction",
       x="Average predicted probability", y="Empirical positive rate") +
  theme_minimal()

print(p1)
print(p2)
