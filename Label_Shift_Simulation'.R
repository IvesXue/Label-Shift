# =========================================================
# Label Shift (tweak-one), RF + BBSE + BBSC (posterior)
# Sweep over prevalence and separability
# =========================================================
set.seed(1)
suppressPackageStartupMessages({
  library(ranger)
  library(dplyr)
  library(ggplot2)
  library(purrr)
  library(tidyr)
})

# --------------------------
# 1) Data generation
# --------------------------
make_data <- function(N = 20000, d = 2, sep = 2.8, pi1 = 0.5) {
  y  <- rbinom(N, 1, pi1)                  # labels in {0,1}
  mu0 <- rep(-sep/2, d); mu1 <- rep(sep/2, d)
  X  <- matrix(NA_real_, N, d)
  X[y == 0, ] <- sweep(matrix(rnorm(sum(y == 0) * d), ncol = d), 2, mu0, `+`)
  X[y == 1, ] <- sweep(matrix(rnorm(sum(y == 1) * d), ncol = d), 2, mu1, `+`)
  df <- data.frame(X1 = X[,1], X2 = X[,2], y = y)
  df
}

tweak_one_prior <- function(rho) c(`0` = 1 - rho, `1` = rho)

# Class-wise resampling (with replacement) to enforce label shift
sample_by_prior <- function(df, n, prior) {
  stopifnot(abs(sum(prior) - 1) < 1e-8, all(names(prior) %in% c("0","1")))
  n1 <- rbinom(1, n, prob = prior["1"])   # how many class-1 draws
  n0 <- n - n1
  i0 <- sample(which(df$y == 0), n0, replace = TRUE)
  i1 <- sample(which(df$y == 1), n1, replace = TRUE)
  df[c(i0, i1), , drop = FALSE]
}

emp_pi <- function(y) c(`0` = mean(y == 0), `1` = mean(y == 1))

# --------------------------
# 2) Model, BBSE (soft), BBSC (posterior correction)
# --------------------------
fit_rf <- function(train) {
  train$y <- factor(train$y, levels = c(0,1))
  ranger(
    y ~ X1 + X2, data = train, probability = TRUE,
    num.trees = 500, mtry = 2, min.node.size = 10, seed = 1
  )
}

predict_prob <- function(rf, data) predict(rf, data = data)$predictions[, "1"]

# Soft BBSE: use expected probs rather than hard labels
bbse_soft <- function(y_true_factor, p_train, p_test) {
  y_true <- as.integer(as.character(y_true_factor))
  s1_tr <- p_train; s0_tr <- 1 - p_train
  s1_te <- p_test;  s0_te <- 1 - p_test
  
  C <- matrix(0, 2, 2)
  for (j in 0:1) {
    idx <- which(y_true == j)
    C[1, j+1] <- mean(s0_tr[idx])      # E[p(0|X)|y=j]
    C[2, j+1] <- mean(s1_tr[idx])      # E[p(1|X)|y=j]
  }
  mu_hat <- c(mean(s0_te), mean(s1_te))
  pi_hat <- as.numeric(solve(C, mu_hat))
  pi_hat[pi_hat < 0] <- 0
  pi_hat <- pi_hat / sum(pi_hat)
  names(pi_hat) <- c("P(Y*=0)", "P(Y*=1)")
  list(pi_hat = pi_hat, C = C, mu_hat = mu_hat)
}

# Posterior correction (Saerens–Latinne–Decaestecker)
adjust_posteriors <- function(p, pi1_train, pi1_hat_test) {
  num <- p * (pi1_hat_test / pi1_train)
  den <- num + (1 - p) * ((1 - pi1_hat_test) / (1 - pi1_train))
  pmin(pmax(num / den, 1e-8), 1 - 1e-8)
}

# Calibration helpers & metrics
calibration_df <- function(y_true_factor, p_hat, nbins = 12) {
  tibble(y = as.integer(as.character(y_true_factor)), p = p_hat) |>
    mutate(bin = cut(p, breaks = seq(0,1,length.out = nbins+1), include.lowest = TRUE)) |>
    group_by(bin) |>
    summarise(avg_p = mean(p), frac_pos = mean(y), n = n(), .groups = "drop")
}
brier <- function(p, y) mean((p - as.integer(as.character(y)))^2)
ece <- function(yf, p, nbins=12){
  df <- calibration_df(yf, p, nbins)
  w <- df$n / sum(df$n); sum(w * abs(df$avg_p - df$frac_pos))
}

# --------------------------
# 3) One scenario runner
# --------------------------
run_scenario <- function(sep = 2.8, rho = 0.75,
                         N_pool = 20000, n_train = 10000, n_test = 5000,
                         plot = TRUE) {
  # Base pool and splits
  base <- make_data(N = N_pool, sep = sep, pi1 = 0.5)
  train <- sample_by_prior(base, n = n_train, prior = c(`0`=0.5, `1`=0.5))
  test  <- sample_by_prior(base, n = n_test,  prior = tweak_one_prior(rho))
  
  # Train RF
  rf <- fit_rf(train)
  p_tr <- predict_prob(rf, train)
  p_te <- predict_prob(rf, test)
  
  # BBSE (soft)
  bbseS <- bbse_soft(factor(train$y, levels = c(0,1)), p_tr, p_te)
  pi1_hat <- as.numeric(bbseS$pi_hat["P(Y*=1)"])
  pi1_tr  <- mean(train$y == 1)
  pi1_te_true <- mean(test$y == 1)
  
  # Prior-correct test probs
  p_te_adj <- adjust_posteriors(p_te, pi1_tr, pi1_hat)
  
  # Metrics
  acc_before <- mean((p_te >= 0.5) == (test$y == 1))
  acc_after  <- mean((p_te_adj >= 0.5) == (test$y == 1))
  out <- tibble(
    sep = sep, rho = rho,
    true_prev = pi1_te_true,
    bbse_hat  = pi1_hat,
    acc_before = acc_before,
    acc_after  = acc_after,
    brier_before = brier(p_te, test$y),
    brier_after  = brier(p_te_adj, test$y),
    ece_before   = ece(factor(test$y, levels=c(0,1)), p_te),
    ece_after    = ece(factor(test$y, levels=c(0,1)), p_te_adj)
  )
  
  # Plots (optional)
  if (isTRUE(plot)) {
    cal_before <- calibration_df(factor(test$y, levels = c(0,1)), p_te)
    cal_after  <- calibration_df(factor(test$y, levels = c(0,1)), p_te_adj)
    
    p1 <- ggplot(cal_before, aes(avg_p, frac_pos)) +
      geom_abline(slope=1, intercept=0, linetype=2) +
      geom_point(size=2) + geom_line() +
      coord_equal(xlim=c(0,1), ylim=c(0,1)) +
      labs(title=sprintf("Before (RF) | sep=%.2f, rho=%.2f", sep, rho),
           x="Avg predicted prob", y="Empirical positive rate") +
      theme_minimal()
    
    p2 <- ggplot(cal_after, aes(avg_p, frac_pos)) +
      geom_abline(slope=1, intercept=0, linetype=2) +
      geom_point(size=2) + geom_line() +
      coord_equal(xlim=c(0,1), ylim=c(0,1)) +
      labs(title="After BBSC (posterior correction)",
           x="Avg predicted prob", y="Empirical positive rate") +
      theme_minimal()
    
    print(p1); print(p2)
  }
  
  return(out)
}

# --------------------------
# 4) Try multiple combinations
# ---- FIXED SWEEP OVER (sep, rho) ----
suppressPackageStartupMessages({ library(tidyr); library(purrr) })

rhos <- c(0.6, 0.75, 0.9)   # target P(Y*=1)
seps <- c(2.0, 2.8, 3.6)    # separability

grid <- expand_grid(sep = seps, rho = rhos)

# pmap_dfr binds the per-scenario tibbles row-wise; no duplicate columns
results <- pmap_dfr(list(grid$sep, grid$rho),
                    ~ run_scenario(sep = ..1, rho = ..2, plot = FALSE))

print(results)

# (Optional) run one scenario WITH plots
invisible(run_scenario(sep = 2.8, rho = 0.90, plot = TRUE))



###########Original Paper Correction: 
# === BBSC (paper method): importance-weighted retraining ===
suppressPackageStartupMessages({ library(ranger); library(dplyr); library(ggplot2) })

# Ensure factors
train$y      <- factor(train$y, levels = c(0,1))
test_tweak$y <- factor(test_tweak$y, levels = c(0,1))

# 1) Pull the BBSE estimate of test prevalence (use what's already in your session)
pi1_hat_test <- if (exists("pi1_hat_test")) {
  as.numeric(pi1_hat_test)
} else {
  as.numeric(bbseS$pi_hat["P(Y*=1)"])
}

# 2) Class weights w(y) = q(y)/p(y)
pi_train <- c(`0` = mean(train$y == "0"), `1` = mean(train$y == "1"))
w_class  <- c(`0` = (1 - pi1_hat_test) / pi_train["0"],
              `1` = (    pi1_hat_test) / pi_train["1"])
case_wts <- ifelse(train$y == "1", w_class["1"], w_class["0"])

# 3) Retrain a weighted RF (BBSC)
rf_bbsc <- ranger(
  y ~ X1 + X2,
  data = train,
  num.trees = 500,
  mtry = 2,
  min.node.size = 10,
  probability = TRUE,
  case.weights = case_wts,
  seed = 1
)

# 4) Predict before/after and remake calibration plot
p_before <- predict(rf,       data = test_tweak)$predictions[, "1"]
p_after  <- predict(rf_bbsc, data = test_tweak)$predictions[, "1"]

calibration_df <- function(yf, p, nbins = 12) {
  tibble(y = as.integer(as.character(yf)), p = p) |>
    mutate(bin = cut(p, breaks = seq(0,1,length.out = nbins+1), include.lowest = TRUE)) |>
    group_by(bin) |>
    summarise(avg_p = mean(p), frac_pos = mean(y), n = n(), .groups = "drop")
}

cal_before <- calibration_df(test_tweak$y, p_before)
cal_after  <- calibration_df(test_tweak$y, p_after)

p1 <- ggplot(cal_before, aes(avg_p, frac_pos)) +
  geom_abline(slope=1, intercept=0, linetype=2) +
  geom_point(size=2) + geom_line() +
  coord_equal(xlim=c(0,1), ylim=c(0,1)) +
  labs(title="Before (baseline RF)", x="Avg predicted prob", y="Empirical positive rate") +
  theme_minimal()

p2 <- ggplot(cal_after, aes(avg_p, frac_pos)) +
  geom_abline(slope=1, intercept=0, linetype=2) +
  geom_point(size=2) + geom_line() +
  coord_equal(xlim=c(0,1), ylim=c(0,1)) +
  labs(title="After BBSC (weighted RF)", x="Avg predicted prob", y="Empirical positive rate") +
  theme_minimal()

print(p1); print(p2)

# (Optional) quick numeric check
brier <- function(p, y) mean((p - as.integer(as.character(y)))^2)
cat(sprintf("Brier before/after: %.4f / %.4f\n",
            brier(p_before, test_tweak$y), brier(p_after, test_tweak$y)))
