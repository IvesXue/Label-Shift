## =========================================================
## MNIST → t-SNE (2D) → label-shift data prep (no modeling)
## - One joint t-SNE over train+test (shared 2D space)
## - Source: from TRAIN bucket → split train/val (iid)
## - Target: from TEST bucket → tweak-one prior q(y)
## - Saves .rds for BBSE next
## =========================================================
## install.packages(c("Rtsne","ggplot2"))

library(Rtsne)

## 0) Paths (edit train paths to your location)
img_train <- "C://Users/xueho/Desktop/Google downloads/archive/train-images.idx3-ubyte"
lbl_train <- "C://Users/xueho/Desktop/Google downloads/archive/train-labels.idx1-ubyte"
img_test  <- "C:/Users/xueho/Desktop/Google downloads/archive/t10k-images.idx3-ubyte"
lbl_test  <- "C:/Users/xueho/Desktop/Google downloads/archive/t10k-labels.idx1-ubyte"

## 1) Minimal IDX readers
read_idx_images <- function(path) {
  con <- file(path, "rb"); on.exit(close(con), add = TRUE)
  magic <- readBin(con, "integer", 1, 4, endian="big")
  n     <- readBin(con, "integer", 1, 4, endian="big")
  nr    <- readBin(con, "integer", 1, 4, endian="big")
  nc    <- readBin(con, "integer", 1, 4, endian="big")
  stopifnot(magic == 2051L)
  pix   <- nr * nc
  Xraw  <- readBin(con, "raw", n * pix, 1, endian="big")
  X     <- matrix(as.integer(Xraw), nrow = n, ncol = pix, byrow = TRUE)
  attr(X,"nrow") <- nr; attr(X,"ncol") <- nc
  X
}
read_idx_labels <- function(path) {
  con <- file(path, "rb"); on.exit(close(con), add = TRUE)
  magic <- readBin(con, "integer", 1, 4, endian="big")
  n     <- readBin(con, "integer", 1, 4, endian="big")
  stopifnot(magic == 2049L)
  as.integer(readBin(con, "raw", n, 1, endian="big"))
}

## 2) Load buckets
X_tr <- read_idx_images(img_train) / 255
y_tr <- read_idx_labels(lbl_train)
X_te <- read_idx_images(img_test)  / 255
y_te <- read_idx_labels(lbl_test)

stopifnot(nrow(X_tr) == length(y_tr), nrow(X_te) == length(y_te))

## 3) One joint t-SNE (train+test) → shared 2D
X_all   <- rbind(X_tr, X_te)
y_all   <- c(y_tr, y_te)
is_test <- c(rep(FALSE, length(y_tr)), rep(TRUE, length(y_te)))

set.seed(123)
tsne_all <- Rtsne(
  X_all,
  dims = 2,
  perplexity = 30,          # you can try 30–50
  theta = 0.5,
  pca = FALSE,
  check_duplicates = FALSE,
  verbose = TRUE,
  max_iter = 1000,
  eta = 200,
  exaggeration_factor = 12,
  num_threads = max(1L, parallel::detectCores() - 1L)
)
Z_all <- tsne_all$Y
colnames(Z_all) <- c("z1","z2")

## Split back into TRAIN/TEST 2D
Z_tr <- Z_all[!is_test, , drop=FALSE]
Z_te <- Z_all[ is_test, , drop=FALSE]

## 4) Source: TRAIN bucket → random split (iid)
set.seed(7)
n_tr <- nrow(Z_tr)
idx  <- sample.int(n_tr)
frac_train <- 0.8
n_src_train <- floor(frac_train * n_tr)

src_train_idx <- idx[1:n_src_train]
src_val_idx   <- idx[(n_src_train+1):n_tr]

src_train <- list(X = Z_tr[src_train_idx, , drop=FALSE], y = y_tr[src_train_idx])
src_val   <- list(X = Z_tr[src_val_idx,   , drop=FALSE], y = y_tr[src_val_idx])

## Source prior p_hat(y) from the *training* split (for BBSE later)
p_hat <- table(factor(src_train$y, levels = 0:9))
p_hat <- as.numeric(p_hat) / sum(p_hat)

## 5) Target: TEST bucket → tweak-one prior q(y), resample with replacement
tweak_one_prior_10 <- function(rho, digit = 5L) {
  stopifnot(digit %in% 0:9, rho >= 0, rho <= 1)
  q <- rep((1 - rho) / 9, 10); q[digit + 1L] <- rho; q
}
sample_by_prior_multiclass <- function(X, y, prior, n_out = length(y)) {
  stopifnot(length(prior) == 10, abs(sum(prior) - 1) < 1e-8)
  tgt_counts <- round(prior * n_out)
  d <- n_out - sum(tgt_counts)
  if (d != 0) {
    adj <- order(prior, decreasing = (d > 0))[1:abs(d)]
    tgt_counts[adj] <- tgt_counts[adj] + sign(d)
  }
  keep <- integer(0)
  for (k in 0:9) {
    cand <- which(y == k)
    keep <- c(keep, sample(cand, size = tgt_counts[k+1L], replace = TRUE))
  }
  list(X = X[keep, , drop=FALSE], y = y[keep])
}

rho <- 0.50           # boosted mass
d_star <- 5L          # which digit to boost
q_true <- tweak_one_prior_10(rho, d_star)

N_tgt <- nrow(Z_te)   # usually keep same size as TEST bucket
tgt <- sample_by_prior_multiclass(Z_te, y_te, q_true, n_out = N_tgt)

## 6) Sanity checks
q_emp <- table(factor(tgt$y, levels = 0:9)); q_emp <- as.numeric(q_emp) / sum(q_emp)
cat("\nSource p_hat(y):\n"); print(round(p_hat, 4))
cat("Target q_true(y) requested:\n"); print(round(q_true, 4))
cat("Target q_emp(y) realized:\n"); print(round(q_emp, 4))

## 7) Save artifacts (2D features)
saveRDS(src_train, file = "src_train_2d.rds")           # list(X: [*,2], y)
saveRDS(src_val,   file = "src_val_2d.rds")
saveRDS(tgt,       file = "tgt_tweak_one_2d.rds")
saveRDS(list(p_hat = p_hat, q_true = q_true, q_emp = q_emp,
             rho = rho, digit = d_star, space = "tSNE-2D"),
        file = "priors_meta_2d.rds")
cat("\nSaved: src_train_2d.rds, src_val_2d.rds, tgt_tweak_one_2d.rds, priors_meta_2d.rds\n")



## =========================================================
## MNIST (2D) — RF + BBSE + Posterior Calibration
## * Binary one-vs-all for the boosted digit (Y=1 if digit=d_star)
## * Calibration plots before/after correction
## * Sweeps over prevalence rho and effective separability
## =========================================================
suppressPackageStartupMessages({
  library(ranger)
  library(dplyr)
  library(ggplot2)
  library(purrr)
  library(tidyr)
})

## -------------------------
## 0) Load 2D data artifacts
## -------------------------
src_train <- readRDS("src_train_2d.rds")   # list(X=[n,2], y = digit 0..9)
src_val   <- readRDS("src_val_2d.rds")
tgt       <- readRDS("tgt_tweak_one_2d.rds")
meta      <- readRDS("priors_meta_2d.rds") # list(p_hat, q_true, q_emp, rho, digit, ...)

d_star <- meta$digit %||% 5L   # boosted digit (0..9)
rho0   <- meta$rho  %||% 0.50  # original target prevalence used

`%||%` <- function(a,b) if (is.null(a)) b else a

## Make binary labels: 1 if digit == d_star, else 0
to_binary <- function(y, pos = d_star) as.integer(y == pos)
dat_tr <- data.frame(z1 = src_train$X[,1], z2 = src_train$X[,2], y = to_binary(src_train$y))
dat_va <- data.frame(z1 = src_val$X[,1],   z2 = src_val$X[,2],   y = to_binary(src_val$y))
dat_te <- data.frame(z1 = tgt$X[,1],       z2 = tgt$X[,2],       y = to_binary(tgt$y))

## Empirical source prior (train split)
pi1_tr <- mean(dat_tr$y == 1)

## -------------------------
## 1) Helpers
## -------------------------
predict_prob <- function(model, data) {
  pred <- predict(model, data = data)$predictions
  if (is.matrix(pred)) {
    if ("1" %in% colnames(pred)) pred[, "1"] else pred[, 2]
  } else as.numeric(pred)  # ranger may return vector for binary
}

## Soft BBSE (binary): C := [ E[1-p|y=0]  E[1-p|y=1] ; E[p|y=0]  E[p|y=1] ]
bbse_soft_binary <- function(y_true, p_train, p_test) {
  y_true <- as.integer(y_true)
  idx0 <- which(y_true == 0); idx1 <- which(y_true == 1)
  C <- matrix(0, 2, 2)
  C[1,1] <- mean(1 - p_train[idx0]); C[2,1] <- mean(p_train[idx0])
  C[1,2] <- mean(1 - p_train[idx1]); C[2,2] <- mean(p_train[idx1])
  mu_hat <- c(mean(1 - p_test), mean(p_test))
  w_hat <- tryCatch(solve(C, mu_hat), error = function(e) MASS::ginv(C) %*% mu_hat)
  w_hat <- pmax(as.numeric(w_hat), 0)
  # Convert to priors with source p(y): p(y=1)=pi1_tr, p(y=0)=1-pi1_tr
  p_vec <- c(1 - pi1_tr, pi1_tr)
  q_hat <- p_vec * w_hat
  q_hat <- q_hat / sum(q_hat)
  list(pi1_hat = q_hat[2], C = C, mu_hat = mu_hat, w_hat = w_hat, q_hat = q_hat)
}

## Saerens–Latinne–Decaestecker posterior correction
adjust_posteriors <- function(p, pi1_train, pi1_hat_test) {
  num <- p * (pi1_hat_test / pi1_train)
  den <- num + (1 - p) * ((1 - pi1_hat_test) / (1 - pi1_train))
  pmin(pmax(num / den, 1e-8), 1 - 1e-8)
}

## Calibration utilities
calibration_df <- function(y, p, nbins = 12) {
  tibble(y = as.integer(y), p = p) |>
    mutate(bin = cut(p, breaks = seq(0,1,length.out = nbins+1), include.lowest = TRUE)) |>
    group_by(bin) |>
    summarise(avg_p = mean(p), frac_pos = mean(y), n = n(), .groups = "drop")
}
brier <- function(p, y) mean((p - as.integer(y))^2)
ece <- function(y, p, nbins=12){
  df <- calibration_df(y, p, nbins)
  w <- df$n / sum(df$n); sum(w * abs(df$avg_p - df$frac_pos))
}

plot_cal <- function(df, title) {
  ggplot(df, aes(avg_p, frac_pos)) +
    geom_abline(slope=1, intercept=0, linetype=2) +
    geom_point(size=2) + geom_line() +
    coord_equal(xlim=c(0,1), ylim=c(0,1)) +
    labs(title = title, x = "Avg predicted prob", y = "Empirical positive rate") +
    theme_minimal()
}

## -------------------------
## 2) One full run on current target
## -------------------------
run_once <- function(dat_tr, dat_va, dat_te, label = NULL, make_plots = TRUE) {
  # RF (probability = TRUE)
  tr <- mutate(dat_tr, y = factor(y, levels = c(0,1)))
  rf <- ranger(y ~ z1 + z2, data = tr, probability = TRUE,
               num.trees = 500, mtry = 2, min.node.size = 10, seed = 1)
  
  p_tr <- predict_prob(rf, dat_tr)
  p_va <- predict_prob(rf, dat_va)  # not used in BBSE, but you can inspect
  p_te <- predict_prob(rf, dat_te)
  
  # BBSE (soft) using *source validation* or *source train*?
  # Paper uses a holdout; we’ll use src_val for C (more faithful).
  va <- mutate(dat_va, y = as.integer(y))
  bbse <- bbse_soft_binary(va$y, p_train = p_va, p_test = p_te)
  pi1_hat <- as.numeric(bbse$pi1_hat)
  
  # Before/after calibration
  cal_before <- calibration_df(dat_te$y, p_te)
  p_te_adj   <- adjust_posteriors(p_te, pi1_tr, pi1_hat)
  cal_after  <- calibration_df(dat_te$y, p_te_adj)
  
  # Metrics
  acc_before <- mean((p_te >= 0.5) == (dat_te$y == 1))
  acc_after  <- mean((p_te_adj >= 0.5) == (dat_te$y == 1))
  out <- tibble(
    label = label %||% "current_target",
    pi1_true = mean(dat_te$y == 1),
    pi1_hat  = pi1_hat,
    acc_before = acc_before,
    acc_after  = acc_after,
    brier_before = brier(p_te, dat_te$y),
    brier_after  = brier(p_te_adj, dat_te$y),
    ece_before   = ece(dat_te$y, p_te),
    ece_after    = ece(dat_te$y, p_te_adj)
  )
  
  if (isTRUE(make_plots)) {
    print(plot_cal(cal_before, sprintf("Before (RF) | %s", out$label)))
    print(plot_cal(cal_after,  "After BBSC (posterior correction)"))
  }
  list(summary = out, rf = rf, pi1_hat = pi1_hat,
       cal_before = cal_before, cal_after = cal_after)
}

res_current <- run_once(dat_tr, dat_va, dat_te, label = sprintf("rho=%.2f (digit=%d)", rho0, d_star), make_plots = TRUE)
print(res_current$summary)

## -------------------------------------------------------
## 3) Sweeps: change prevalence (rho) and “separability”
##    - Prevalence: rebuild *target* by resampling test set with new rho
##    - Separability proxy: train on a fraction of src_train to weaken model
## -------------------------------------------------------
# helper to (re)build tweak-one target from original test pool we stored:
rebuild_target <- function(rho, Z_te = tgt$X, y_te_digits = tgt$y * 0 + tgt$y) {
  # We don't have original full test pool in this file, so reuse current tgt pool
  # If you saved original test bucket separately, swap Z_te, y_te_digits accordingly.
  q <- rep((1 - rho)/9, 10); q[d_star + 1] <- rho
  tgt_counts <- round(q * nrow(Z_te))
  diff <- nrow(Z_te) - sum(tgt_counts)
  if (diff != 0) {
    adj <- order(q, decreasing = (diff > 0))[1:abs(diff)]
    tgt_counts[adj] <- tgt_counts[adj] + sign(diff)
  }
  keep <- integer(0)
  for (k in 0:9) {
    cand <- which((tgt$y) == k)  # falls back to current pool's labels
    if (length(cand) == 0) next
    keep <- c(keep, sample(cand, size = min(length(cand), tgt_counts[k+1]), replace = TRUE))
  }
  list(
    X = tgt$X[keep, , drop = FALSE],
    y = to_binary(tgt$y[keep])
  )
}

run_scenario <- function(rho = 0.75, train_frac = 1.0, plot = FALSE) {
  # shrink source-train to alter effective separability
  set.seed(42)
  sel <- sample(seq_len(nrow(dat_tr)), size = floor(train_frac * nrow(dat_tr)))
  tr_small <- dat_tr[sel, , drop = FALSE]
  
  # rebuild target with new rho (tweak-one)
  tgt_new <- rebuild_target(rho)
  te_new  <- data.frame(z1 = tgt_new$X[,1], z2 = tgt_new$X[,2], y = tgt_new$y)
  
  # reuse same src_val
  res <- run_once(tr_small, dat_va, te_new,
                  label = sprintf("rho=%.2f | train_frac=%.2f", rho, train_frac),
                  make_plots = plot)
  res$summary
}

rhos  <- c(0.60, 0.75, 0.90)
fracs <- c(0.25, 0.50, 1.00)

grid <- expand_grid(rho = rhos, train_frac = fracs)
results <- pmap_dfr(list(grid$rho, grid$train_frac),
                    ~ run_scenario(rho = ..1, train_frac = ..2, plot = FALSE))
print(results)

## (Optional) visualize one sweep with plots:
invisible(run_scenario(rho = 0.90, train_frac = 0.50, plot = TRUE))

###################################################################################
## 0) Load 2D artifacts (from your earlier step)
src_train <- readRDS("src_train_2d.rds")   # list(X=[n,2], y = digit 0..9)
src_val   <- readRDS("src_val_2d.rds")
tgt       <- readRDS("tgt_tweak_one_2d.rds")
meta      <- readRDS("priors_meta_2d.rds") # has p_hat (source prior), q_true, etc.

## Prepare data frames
df_tr <- data.frame(z1 = src_train$X[,1], z2 = src_train$X[,2], y = factor(src_train$y, levels = 0:9))
df_va <- data.frame(z1 = src_val$X[,1],   z2 = src_val$X[,2],   y = factor(src_val$y,   levels = 0:9))
df_te <- data.frame(z1 = tgt$X[,1],       z2 = tgt$X[,2],       y = factor(tgt$y,       levels = 0:9))

## Source prior p_hat from the *training* split (needed for correction)
p_hat <- as.numeric(table(df_tr$y))[1:10]; p_hat <- p_hat / sum(p_hat)

## 1) Train a **multiclass** RF
set.seed(1)
rf <- ranger(y ~ z1 + z2, data = df_tr,
             probability = TRUE, num.trees = 500, mtry = 2,
             min.node.size = 10, seed = 1)

pred_mat <- function(model, data) {
  pr <- predict(model, data = data)$predictions
  # ensure columns in order "0","1",...,"9"
  pr[, as.character(0:9), drop = FALSE]
}
P_tr <- pred_mat(rf, df_tr)  # n_tr × 10
P_va <- pred_mat(rf, df_va)  # n_va × 10
P_te <- pred_mat(rf, df_te)  # n_te × 10

## 2) BBSE (soft) in K=10
## Build the **soft confusion** matrix M where M[i,j] = E[P(ŷ=i|X)| y=j]
## (columns ~ classes j; rows ~ predicted class-prob components i)
K <- 10
y_va_int <- as.integer(as.character(df_va$y))   # 0..9
M <- matrix(0, nrow = K, ncol = K)
for (j in 0:9) {
  idx <- which(y_va_int == j)
  if (length(idx) > 0) M[, j+1] <- colMeans(P_va[idx, , drop = FALSE])
}

## μ_hat = E_target[P(ŷ|X)] (mean probability vector on target)
mu_hat <- colMeans(P_te)       # length 10; sums to 1

## Solve M * q_hat ≈ μ_hat  (clip + renormalize)
q_hat <- tryCatch(solve(M, mu_hat), error = function(e) MASS::ginv(M) %*% mu_hat)
q_hat <- as.numeric(q_hat)
q_hat[q_hat < 0] <- 0
q_hat <- q_hat / sum(q_hat)

cat("\nBBSE estimate of target prior q_hat(y):\n")
print(round(q_hat, 4))
if (!is.null(meta$q_true)) {
  cat("True q(y) (used to build target):\n")
  print(round(meta$q_true, 4))
}

## 3) Multiclass posterior **prior-shift** correction
## For each example x: p'_k(x) ∝ p_k(x) * (q_hat_k / p_hat_k), then normalize.
ratio <- q_hat / p_hat
ratio[!is.finite(ratio)] <- 0
P_te_adj <- sweep(P_te, 2, ratio, `*`)
P_te_adj <- P_te_adj / rowSums(P_te_adj)

## 4) Calibration & metrics (multiclass)
## Per-class reliability (one-vs-rest) + macro ECE/Brier
calibration_df_onevsrest <- function(y_true, P, k, nbins = 12) {
  yk <- as.integer(as.character(y_true)) == k
  pk <- P[, k+1]
  tibble(y = yk, p = pk) |>
    mutate(bin = cut(p, breaks = seq(0,1,length.out = nbins+1), include.lowest = TRUE)) |>
    group_by(bin) |>
    summarise(avg_p = mean(p), frac_pos = mean(y), n = n(), .groups = "drop") |>
    mutate(class = k)
}
ece_onevsrest <- function(y_true, P, nbins = 12) {
  per_class <- map_dfr(0:9, ~ calibration_df_onevsrest(y_true, P, .x, nbins))
  per_class %>%
    group_by(class) %>%
    summarise(ece = sum((n / sum(n)) * abs(avg_p - frac_pos)), .groups = "drop") %>%
    summarise(macro_ece = mean(ece)) %>% pull(macro_ece)
}
brier_multiclass <- function(y_true, P) {
  Y <- model.matrix(~ factor(y_true, levels = 0:9) - 1)
  mean(rowSums((P - Y)^2))
}

# ECE/Brier before and after
ece_before <- ece_onevsrest(df_te$y, P_te)
ece_after  <- ece_onevsrest(df_te$y, P_te_adj)
brier_before <- brier_multiclass(df_te$y, P_te)
brier_after  <- brier_multiclass(df_te$y, P_te_adj)

cat(sprintf("\nMacro ECE  before/after: %.4f / %.4f\n", ece_before, ece_after))
cat(sprintf("Brier (multi) before/after: %.4f / %.4f\n", brier_before, brier_after))

## 5) Optional: plot a few per-class calibration curves
plot_cal_class <- function(df_cal, k, title_suffix="") {
  ggplot(df_cal %>% filter(class == k), aes(avg_p, frac_pos)) +
    geom_abline(slope=1, intercept=0, linetype=2) +
    geom_point(size=2) + geom_line() +
    coord_equal(xlim=c(0,1), ylim=c(0,1)) +
    labs(title = sprintf("Class %d %s", k, title_suffix),
         x = "Avg predicted prob", y = "Empirical positive rate") +
    theme_minimal()
}
cal_before_k <- map_dfr(0:9, ~ calibration_df_onevsrest(df_te$y, P_te, .x))
cal_after_k  <- map_dfr(0:9, ~ calibration_df_onevsrest(df_te$y, P_te_adj, .x))