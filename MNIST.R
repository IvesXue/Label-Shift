## ============================================
## MNIST t-SNE on ALL samples (no PCA) in R
## ============================================
## Install once if needed:
## install.packages(c("Rtsne","ggplot2","dplyr"))

library(Rtsne)
library(ggplot2)
library(dplyr)

## -------------------------
## 1) Your file paths
## -------------------------
img_path <- "C:/Users/xueho/Desktop/Google downloads/archive/t10k-images.idx3-ubyte"
lbl_path <- "C:/Users/xueho/Desktop/Google downloads/archive/t10k-labels.idx1-ubyte"  # make sure this exists

## If you want to switch to the TRAIN set (60,000):
## img_path <- "C:/.../train-images.idx3-ubyte"
## lbl_path <- "C:/.../train-labels.idx1-ubyte"

## -----------------------------------------
## 2) Minimal IDX readers (big-endian bytes)
##    Pattern per StackOverflow post
## -----------------------------------------
read_idx_images <- function(path) {
  con <- file(path, "rb"); on.exit(close(con), add = TRUE)
  magic  <- readBin(con, "integer", n=1, size=4, endian="big")
  n      <- readBin(con, "integer", n=1, size=4, endian="big")
  nrow   <- readBin(con, "integer", n=1, size=4, endian="big")
  ncol   <- readBin(con, "integer", n=1, size=4, endian="big")
  if (magic != 2051L) stop("Not an IDX image file (magic != 2051).")
  n_pix <- nrow * ncol
  rawv  <- readBin(con, what="raw", n=n * n_pix, size=1, endian="big")
  X     <- matrix(as.integer(rawv), nrow=n, ncol=n_pix, byrow=TRUE)
  attr(X,"nrow") <- nrow; attr(X,"ncol") <- ncol
  X
}

read_idx_labels <- function(path) {
  con <- file(path, "rb"); on.exit(close(con), add = TRUE)
  magic <- readBin(con, "integer", n=1, size=4, endian="big")
  n     <- readBin(con, "integer", n=1, size=4, endian="big")
  if (magic != 2049L) stop("Not an IDX label file (magic != 2049).")
  as.integer(readBin(con, what="raw", n=n, size=1, endian="big"))
}

## ----------------------------
## 3) Load + prepare all samples
## ----------------------------
X <- read_idx_images(img_path)    # [N x 784], 0..255
y <- read_idx_labels(lbl_path)    # length N, 0..9
stopifnot(nrow(X) == length(y))
message(sprintf("Loaded %d images (%dx%d).", nrow(X), attr(X,"nrow"), attr(X,"ncol")))

## Normalize pixels to 0..1 for t-SNE stability
Xn <- X / 255

## ---------------------------------
## 4) t-SNE (Barnes–Hut; no PCA)
##    - Settings per Appsilon + Rtsne manual
## ---------------------------------
set.seed(123)
tsne_out <- Rtsne(
  Xn,
  dims = 2,
  perplexity = 30,        # good default; try 30–50 for MNIST size
  theta = 0.5,            # Barnes–Hut (speed/accuracy tradeoff)
  pca = FALSE,            # you asked: NO PCA
  check_duplicates = FALSE, # MNIST can have duplicates after normalization
  verbose = TRUE,
  max_iter = 1000,
  eta = 200,                # learning rate per Rtsne docs
  exaggeration_factor = 12, # standard early exaggeration
  num_threads = max(1, parallel::detectCores() - 1)  # parallel if available
)

## ---------------------
## 5) 2D visualization
## ---------------------
emb <- as.data.frame(tsne_out$Y)
names(emb) <- c("TSNE1","TSNE2")
emb$digit <- factor(y)

ggplot(emb, aes(TSNE1, TSNE2, color = digit)) +
  geom_point(alpha = 0.75, size = 1.2) +
  coord_equal() +
  theme_minimal() +
  labs(
    title = "MNIST t-SNE",
    subtitle = sprintf("N = %d  |  perplexity = %d  |  theta = %.1f",
                       nrow(emb), 30, 0.5),
    color = "Digit"
  )
