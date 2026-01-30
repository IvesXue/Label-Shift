# ============================================================
# SIMPLE label-shift simulation (BEGINNER FRIENDLY, heavily commented)
# ============================================================
# Goal:
#   Create simulated training data from M different "populations" (sites/domains).
#   - Each population m has a different class prior P(Y=1)  -> label shift happens in P(Y)
#   - Some features are "stable":   P(X | Y) is the SAME for every population
#   - Some features "violate":      P(X | Y, m) CHANGES across populations (so label shift fails)
#   - Some features are "noise":    unrelated to Y and unrelated to population
#
# Output:
#   dat: a data.frame with columns:
#        pop (population id), y (label), X1...Xk (features)
#   stable_features / violating_features / noise_features: ground-truth feature groups
# ============================================================


# ---------------------------
# 0) Reproducibility
# ---------------------------

set.seed(1834)
# set.seed() makes random results repeatable.
# If you run the code again with the same seed, you get the same simulated dataset.


# ---------------------------
# 1) Basic simulation size settings
# ---------------------------

M <- 10
# M = number of populations (m = 1, 2, ..., M)

n <- 5000
# n = sample size per population (same n for each population in this simple version)

k <- 50
# k = total number of features (X1, X2, ..., Xk)


# ---------------------------
# 2) Decide how many features are stable / violating / noise
# ---------------------------

n_stable <- 10
# n_stable = number of "stable" features that satisfy label shift:
#   P(X_j | Y) is identical across all populations

n_violate <- 20
# n_violate = number of "violating" features:
#   P(X_j | Y, population) changes with population

n_noise <- k - n_stable - n_violate
# n_noise = remaining features are noise (not related to Y or population)
# This makes sure total features = stable + violate + noise = k.


# ---------------------------
# 3) Label shift setup: choose class priors P(Y=1) per population
# ---------------------------

pi_min <- 0.05
pi_max <- 0.95
# These are the minimum and maximum values for the class prior across populations.
# Example: population 1 might have P(Y=1)=0.05 while population M might have 0.95.

pi_vec <- seq(pi_min, pi_max, length.out = M)
# pi_vec is a length-M vector:
#   pi_vec[m] = P(Y=1 in population m)
# Using seq() makes them change smoothly from pi_min to pi_max.
#
# This is the KEY reason we have label shift:
#   P(Y) differs across populations.


# ---------------------------
# 4) Define distributions for the stable features
# ---------------------------

stable_mu0 <- 0
stable_mu1 <- 1.0
# These are the means of stable features conditional on label:
#   if Y=0: mean = stable_mu0
#   if Y=1: mean = stable_mu1
#
# We set variance = 1 (sd = 1) later.
# IMPORTANT: These stable-feature parameters DO NOT depend on population.
# So P(X | Y) is the same for all populations -> obey label shift assumption.


# ---------------------------
# 5) Define distributions for the violating features
# ---------------------------

violate_mu0 <- 0
violate_mu1 <- 1.0
# Base means of violating features conditional on label (before adding population effects)

shift_strength <- 1.0
# How strongly population changes the violating features.
# Bigger shift_strength = bigger differences of P(X|Y,m) across populations.


# ---------------------------
# 6) Define distributions for noise features
# ---------------------------

noise_mean <- 0
noise_sd <- 1
# Noise features:
#   X_noise ~ Normal(noise_mean, noise_sd^2)
# They are independent of Y and population.
# This tests whether methods mistakenly select useless features.


# ---------------------------
# 7) Create feature index sets (which columns are stable/violate/noise)
# ---------------------------

stable_idx  <- 1:n_stable
# stable_idx = feature positions for stable features, e.g. 1..10

violate_idx <- (n_stable + 1):(n_stable + n_violate)
# violate_idx = positions after stable, e.g. 11..30

noise_idx   <- (n_stable + n_violate + 1):k
# noise_idx = remaining positions, e.g. 31..50

feature_names <- paste0("X", 1:k)
# creates names: "X1", "X2", ..., "Xk"
# we will use these as column names


# ---------------------------
# 8) Create a simple population factor (how population affects violations)
# ---------------------------

pop_factor <- seq(-1, 1, length.out = M)
# pop_factor is a length-M vector:
#   pop_factor[m] ranges from -1 (pop1) to +1 (popM).
# We use it to create a smooth change across populations.
#
# In violating features we will add:
#   + shift_strength * pop_factor[m]  for Y=1
#   - shift_strength * pop_factor[m]  for Y=0
#
# That means P(X|Y,m) differs across populations, violating label shift.


# ---------------------------
# 9) Generate the dataset population by population
# ---------------------------

all_data <- list()
# We'll store each population's data.frame in a list, then combine at the end.

for (m in 1:M) {
  # loop over populations m = 1..M
  
  # ----- 9.1) Generate labels y for population m -----
  y <- rbinom(n, size = 1, prob = pi_vec[m])
  # rbinom(n, 1, prob) generates n draws from Bernoulli(prob).
  # Here:
  #   y[i] = 1 with probability pi_vec[m]
  #   y[i] = 0 with probability 1 - pi_vec[m]
  #
  # This makes P(Y=1) different in each population -> label shift in priors.
  
  # ----- 9.2) Prepare an empty feature matrix -----
  X <- matrix(NA, nrow = n, ncol = k)
  # X will hold all features for population m, shape n x k.
  
  # ==================================================
  # A) Stable features: obey label shift
  # ==================================================
  for (j in stable_idx) {
    # For each stable feature j:
    mean_j <- ifelse(y == 1, stable_mu1, stable_mu0)
    # ifelse() returns a vector length n:
    #   mean_j[i] = stable_mu1 if y[i]==1
    #             = stable_mu0 if y[i]==0
    
    X[, j] <- rnorm(n, mean = mean_j, sd = 1)
    # rnorm() draws Normal random variables.
    # Here we draw n values:
    #   X_ij ~ Normal(mean_j[i], 1^2)
    #
    # Because mean_j depends only on y (not on m),
    # P(X_j | Y) is the SAME across populations.
  }
  
  # ==================================================
  # B) Violating features: break label shift
  # ==================================================
  
  shift_m <- shift_strength * pop_factor[m]
  # shift_m is the population-specific shift value for population m.
  # Example:
  #   if pop_factor[m] = -1 and shift_strength=1 -> shift_m = -1
  #   if pop_factor[m] = +1 and shift_strength=1 -> shift_m = +1
  
  for (j in violate_idx) {
    
    base_mean <- ifelse(y == 1, violate_mu1, violate_mu0)
    # base_mean depends on Y (like stable), but next we ADD a population effect.
    
    extra <- ifelse(y == 1, +shift_m, -shift_m)
    # extra is the key label-dependent population effect:
    #   if y=1, add +shift_m
    #   if y=0, add -shift_m
    #
    # So both classes are shifted in opposite directions as population changes.
    
    X[, j] <- rnorm(n, mean = base_mean + extra, sd = 1)
    # This means:
    #   X_j | (Y=1, pop=m) ~ Normal(violate_mu1 + shift_m, 1)
    #   X_j | (Y=0, pop=m) ~ Normal(violate_mu0 - shift_m, 1)
    #
    # Because shift_m depends on population m, P(X|Y,m) changes across populations.
    # Therefore these features violate the label shift assumption.
  }
  
  # ==================================================
  # C) Noise features: unrelated to everything
  # ==================================================
  for (j in noise_idx) {
    X[, j] <- rnorm(n, mean = noise_mean, sd = noise_sd)
    # Noise does not depend on Y or population.
  }
  
  # ----- 9.3) Add column names -----
  colnames(X) <- feature_names
  
  # ----- 9.4) Build a data.frame for population m -----
  df_m <- data.frame(
    pop = paste0("pop", m),
    # pop is a string label for the population: "pop1", "pop2", ..., "popM"
    
    y = y,
    # y is the label vector (0/1)
    
    X,
    # all k features
    
    stringsAsFactors = FALSE
    # keep pop as a character, not factor (beginner-friendly)
  )
  
  all_data[[m]] <- df_m
  # store this population's data.frame
}

# ---------------------------
# 10) Combine all populations into one dataset
# ---------------------------
dat <- do.call(rbind, all_data)
# rbind stacks data.frames row-wise
# do.call applies rbind to the whole list


# ---------------------------
# 11) Ground-truth feature groups (useful for checking selection)
# ---------------------------
stable_features <- feature_names[stable_idx]
violating_features <- feature_names[violate_idx]
noise_features <- feature_names[noise_idx]

# stable_features are the TRUE features that obey label shift
# violating_features are the TRUE features that break it
# noise_features are irrelevant


# ---------------------------
# 12) Quick checks (optional)
# ---------------------------
head(dat)
# shows first rows of the full dataset

stable_features
# shows names of stable features

pi_vec
# shows P(Y=1) in each population (how label shift was created)
