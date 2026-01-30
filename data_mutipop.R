#set up
set.seed(42)

M <- 10 #number of pop


n <- 5000 # sample size for each pop

k <- 50 # total number of features

n_stable <- 10 #number of features that satisfy label shift

n_violate <- 20 # number of violating features:

n_noise <- 20 # number o noisy features

pi_vec <- seq(0.05, 0.95, length.out = M) # p(y=1) for each pop

# mu
stable_mu0 <- 0
stable_mu1 <- 2

violate_mu0 <- 0
violate_mu1 <- 2

shift_strength <- 1.0 # how strong the shift for each pop is

noise_mean <- 0
noise_sd <- 1


stable_idx  <- 1:n_stable # position for stable features

violate_idx <- (n_stable + 1):(n_stable + n_violate) #similar as above

noise_idx   <- (n_stable + n_violate + 1):k

feature_names <- paste0("X", 1:k) # name features X1, X2....

pop_factor <- seq(-1, 1, length.out = M) # create pop factor to shift each population


# generate data
dat <- data.frame()
# stable
for (m in 1:M) {
  y <- rbinom(n, size = 1, prob = pi_vec[m]) # generate labels for each pop
  
  X <- matrix(NA, nrow = n, ncol = k)# create empty matrix to store features for population m
  
  for (j in stable_idx) {
    mean_j <- ifelse(y == 1, stable_mu1, stable_mu0) # for each stable feature j, if y =1, mean be mu1,if not, mean be mu0
    
    X[, j] <- rnorm(n, mean = mean_j, sd = 1) # draw normal distribution with mu0 & mu1 according to Y. 
  }
  
  # violated
  shift_m <- shift_strength * pop_factor[m] # shift for each pop
  
  for (j in violate_idx) {
    base_mean <- ifelse(y == 1, violate_mu1, violate_mu0) # base mean similar to stable
    extra <- ifelse(y == 1, +shift_m, -shift_m) # extra effect, if y=1 +shiftm, if not -shiftm
    
    # draw normal distribution according to the rule
    X[,j] <- rnorm(n, mean = base_mean + extra, sd = 1)
  }
  
  # Noisy
  for (j in noise_idx) {
    X[, j] <- rnorm(n, mean = noise_mean, sd = noise_sd)
  }
  
  # name features
  colnames(X) <- feature_names
  
  # built data frame
  df_m <- data.frame(
    pop = paste0("pop", m),
    y = y,
    X
  )
  dat <- rbind(dat, df_m)
}

# check
stable_features <- feature_names[stable_idx]
violating_features <- feature_names[violate_idx]
noise_features <- feature_names[noise_idx]

head(dat)

stable_features

violating_features

noise_features

pi_vec

# visual
library(ggplot2)
prop_df <- aggregate(y ~ pop, data = dat, FUN = mean)

# y=1 for each pop
ggplot(prop_df, aes(x = pop, y = y)) +
  geom_col() +
  labs(title = "P(Y=1) by Population", x = "Population", y = "Proportion of y=1") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# for X1(stable), density by y: should center at 0 and 2
ggplot(dat, aes(x = X1, fill = factor(y))) +
  geom_density(alpha = 0.4) +
  facet_wrap(~ pop, ncol = 5) +
  labs(title = "Stable feature X1: density by y, faceted by population",
       fill = "y") 

# for X15, violated, density by y
ggplot(dat, aes(x = X15, fill = factor(y))) +
  geom_density(alpha = 0.4) +
  facet_wrap(~ pop, ncol = 5) +
  labs(title = "Violating feature X15: density by y, faceted by population",
       fill = "y")
