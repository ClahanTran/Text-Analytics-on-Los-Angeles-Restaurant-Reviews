# =============================================================================
# config.R
# User-configurable settings for the LA Restaurant Reviews analysis pipeline
# =============================================================================
# INSTRUCTIONS:
#   1. Set DATA_PATH to the location of the CSV on your machine.
#   2. Set RESULTS_DIR to where you want plots/outputs saved.
#   3. Source this file at the top of any script, or let run_all.R handle it.
# =============================================================================

# --- Data ---------------------------------------------------------------
# Path to the Yelp dataset CSV (update this for your machine)
DATA_PATH <- "C:/Users/tranc/Downloads/Text-Analytics-on-Los-Angeles-Restaurant-Reviews/archive (1)/top 240 restaurants recommanded in los angeles 2.csv"

# Directory for saving plots and output files (created automatically if missing)
RESULTS_DIR <- "results"

# --- Reproducibility ----------------------------------------------------
SEED <- 3847   # global random seed used throughout all scripts

# --- Word Embeddings (Steps 5-6) ----------------------------------------
EMBEDDING_DIM    <- 100   # number of embedding dimensions
EMBEDDING_ITER   <- 20    # training iterations
EMBEDDING_WINDOW <- 5     # context window size
EMBEDDING_MIN    <- 3     # minimum token frequency to include

# --- Topic Modeling (Step 8) --------------------------------------------
LDA_K      <- 6    # number of LDA topics
LDA_BURNIN <- 500  # Gibbs sampler burn-in iterations
LDA_ITER   <- 1000 # Gibbs sampler iterations after burn-in
LDA_THIN   <- 10   # thinning interval

# --- Clustering (Step 7) ------------------------------------------------
KMEANS_K <- 5   # number of k-means clusters

# --- ML Modeling (Step 9) -----------------------------------------------
TRAIN_PROP <- 0.80   # proportion of data used for training
N_TREES    <- 500    # number of trees in random forest models
