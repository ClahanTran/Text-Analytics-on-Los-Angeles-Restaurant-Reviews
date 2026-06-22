# =============================================================================
# Step 1: Setup & Data Loading
# Text Analytics on Los Angeles Restaurant Reviews
# =============================================================================

# Install any missing packages
pkgs <- c(
  "tidyverse", "tidytext", "textstem", "textclean",
  "wordcloud", "RColorBrewer",
  "topicmodels", "tm", "SnowballC",
  "word2vec", "text2vec", "textdata",
  "glmnet", "ranger",
  "umap", "sentimentr",
  "scales", "ggrepel", "patchwork"
)

installed <- rownames(installed.packages())
to_install <- pkgs[!pkgs %in% installed]
if (length(to_install) > 0) {
  message("Installing: ", paste(to_install, collapse = ", "))
  install.packages(to_install)
}

# Load libraries
suppressPackageStartupMessages({
  library(tidyverse)
  library(tidytext)
  library(textstem)
  library(textclean)
  library(wordcloud)
  library(RColorBrewer)
  library(topicmodels)
  library(tm)
  library(word2vec)
  library(text2vec)
  library(textdata)
  library(sentimentr)
  library(glmnet)
  library(ranger)
  library(umap)
  library(scales)
  library(ggrepel)
  library(patchwork)
})

# Load data
# Update DATA_PATH to wherever you have the CSV stored locally
DATA_PATH <- "archive (1)/top 240 restaurants recommanded in los angeles 2.csv"

reviews_raw <- read_csv(DATA_PATH, show_col_types = FALSE)

cat("Loaded:", nrow(reviews_raw), "rows x", ncol(reviews_raw), "cols\n")

# Missingness check
cat("\nMissing values per column:\n")
reviews_raw |>
  summarise(across(everything(), ~sum(is.na(.)))) |>
  pivot_longer(everything(), names_to = "column", values_to = "n_missing") |>
  print()

# Quick overview
cat("\nStar rating range:", range(reviews_raw$StarRating), "\n")
cat("Unique restaurants:", n_distinct(reviews_raw$RestaurantName), "\n")
cat("Date range:", as.character(range(reviews_raw$CommentDate)), "\n")
