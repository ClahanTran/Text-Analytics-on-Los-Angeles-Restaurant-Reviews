# =============================================================================
# Step 4: Keyword Extraction (TF-IDF)
# Text Analytics on Los Angeles Restaurant Reviews
# =============================================================================
# Load user configuration (DATA_PATH, RESULTS_DIR, SEED, model params)
if (!exists('DATA_PATH')) source(file.path(getwd(), 'config.R'))

# Requires: tokens_lemma, reviews_clean, sentiment_full from prior steps

suppressPackageStartupMessages({
  library(tidyverse)
  library(tidytext)
  library(scales)
  library(patchwork)
  library(ggrepel)
})

RESULTS_DIR <- "results"
dir.create(RESULTS_DIR, showWarnings = FALSE)

# --- 1. TF-IDF by restaurant ----------------------------------------------
# Treats each restaurant as one "document" to surface its most distinctive words
tfidf_restaurant <- tokens_lemma |>
  count(RestaurantName, word, sort = TRUE) |>
  bind_tf_idf(word, RestaurantName, n)

cat("Top 10 highest TF-IDF terms across all restaurants:\n")
tfidf_restaurant |>
  slice_max(tf_idf, n = 10) |>
  select(RestaurantName, word, n, tf_idf) |>
  print()

# Visualise a curated sample of restaurants
sample_restaurants <- c(
  "Marugame Monzo", "Ghost Sando Shop", "JAPAN HOUSE Los Angeles",
  "Tita Lina's", "Ggor Ghap Kimbob", "LA Wangbal",
  "GRANVILLE", "Bacchus Tables", "Ubuntu", "Casa Madera - West Hollywood"
)

p_rest <- tfidf_restaurant |>
  filter(RestaurantName %in% sample_restaurants) |>
  group_by(RestaurantName) |>
  slice_max(tf_idf, n = 6) |>
  ungroup() |>
  mutate(word = fct_reorder(word, tf_idf)) |>
  ggplot(aes(x = tf_idf, y = word, fill = RestaurantName)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~RestaurantName, scales = "free_y", ncol = 2) +
  labs(title = "Top TF-IDF terms by restaurant",
       subtitle = "Words most distinctive to each restaurant",
       x = "TF-IDF", y = NULL) +
  theme(strip.text = element_text(size = 7),
        axis.text.y = element_text(size = 7))

ggsave(file.path(RESULTS_DIR, "tfidf_by_restaurant.png"),
       p_rest, width = 10, height = 12, dpi = 150)

# --- 2. Log-odds ratio: positive vs. negative reviews ---------------------
# More robust than TF-IDF for 2-group comparison
log_odds <- tokens_lemma |>
  count(sentiment_label, word) |>
  group_by(sentiment_label) |>
  mutate(total = sum(n)) |>
  ungroup() |>
  pivot_wider(names_from = sentiment_label, values_from = c(n, total),
              values_fill = 0) |>
  mutate(
    log_odds = log((n_positive + 0.5) / (total_positive + 0.5)) -
               log((n_negative + 0.5) / (total_negative + 0.5))
  ) |>
  filter((n_positive + n_negative) >= 10)

p_logodds <- bind_rows(
  log_odds |> slice_max(log_odds, n = 15) |> mutate(group = "More positive"),
  log_odds |> slice_min(log_odds, n = 15) |> mutate(group = "More negative")
) |>
  ggplot(aes(x = log_odds, y = fct_reorder(word, log_odds), fill = group)) +
  geom_col() +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey40") +
  scale_fill_manual(values = c("More positive" = "#4E79A7",
                               "More negative" = "#E15759")) +
  labs(title = "Log-odds ratio: positive vs. negative reviews",
       subtitle = "Words appearing >=10 times",
       x = "Log-odds ratio", y = NULL, fill = NULL) +
  theme(legend.position = "top")

ggsave(file.path(RESULTS_DIR, "log_odds_sentiment.png"),
       p_logodds, width = 8, height = 7, dpi = 150)

# --- 3. TF-IDF by price tier ----------------------------------------------
tfidf_price <- tokens_lemma |>
  filter(price_tier != "Unknown") |>
  count(price_tier, word, sort = TRUE) |>
  bind_tf_idf(word, price_tier, n)

p_price <- tfidf_price |>
  group_by(price_tier) |>
  slice_max(tf_idf, n = 10) |>
  ungroup() |>
  mutate(
    word = fct_reorder(word, tf_idf),
    price_tier = factor(price_tier,
                        levels = c("Budget", "Moderate", "Upscale", "Fine Dining"))
  ) |>
  ggplot(aes(x = tf_idf, y = word, fill = price_tier)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~price_tier, scales = "free_y") +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "Top TF-IDF terms by price tier",
       subtitle = "Words most distinctive to each price segment",
       x = "TF-IDF", y = NULL)

ggsave(file.path(RESULTS_DIR, "tfidf_by_price_tier.png"),
       p_price, width = 10, height = 6, dpi = 150)

# --- 4. TF-IDF by star rating band ----------------------------------------
tfidf_rating <- tokens_lemma |>
  mutate(rating_bin = case_when(
    StarRating < 4.0 ~ "Below 4.0",
    StarRating < 4.5 ~ "4.0 - 4.4",
    StarRating < 4.8 ~ "4.5 - 4.7",
    TRUE             ~ "4.8 - 5.0"
  )) |>
  count(rating_bin, word, sort = TRUE) |>
  bind_tf_idf(word, rating_bin, n)

p_rating <- tfidf_rating |>
  group_by(rating_bin) |>
  slice_max(tf_idf, n = 10) |>
  ungroup() |>
  mutate(
    word = fct_reorder(word, tf_idf),
    rating_bin = factor(rating_bin,
                        levels = c("Below 4.0","4.0 - 4.4","4.5 - 4.7","4.8 - 5.0"))
  ) |>
  ggplot(aes(x = tf_idf, y = word, fill = rating_bin)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~rating_bin, scales = "free_y") +
  scale_fill_brewer(palette = "RdYlGn", direction = 1) +
  labs(title = "Top TF-IDF terms by star rating band",
       subtitle = "Words most distinctive to each rating tier",
       x = "TF-IDF", y = NULL)

ggsave(file.path(RESULTS_DIR, "tfidf_by_rating_band.png"),
       p_rating, width = 10, height = 7, dpi = 150)

cat("All TF-IDF plots saved to", RESULTS_DIR, "\n")
