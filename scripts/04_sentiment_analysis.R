# =============================================================================
# Step 3: Sentiment Analysis
# Text Analytics on Los Angeles Restaurant Reviews
# =============================================================================
# Load user configuration (DATA_PATH, RESULTS_DIR, SEED, model params)
if (!exists('DATA_PATH')) source(file.path(getwd(), 'config.R'))

# Requires: reviews_clean and tokens_lemma from 02_preprocessing.R

suppressPackageStartupMessages({
  library(tidyverse)
  library(tidytext)
  library(textdata)
  library(sentimentr)
  library(scales)
  library(patchwork)
})

# --- 1. AFINN: numeric sentiment score per comment (-5 to +5 per word) ----
afinn <- get_sentiments("afinn")

afinn_scores <- tokens_lemma |>
  inner_join(afinn, by = "word") |>
  group_by(comment_id, RestaurantName, StarRating, price_tier, sentiment_label) |>
  summarise(
    afinn_score       = sum(value),
    afinn_mean        = mean(value),
    n_sentiment_words = n(),
    .groups = "drop"
  ) |>
  mutate(
    afinn_class = case_when(
      afinn_score > 0 ~ "positive",
      afinn_score < 0 ~ "negative",
      TRUE            ~ "neutral"
    )
  )

cat("AFINN class distribution:\n")
afinn_scores |> count(afinn_class) |> mutate(pct = percent(n / sum(n))) |> print()

cat("\nAgreement with star-rating label:\n")
afinn_scores |>
  filter(afinn_class != "neutral") |>
  summarise(agreement = percent(mean(afinn_class == sentiment_label))) |>
  print()

# --- 2. Bing: positive / negative word counts per comment -----------------
bing <- get_sentiments("bing")

bing_scores <- tokens_lemma |>
  inner_join(bing, by = "word") |>
  count(comment_id, RestaurantName, StarRating, price_tier, sentiment_label, sentiment) |>
  pivot_wider(names_from = sentiment, values_from = n, values_fill = 0) |>
  mutate(
    bing_score = positive - negative,
    bing_class = case_when(
      bing_score > 0 ~ "positive",
      bing_score < 0 ~ "negative",
      TRUE           ~ "neutral"
    )
  )

# --- 3. NRC: emotion lexicon (8 emotions) ----------------------------------
nrc <- get_sentiments("nrc")

nrc_emotions <- tokens_lemma |>
  inner_join(nrc, by = "word", relationship = "many-to-many") |>
  filter(!sentiment %in% c("positive", "negative")) |>
  count(comment_id, sentiment) |>
  pivot_wider(names_from = sentiment, values_from = n, values_fill = 0)

# --- 4. sentimentr: negation-aware sentence-level sentiment ---------------
sent_sentences <- reviews_clean |>
  mutate(sentences = get_sentences(Comment))

sent_r <- with(sent_sentences,
               sentiment_by(sentences, list(comment_id, RestaurantName,
                                            StarRating, price_tier)))

# --- 5. Combine all scores ------------------------------------------------
sentiment_full <- reviews_clean |>
  select(comment_id, RestaurantName, StarRating, price_tier,
         sentiment_label, Style, review_length) |>
  left_join(afinn_scores |> select(comment_id, afinn_score, afinn_mean, afinn_class),
            by = "comment_id") |>
  left_join(bing_scores  |> select(comment_id, positive, negative, bing_score, bing_class),
            by = "comment_id") |>
  left_join(nrc_emotions, by = "comment_id") |>
  left_join(sent_r |> select(comment_id, ave_sentiment), by = "comment_id") |>
  mutate(
    afinn_score = replace_na(afinn_score, 0),
    bing_score  = replace_na(bing_score,  0)
  )

# --- 6. Visualisations ----------------------------------------------------
RESULTS_DIR <- "results"
dir.create(RESULTS_DIR, showWarnings = FALSE)

# 6a. AFINN distribution + mean by price tier
p1 <- ggplot(sentiment_full, aes(x = afinn_score, fill = sentiment_label)) +
  geom_histogram(bins = 40, position = "identity", alpha = 0.7) +
  scale_fill_manual(values = c(positive = "#4E79A7", negative = "#E15759")) +
  labs(title = "AFINN sentiment score distribution",
       subtitle = "Coloured by star-rating label (>=4.0 = positive)",
       x = "AFINN score (sum)", y = "Count", fill = NULL) +
  theme(legend.position = "top")

p2 <- sentiment_full |>
  filter(price_tier != "Unknown") |>
  group_by(price_tier) |>
  summarise(mean_afinn = mean(afinn_score, na.rm = TRUE),
            se = sd(afinn_score, na.rm = TRUE) / sqrt(n())) |>
  mutate(price_tier = fct_reorder(price_tier, mean_afinn)) |>
  ggplot(aes(x = mean_afinn, y = price_tier)) +
  geom_col(fill = "#76B7B2") +
  geom_errorbar(aes(xmin = mean_afinn - se, xmax = mean_afinn + se), width = 0.3) +
  labs(title = "Mean AFINN score by price tier", x = "Mean AFINN score", y = NULL)

p3 <- sentiment_full |>
  ggplot(aes(x = StarRating, y = afinn_score)) +
  geom_jitter(alpha = 0.2, size = 0.8, color = "#4E79A7") +
  geom_smooth(method = "lm", se = TRUE, color = "#E15759") +
  labs(title = "AFINN score vs. star rating",
       x = "Star rating", y = "AFINN score")

p4 <- tokens_lemma |>
  inner_join(bing, by = "word") |>
  count(word, sentiment, sort = TRUE) |>
  group_by(sentiment) |>
  slice_max(n, n = 10) |>
  ungroup() |>
  mutate(word = fct_reorder(word, n)) |>
  ggplot(aes(x = n, y = word, fill = sentiment)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~sentiment, scales = "free_y") +
  scale_fill_manual(values = c(positive = "#4E79A7", negative = "#E15759")) +
  labs(title = "Top Bing positive & negative words", x = "Frequency", y = NULL)

panel1 <- (p1 + p2) / (p3 + p4)
ggsave(file.path(RESULTS_DIR, "sentiment_overview.png"),
       panel1, width = 11, height = 8, dpi = 150)

# 6b. NRC emotion profile
p5 <- sentiment_full |>
  select(comment_id, sentiment_label, fear, joy, sadness, trust,
         anger, disgust, anticipation, surprise) |>
  pivot_longer(fear:surprise, names_to = "emotion", values_to = "count") |>
  group_by(emotion, sentiment_label) |>
  summarise(mean_count = mean(count, na.rm = TRUE), .groups = "drop") |>
  ggplot(aes(x = mean_count, y = fct_reorder(emotion, mean_count),
             fill = sentiment_label)) +
  geom_col(position = "dodge") +
  scale_fill_manual(values = c(positive = "#4E79A7", negative = "#E15759")) +
  labs(title = "NRC emotion profile: positive vs. negative reviews",
       x = "Mean word count per review", y = NULL, fill = NULL) +
  theme(legend.position = "top")

# 6c. Restaurant ranking by AFINN
restaurant_sentiment <- sentiment_full |>
  group_by(RestaurantName) |>
  summarise(mean_afinn = mean(afinn_score, na.rm = TRUE),
            mean_rating = mean(StarRating), n_reviews = n(), .groups = "drop")

p6 <- bind_rows(
  restaurant_sentiment |> slice_max(mean_afinn, n = 10) |> mutate(group = "Top 10"),
  restaurant_sentiment |> slice_min(mean_afinn, n = 10) |> mutate(group = "Bottom 10")
) |>
  mutate(RestaurantName = fct_reorder(RestaurantName, mean_afinn)) |>
  ggplot(aes(x = mean_afinn, y = RestaurantName, fill = group)) +
  geom_col() +
  scale_fill_manual(values = c("Top 10" = "#4E79A7", "Bottom 10" = "#E15759")) +
  labs(title = "Restaurants by mean AFINN sentiment",
       x = "Mean AFINN score", y = NULL, fill = NULL) +
  theme(legend.position = "top", axis.text.y = element_text(size = 7))

panel2 <- p5 + p6
ggsave(file.path(RESULTS_DIR, "sentiment_emotions_restaurants.png"),
       panel2, width = 11, height = 5.5, dpi = 150)

# --- 7. Key correlations --------------------------------------------------
cat("\nCorrelation (AFINN vs StarRating):     ",
    round(cor(sentiment_full$afinn_score, sentiment_full$StarRating), 3), "\n")
cat("Correlation (sentimentr vs StarRating):",
    round(cor(sentiment_full$ave_sentiment, sentiment_full$StarRating, use = "complete"), 3), "\n")

cat("\nMean sentiment by price tier:\n")
sentiment_full |>
  filter(price_tier != "Unknown") |>
  group_by(price_tier) |>
  summarise(n = n(), afinn_mean = round(mean(afinn_score), 2),
            sentr_mean = round(mean(ave_sentiment, na.rm = TRUE), 3),
            pct_positive = percent(mean(sentiment_label == "positive"))) |>
  arrange(desc(afinn_mean)) |>
  print()

cat("\nPlots saved to", RESULTS_DIR, "\n")
