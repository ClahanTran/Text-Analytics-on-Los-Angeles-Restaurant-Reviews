# =============================================================================
# Step 3: Exploratory Text Analysis
# Text Analytics on Los Angeles Restaurant Reviews
# =============================================================================
# Requires: reviews_clean and tokens_lemma from 02_preprocessing.R

library(patchwork)

RESULTS_DIR <- "results"
dir.create(RESULTS_DIR, showWarnings = FALSE)

# --- 3a. Review length distribution ----------------------------------------
p1 <- reviews_clean |>
  ggplot(aes(x = review_length)) +
  geom_histogram(bins = 40, fill = "#4E79A7") +
  scale_x_continuous(limits = c(0, 200)) +
  labs(title = "Review Length Distribution",
       x = "Words per Review", y = "Count")

# --- 3b. Star rating distribution (per restaurant) -------------------------
p2 <- reviews_clean |>
  distinct(RestaurantName, StarRating) |>
  ggplot(aes(x = StarRating)) +
  geom_histogram(bins = 20, fill = "#F28E2B") +
  labs(title = "Star Ratings (per Restaurant)",
       x = "Star Rating", y = "Count")

# --- 3c. Reviews by price tier ---------------------------------------------
p3 <- reviews_clean |>
  filter(price_tier != "Unknown") |>
  count(price_tier) |>
  mutate(price_tier = fct_reorder(price_tier, n)) |>
  ggplot(aes(x = n, y = price_tier)) +
  geom_col(fill = "#76B7B2") +
  labs(title = "Reviews by Price Tier", x = "Count", y = NULL)

# --- 3d. Review length vs. star rating ------------------------------------
p4 <- reviews_clean |>
  ggplot(aes(x = factor(round(StarRating, 1)), y = review_length)) +
  geom_boxplot(fill = "#E15759", outlier.size = 0.5) +
  scale_y_continuous(limits = c(0, 150)) +
  labs(title = "Review Length vs. Star Rating",
       x = "Star Rating", y = "Word Count")

eda_panel <- (p1 + p2) / (p3 + p4)
print(eda_panel)
ggsave(file.path(RESULTS_DIR, "eda_distributions.png"),
       eda_panel, width = 10, height = 7, dpi = 150)

# --- 3e. Top 25 most frequent lemmas ---------------------------------------
top_words <- tokens_lemma |>
  count(word, sort = TRUE) |>
  slice_head(n = 25)

p_top <- ggplot(top_words, aes(x = n, y = fct_reorder(word, n))) +
  geom_col(fill = "#4E79A7") +
  labs(title = "Top 25 Most Frequent Words (Lemmatized)",
       x = "Frequency", y = NULL)

print(p_top)
ggsave(file.path(RESULTS_DIR, "top_words.png"),
       p_top, width = 8, height = 6, dpi = 150)

# --- 3f. Word cloud --------------------------------------------------------
word_freq <- tokens_lemma |> count(word, sort = TRUE)

png(file.path(RESULTS_DIR, "wordcloud_all.png"),
    width = 800, height = 600)
set.seed(4821)
wordcloud(
  words        = word_freq$word,
  freq         = word_freq$n,
  max.words    = 120,
  random.order = FALSE,
  colors       = brewer.pal(8, "Dark2"),
  scale        = c(4, 0.5)
)
title("Word Cloud — All Reviews")
dev.off()

# --- 3g. Positive vs. negative word clouds ---------------------------------
for (lbl in c("positive", "negative")) {
  wf <- tokens_lemma |>
    filter(sentiment_label == lbl) |>
    count(word, sort = TRUE)

  pal <- if (lbl == "positive") brewer.pal(8, "Blues") else brewer.pal(8, "Reds")

  png(file.path(RESULTS_DIR, paste0("wordcloud_", lbl, ".png")),
      width = 800, height = 600)
  set.seed(4821)
  wordcloud(words = wf$word, freq = wf$n, max.words = 100,
            random.order = FALSE, colors = pal, scale = c(3.5, 0.4))
  title(paste("Word Cloud —", str_to_title(lbl), "Reviews"))
  dev.off()
}

cat("EDA plots saved to", RESULTS_DIR, "\n")

# --- 3h. Summary statistics ------------------------------------------------
cat("\nReview length summary (words):\n")
summary(reviews_clean$review_length) |> print()

cat("\nTop 10 most reviewed restaurants:\n")
reviews_clean |>
  count(RestaurantName, sort = TRUE) |>
  slice_head(n = 10) |>
  print()
