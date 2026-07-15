# =============================================================================
# Step 10: Business Insights & Recommendations
# Text Analytics on Los Angeles Restaurant Reviews
# =============================================================================
# Load user configuration (DATA_PATH, RESULTS_DIR, SEED, model params)
if (!exists('DATA_PATH')) source(file.path(getwd(), 'config.R'))

# Requires: topic_meta, cluster_df, restaurant_sentiment, sentiment_full,
#           tokens_lemma, lda_gamma from prior steps

suppressPackageStartupMessages({
  library(tidyverse)
  library(scales)
  library(ggrepel)
  library(patchwork)
})

RESULTS_DIR <- "results"
dir.create(RESULTS_DIR, showWarnings = FALSE)

# =============================================================================
# 1. Topic sentiment evidence
# =============================================================================
topic_evidence <- topic_meta |>
  dplyr::group_by(dominant_label) |>
  dplyr::summarise(
    n_reviews    = dplyr::n(),
    avg_rating   = round(mean(StarRating), 2),
    avg_afinn    = round(mean(afinn_score, na.rm = TRUE), 2),
    pct_positive = round(mean(sentiment_label == "positive") * 100, 1),
    avg_length   = round(mean(review_length), 0),
    .groups = "drop"
  ) |>
  dplyr::arrange(avg_afinn)

cat("Topic business metrics:\n")
print(as.data.frame(topic_evidence))

# =============================================================================
# 2. Dashboard visualisations
# =============================================================================

# 2a. Sentiment gap by topic
p_gap <- ggplot2::ggplot(
  topic_evidence,
  ggplot2::aes(x = avg_afinn,
               y = forcats::fct_reorder(dominant_label, avg_afinn),
               fill = avg_afinn)) +
  ggplot2::geom_col() +
  ggplot2::geom_text(ggplot2::aes(label = round(avg_afinn, 1)),
                     hjust = -0.2, size = 3.5) +
  ggplot2::scale_fill_gradient2(low = "#E15759", mid = "#F5CBA7",
                                 high = "#4E79A7", midpoint = 5) +
  ggplot2::scale_x_continuous(limits = c(0, 13)) +
  ggplot2::labs(title = "Mean AFINN sentiment by review topic",
                subtitle = "Service & Wait Times scores 7x lower than Food Detail",
                x = "Mean AFINN score", y = NULL) +
  ggplot2::theme(legend.position = "none")

# 2b. Cluster bubble chart
cluster_summary <- cluster_df |>
  dplyr::group_by(cluster_label) |>
  dplyr::summarise(avg_afinn  = round(mean(afinn_score, na.rm = TRUE), 1),
                   avg_rating = round(mean(StarRating), 2),
                   n          = dplyr::n(), .groups = "drop")

p_bubble <- ggplot2::ggplot(
  cluster_summary,
  ggplot2::aes(x = avg_afinn,
               y = forcats::fct_reorder(cluster_label, avg_afinn),
               size = n, color = avg_rating)) +
  ggplot2::geom_point() +
  ggplot2::scale_color_gradient(low = "#E15759", high = "#4E79A7") +
  ggplot2::scale_size_continuous(range = c(4, 12)) +
  ggplot2::geom_text_repel(
    ggplot2::aes(label = paste0("n=", n, "  \u2605", avg_rating)),
    size = 3, color = "grey30") +
  ggplot2::labs(title = "Review clusters: sentiment vs. star rating",
                x = "Mean AFINN score", y = NULL,
                color = "Avg \u2605", size = "Reviews") +
  ggplot2::theme(legend.position = "right")

ggplot2::ggsave(file.path(RESULTS_DIR, "insights_dashboard.png"),
                p_gap / p_bubble, width = 9, height = 10, dpi = 150)

# 2c. Top positive & negative word drivers
pos_words <- tokens_lemma |>
  dplyr::inner_join(tidytext::get_sentiments("bing"), by = "word") |>
  dplyr::filter(sentiment == "positive") |>
  dplyr::count(word, sort = TRUE) |>
  dplyr::slice_head(n = 8) |>
  dplyr::mutate(direction = "Positive")

neg_words <- tokens_lemma |>
  dplyr::inner_join(tidytext::get_sentiments("bing"), by = "word") |>
  dplyr::filter(sentiment == "negative") |>
  dplyr::count(word, sort = TRUE) |>
  dplyr::slice_head(n = 8) |>
  dplyr::mutate(direction = "Negative")

p_words <- dplyr::bind_rows(pos_words, neg_words) |>
  dplyr::mutate(word = forcats::fct_reorder(word, n)) |>
  ggplot2::ggplot(ggplot2::aes(x = n, y = word, fill = direction)) +
  ggplot2::geom_col() +
  ggplot2::facet_wrap(~direction, scales = "free_y") +
  ggplot2::scale_fill_manual(values = c(Positive = "#4E79A7",
                                         Negative = "#E15759")) +
  ggplot2::labs(title = "Most frequent sentiment words across all reviews",
                x = "Frequency", y = NULL) +
  ggplot2::theme(legend.position = "none")

# 2d. Restaurant sentiment vs. star rating
p_scatter <- restaurant_sentiment |>
  dplyr::filter(n_reviews >= 5) |>
  ggplot2::ggplot(ggplot2::aes(x = mean_afinn, y = mean_rating)) +
  ggplot2::geom_point(ggplot2::aes(size = n_reviews),
                      alpha = 0.5, color = "#4E79A7") +
  ggplot2::geom_smooth(method = "lm", se = TRUE,
                        color = "#E15759", linewidth = 0.8) +
  ggrepel::geom_text_repel(
    data = dplyr::filter(restaurant_sentiment, n_reviews >= 5,
                          mean_afinn > 14 | mean_afinn < 1 |
                            mean_rating < 3.9),
    ggplot2::aes(label = stringr::str_trunc(RestaurantName, 22)),
    size = 2.8, max.overlaps = 15, color = "grey30") +
  ggplot2::labs(title = "Restaurant-level: mean sentiment vs. star rating",
                x = "Mean AFINN score", y = "Star rating",
                size = "# Reviews")

ggplot2::ggsave(file.path(RESULTS_DIR, "insights_word_drivers.png"),
                p_words + p_scatter, width = 11, height = 5, dpi = 150)

# =============================================================================
# 3. Print recommendations
# =============================================================================
cat("
==============================================================
BUSINESS INSIGHTS & RECOMMENDATIONS
==============================================================

1. INVEST IN STAFF TRAINING (strongest signal)
   Finding : Service & Wait Times topic has AFINN = 1.4 vs 10.1 for
             food-focused reviews. Service Complaints cluster AFINN = 1.8.
   Action  : Train front-of-house on attentiveness and friendliness —
             the top Bing positive words. Rude/attitude reviews are
             recoverable with proactive staff behavior changes.

2. SET ACCURATE WAIT TIME EXPECTATIONS
   Finding : Word2Vec places 'quoted' and 'excessive' near 'wait'.
             Service & Wait Times topic has the lowest avg star rating
             (4.20) of all six topics.
   Action  : Publish honest wait estimates, use text-alert systems,
             and consider reservation-only policies during peak hours.

3. RESPOND TO NEGATIVE REVIEWS WITHIN 48 HOURS
   Finding : Service Complaints cluster uniquely contains 'email',
             'reply', 'scam' — customers who sought help and
             were ignored.
   Action  : Assign a team member to Yelp monitoring daily. A timely,
             personal response converts a detractor and signals to
             other readers that the restaurant cares.

4. MARKET YOUR MOST DISTINCTIVE DISH
   Finding : TF-IDF uniquely identifies each restaurant by a signature
             item. Top-rated restaurants (4.8-5.0) have the most
             niche-specific vocabulary.
   Action  : Lead social media and Yelp profiles with the dish your
             reviewers mention most distinctively — not the most
             popular, the most unique.

5. INVEST IN AMBIANCE FOR MODERATE-PRICED RESTAURANTS
   Finding : 'Ambiance & Location' is the 3rd largest topic (17%),
             dominated by Moderate-priced restaurants. Key terms:
             park, spot, seat, cute, street.
   Action  : Outdoor seating, neighborhood charm, and photogenic
             presentation are high-ROI investments for $$ restaurants
             in LA's dining culture.

6. CULTIVATE FOOD ENTHUSIAST REVIEWERS
   Finding : Detailed Food Descriptions cluster = 726 reviews (largest),
             AFINN = 8.6, avg length = 125 words. These are your
             highest-value advocates.
   Action  : Invite them to new menu tastings. Their long, specific,
             enthusiastic reviews drive discovery for new customers.

7. FOR FINE DINING: DELIVER THE FULL EXPERIENCE
   Finding : Fine Dining tier loads heavily on 'Dining Experience'
             topic (experience, staff, meal, atmosphere, perfect).
   Action  : Fine dining guests judge every touchpoint — greeting,
             pacing, presentation, farewell. Staff consistency and
             story-telling around the menu are table stakes.
==============================================================
")

cat("All insights plots saved to", RESULTS_DIR, "\n")

