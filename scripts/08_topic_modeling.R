# =============================================================================
# Step 8: Topic Modeling (LDA, k = 6)
# Text Analytics on Los Angeles Restaurant Reviews
# =============================================================================
# Requires: tokens_lemma, reviews_clean, sentiment_full, cluster_df

suppressPackageStartupMessages({
  library(tidyverse)
  library(tidytext)
  library(topicmodels)
  library(scales)
  library(patchwork)
})

RESULTS_DIR <- "results"
dir.create(RESULTS_DIR, showWarnings = FALSE)

# =============================================================================
# 1. Build Document-Term Matrix
# =============================================================================
min_tokens <- 5

doc_token_counts <- tokens_lemma |>
  dplyr::count(comment_id) |>
  dplyr::filter(n >= min_tokens)

dtm <- tokens_lemma |>
  dplyr::filter(comment_id %in% doc_token_counts$comment_id) |>
  dplyr::count(comment_id, word) |>
  tidytext::cast_dtm(comment_id, word, n)

cat("DTM:", nrow(dtm), "documents x", ncol(dtm), "terms\n")

# =============================================================================
# 2. Select k via held-out perplexity
# =============================================================================
set.seed(3847)
train_idx <- sample(nrow(dtm), floor(0.8 * nrow(dtm)))
dtm_train <- dtm[train_idx, ]
dtm_test  <- dtm[-train_idx, ]

perplexities <- purrr::map_dbl(c(4, 5, 6, 7, 8, 10), function(k) {
  m <- topicmodels::LDA(dtm_train, k = k, method = "Gibbs",
           control = list(seed = 3847, burnin = 500, iter = 1000, thin = 10))
  topicmodels::perplexity(m, dtm_test)
})

perp_df <- tibble::tibble(k = c(4, 5, 6, 7, 8, 10), perplexity = perplexities)
cat("\nPerplexity by k:\n"); print(perp_df)

p_perp <- ggplot2::ggplot(perp_df, ggplot2::aes(k, perplexity)) +
  ggplot2::geom_line() + ggplot2::geom_point(size = 3) +
  ggplot2::scale_x_continuous(breaks = c(4,5,6,7,8,10)) +
  ggplot2::labs(title = "LDA perplexity vs. number of topics",
                subtitle = "Lower = better fit; k=6 chosen for interpretability",
                x = "k", y = "Perplexity")

ggplot2::ggsave(file.path(RESULTS_DIR, "lda_perplexity.png"),
                p_perp, width = 7, height = 4, dpi = 150)

# =============================================================================
# 3. Fit final LDA (k = 6, full DTM)
# =============================================================================
set.seed(3847)
lda_model <- topicmodels::LDA(
  dtm, k = 6, method = "Gibbs",
  control = list(seed = 3847, burnin = 1000, iter = 2000, thin = 10)
)

# Topic labels (update if model re-runs with different seed)
topic_labels <- c(
  "1" = "Service & Wait Times",
  "2" = "Ambiance & Location",
  "3" = "Dining Experience",
  "4" = "Overall Praise",
  "5" = "Food Detail (Western)",
  "6" = "Asian Cuisine & Flavors"
)

# =============================================================================
# 4. Top terms per topic (beta)
# =============================================================================
lda_terms <- tidytext::tidy(lda_model, matrix = "beta")

top_terms <- lda_terms |>
  dplyr::group_by(topic) |>
  dplyr::slice_max(beta, n = 12) |>
  dplyr::ungroup()

p_terms <- top_terms |>
  dplyr::mutate(
    topic_label = topic_labels[as.character(topic)],
    term = forcats::fct_reorder(term, beta)
  ) |>
  ggplot2::ggplot(ggplot2::aes(x = beta, y = term, fill = topic_label)) +
  ggplot2::geom_col(show.legend = FALSE) +
  ggplot2::facet_wrap(~topic_label, scales = "free_y", ncol = 2) +
  ggplot2::scale_fill_brewer(palette = "Set2") +
  ggplot2::labs(title = "LDA topic model — top terms per topic (k = 6)",
                subtitle = "beta = probability a word belongs to that topic",
                x = "beta (word-topic probability)", y = NULL) +
  ggplot2::theme(strip.text = ggplot2::element_text(size = 8.5, face = "bold"))

ggplot2::ggsave(file.path(RESULTS_DIR, "lda_top_terms.png"),
                p_terms, width = 10, height = 8, dpi = 150)

# =============================================================================
# 5. Document-topic proportions (gamma)
# =============================================================================
lda_gamma <- tidytext::tidy(lda_model, matrix = "gamma") |>
  dplyr::mutate(
    document    = as.integer(document),
    topic_label = topic_labels[as.character(topic)]
  ) |>
  dplyr::rename(comment_id = document)

# Dominant topic per review
dominant_topic <- lda_gamma |>
  dplyr::group_by(comment_id) |>
  dplyr::slice_max(gamma, n = 1) |>
  dplyr::ungroup() |>
  dplyr::rename(dominant_topic = topic, dominant_label = topic_label,
                dominant_gamma = gamma)

topic_meta <- dominant_topic |>
  dplyr::left_join(dplyr::select(reviews_clean, comment_id, StarRating,
                                  price_tier, sentiment_label, review_length),
                   by = "comment_id") |>
  dplyr::left_join(dplyr::select(sentiment_full, comment_id, afinn_score),
                   by = "comment_id")

cat("\nDominant topic distribution:\n")
topic_meta |>
  dplyr::count(dominant_label, sort = TRUE) |>
  dplyr::mutate(pct = scales::percent(n / sum(n), .1)) |>
  print()

# =============================================================================
# 6. Business metric visualisations
# =============================================================================

# Topic mix by price tier
p_price <- lda_gamma |>
  dplyr::left_join(dplyr::select(reviews_clean, comment_id, price_tier),
                   by = "comment_id") |>
  dplyr::filter(price_tier != "Unknown") |>
  dplyr::group_by(price_tier, topic_label) |>
  dplyr::summarise(mean_gamma = mean(gamma), .groups = "drop") |>
  dplyr::mutate(price_tier = factor(price_tier,
                 levels = c("Moderate","Upscale","Fine Dining"))) |>
  ggplot2::ggplot(ggplot2::aes(x = mean_gamma,
                                y = forcats::fct_reorder(topic_label, mean_gamma),
                                fill = price_tier)) +
  ggplot2::geom_col(position = "dodge") +
  ggplot2::scale_fill_brewer(palette = "Set2") +
  ggplot2::scale_x_continuous(labels = scales::percent) +
  ggplot2::labs(title = "Mean topic proportion by price tier",
                x = "Mean gamma", y = NULL, fill = "Price tier") +
  ggplot2::theme(legend.position = "top")

# AFINN by dominant topic
p_sent <- topic_meta |>
  ggplot2::ggplot(ggplot2::aes(
    x = forcats::fct_reorder(dominant_label, afinn_score, median),
    y = afinn_score, fill = dominant_label)) +
  ggplot2::geom_boxplot(show.legend = FALSE, outlier.size = 0.5) +
  ggplot2::scale_fill_brewer(palette = "Set2") +
  ggplot2::scale_x_discrete(labels = \(x) stringr::str_wrap(x, 12)) +
  ggplot2::coord_cartesian(ylim = c(-20, 45)) +
  ggplot2::labs(title = "AFINN by dominant topic", x = NULL, y = "AFINN score")

# Star rating by dominant topic
p_rating <- topic_meta |>
  dplyr::group_by(dominant_label) |>
  dplyr::summarise(mean_rating = mean(StarRating),
                   se = sd(StarRating) / sqrt(dplyr::n()), .groups = "drop") |>
  ggplot2::ggplot(ggplot2::aes(x = mean_rating,
                                y = forcats::fct_reorder(dominant_label, mean_rating),
                                fill = dominant_label)) +
  ggplot2::geom_col(show.legend = FALSE) +
  ggplot2::geom_errorbar(ggplot2::aes(xmin = mean_rating - se,
                                       xmax = mean_rating + se), width = 0.3) +
  ggplot2::scale_fill_brewer(palette = "Set2") +
  ggplot2::coord_cartesian(xlim = c(4.0, 4.6)) +
  ggplot2::labs(title = "Mean star rating by dominant topic",
                x = "Mean star rating", y = NULL)

ggplot2::ggsave(file.path(RESULTS_DIR, "lda_business_metrics.png"),
                p_price / (p_sent + p_rating), width = 10, height = 10, dpi = 150)

# Topic-cluster heatmap
topic_cluster <- lda_gamma |>
  dplyr::left_join(dplyr::select(cluster_df, comment_id, cluster_label),
                   by = "comment_id") |>
  dplyr::filter(!is.na(cluster_label)) |>
  dplyr::group_by(cluster_label, topic_label) |>
  dplyr::summarise(mean_gamma = mean(gamma), .groups = "drop")

p_heat <- ggplot2::ggplot(
  topic_cluster,
  ggplot2::aes(x = topic_label,
               y = stringr::str_wrap(cluster_label, 15),
               fill = mean_gamma)) +
  ggplot2::geom_tile(color = "white", linewidth = 0.5) +
  ggplot2::geom_text(ggplot2::aes(label = scales::percent(mean_gamma, accuracy = 1)),
                     size = 3.5, color = "white", fontface = "bold") +
  ggplot2::scale_fill_gradient(low = "#d9e8f5", high = "#1a5276",
                                labels = scales::percent) +
  ggplot2::scale_x_discrete(labels = \(x) stringr::str_wrap(x, 11)) +
  ggplot2::labs(title = "Topic-cluster heatmap",
                subtitle = "Mean topic proportion (gamma) per review cluster",
                x = NULL, y = NULL, fill = "Mean gamma") +
  ggplot2::theme(axis.text.x = ggplot2::element_text(size = 8),
                 axis.text.y = ggplot2::element_text(size = 8))

ggplot2::ggsave(file.path(RESULTS_DIR, "lda_cluster_heatmap.png"),
                p_heat, width = 9, height = 5, dpi = 150)

cat("\nAll LDA outputs saved to", RESULTS_DIR, "\n")

