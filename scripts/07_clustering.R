# =============================================================================
# Step 7: Clustering / Segmentation
# Text Analytics on Los Angeles Restaurant Reviews
# =============================================================================
# Requires: doc_embeddings, tokens_lemma, reviews_clean, sentiment_full

suppressPackageStartupMessages({
  library(tidyverse)
  library(cluster)
  library(scales)
  library(patchwork)
})

RESULTS_DIR <- "results"
dir.create(RESULTS_DIR, showWarnings = FALSE)

# Load embeddings if starting fresh
# doc_embeddings <- readRDS("data/doc_embeddings_w2v.rds")

# =============================================================================
# 1. Choose optimal k: elbow + silhouette
# =============================================================================
set.seed(3847)
sub_idx <- sample(nrow(doc_embeddings), 800)
sub_mat  <- doc_embeddings[sub_idx, ]

wss <- purrr::map_dbl(2:10, \(k)
  kmeans(sub_mat, centers = k, nstart = 15, iter.max = 100)$tot.withinss)

sil <- purrr::map_dbl(2:10, \(k) {
  km <- kmeans(sub_mat, centers = k, nstart = 15, iter.max = 100)
  mean(silhouette(km$cluster, dist(sub_mat))[, 3])
})

p_elbow <- tibble(k = 2:10, wss = wss) |>
  ggplot(aes(k, wss)) +
  geom_line() + geom_point(size = 3) +
  scale_x_continuous(breaks = 2:10) +
  labs(title = "Elbow method", x = "k", y = "Total within-cluster SS")

p_sil <- tibble(k = 2:10, sil = sil) |>
  ggplot(aes(k, sil)) +
  geom_line() + geom_point(size = 3) +
  geom_vline(xintercept = which.max(sil) + 1, linetype = "dashed",
             color = "#E15759") +
  scale_x_continuous(breaks = 2:10) +
  labs(title = "Silhouette method", x = "k", y = "Avg silhouette width")

ggsave(file.path(RESULTS_DIR, "clustering_k_selection.png"),
       p_elbow + p_sil, width = 10, height = 4.5, dpi = 150)

# Note: silhouette peaks at k=2; k=5 chosen for business interpretability
# (food quality / service / ambiance / value / experience are the known
#  dimensions of restaurant reviews in the literature)

# =============================================================================
# 2. Fit k-means (k = 5) on full embedding matrix
# =============================================================================
set.seed(3847)
km5 <- kmeans(doc_embeddings, centers = 5, nstart = 25, iter.max = 200)

cat("Cluster sizes:\n")
sort(table(km5$cluster), decreasing = TRUE) |> print()

cluster_df <- tibble(
  comment_id = as.integer(rownames(doc_embeddings)),
  cluster    = factor(km5$cluster)
) |>
  dplyr::left_join(dplyr::select(reviews_clean, comment_id, RestaurantName,
                                  StarRating, price_tier, sentiment_label,
                                  review_length),
                   by = "comment_id") |>
  dplyr::left_join(dplyr::select(sentiment_full, comment_id, afinn_score,
                                  ave_sentiment),
                   by = "comment_id")

# =============================================================================
# 3. Label clusters via TF-IDF
# =============================================================================
cluster_tokens <- tokens_lemma |>
  dplyr::left_join(dplyr::select(cluster_df, comment_id, cluster),
                   by = "comment_id") |>
  dplyr::filter(!is.na(cluster))

tfidf_cluster <- cluster_tokens |>
  dplyr::count(cluster, word, sort = TRUE) |>
  tidytext::bind_tf_idf(word, cluster, n)

top_cluster_terms <- tfidf_cluster |>
  dplyr::group_by(cluster) |>
  dplyr::slice_max(tf_idf, n = 10) |>
  dplyr::ungroup()

# Interpretive labels (update if clusters change with different seeds)
cluster_labels <- c(
  "1" = "Fine Dining & Upscale",
  "2" = "Detailed Food Descriptions",
  "3" = "Service Complaints",
  "4" = "Noodle & Asian Cuisine",
  "5" = "Short Casual Reviews"
)

cluster_df <- cluster_df |>
  dplyr::mutate(cluster_label = cluster_labels[as.character(cluster)])

cat("\nCluster profile:\n")
cluster_df |>
  dplyr::group_by(cluster_label) |>
  dplyr::summarise(
    n            = dplyr::n(),
    avg_rating   = round(mean(StarRating), 2),
    avg_afinn    = round(mean(afinn_score, na.rm = TRUE), 2),
    avg_length   = round(mean(review_length), 1),
    pct_positive = scales::percent(mean(sentiment_label == "positive"), .1)
  ) |> print()

# =============================================================================
# 4. Visualisations
# =============================================================================

# 4a. UMAP with cluster colours
umap_cluster_df <- umap_doc_df |>
  dplyr::mutate(comment_id = as.integer(rownames(doc_embeddings))) |>
  dplyr::left_join(dplyr::select(cluster_df, comment_id, cluster, cluster_label),
                   by = "comment_id")

p_umap <- ggplot2::ggplot(umap_cluster_df,
                           ggplot2::aes(x = x, y = y, color = cluster_label)) +
  ggplot2::geom_point(alpha = 0.45, size = 1) +
  ggplot2::scale_color_brewer(palette = "Set1") +
  ggplot2::labs(title = "K-means clusters (k=5) on UMAP of review embeddings",
                subtitle = "Clusters identified by top TF-IDF terms",
                x = "UMAP 1", y = "UMAP 2", color = "Cluster") +
  ggplot2::guides(color = ggplot2::guide_legend(
    override.aes = list(size = 3, alpha = 1)))

ggplot2::ggsave(file.path(RESULTS_DIR, "clustering_umap.png"),
                p_umap, width = 9, height = 6, dpi = 150)

# 4b. Top TF-IDF terms per cluster
p_terms <- top_cluster_terms |>
  dplyr::mutate(
    cluster_label = cluster_labels[as.character(cluster)],
    word = forcats::fct_reorder(word, tf_idf)
  ) |>
  ggplot2::ggplot(ggplot2::aes(x = tf_idf, y = word, fill = cluster_label)) +
  ggplot2::geom_col(show.legend = FALSE) +
  ggplot2::facet_wrap(~cluster_label, scales = "free_y", ncol = 2) +
  ggplot2::scale_fill_brewer(palette = "Set1") +
  ggplot2::labs(title = "Top TF-IDF terms per cluster",
                x = "TF-IDF", y = NULL) +
  ggplot2::theme(strip.text = ggplot2::element_text(size = 8))

ggplot2::ggsave(file.path(RESULTS_DIR, "clustering_top_terms.png"),
                p_terms, width = 10, height = 8, dpi = 150)

# 4c. Business metrics by cluster
p_rating <- cluster_df |>
  ggplot2::ggplot(ggplot2::aes(x = cluster_label, y = StarRating,
                                fill = cluster_label)) +
  ggplot2::geom_boxplot(show.legend = FALSE) +
  ggplot2::scale_fill_brewer(palette = "Set1") +
  ggplot2::scale_x_discrete(labels = \(x) stringr::str_wrap(x, 12)) +
  ggplot2::labs(title = "Star rating by cluster", x = NULL, y = "Star rating")

p_afinn <- cluster_df |>
  ggplot2::ggplot(ggplot2::aes(x = cluster_label, y = afinn_score,
                                fill = cluster_label)) +
  ggplot2::geom_boxplot(show.legend = FALSE, outlier.size = 0.5) +
  ggplot2::scale_fill_brewer(palette = "Set1") +
  ggplot2::scale_x_discrete(labels = \(x) stringr::str_wrap(x, 12)) +
  ggplot2::coord_cartesian(ylim = c(-30, 50)) +
  ggplot2::labs(title = "AFINN sentiment by cluster", x = NULL,
                y = "AFINN score")

p_price_cl <- cluster_df |>
  dplyr::filter(price_tier != "Unknown") |>
  dplyr::count(cluster_label, price_tier) |>
  dplyr::group_by(cluster_label) |>
  dplyr::mutate(pct = n / sum(n)) |>
  ggplot2::ggplot(ggplot2::aes(x = pct,
                                y = stringr::str_wrap(cluster_label, 15),
                                fill = price_tier)) +
  ggplot2::geom_col() +
  ggplot2::scale_fill_brewer(palette = "Set2") +
  ggplot2::scale_x_continuous(labels = scales::percent) +
  ggplot2::labs(title = "Price tier mix by cluster",
                x = "% of reviews", y = NULL, fill = "Price tier")

ggplot2::ggsave(file.path(RESULTS_DIR, "clustering_metrics.png"),
                (p_rating + p_afinn) / p_price_cl,
                width = 10, height = 8, dpi = 150)

cat("All clustering outputs saved to", RESULTS_DIR, "\n")

