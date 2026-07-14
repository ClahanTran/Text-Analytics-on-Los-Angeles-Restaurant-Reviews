# =============================================================================
# Step 5: Word Embeddings (Word2Vec + GloVe)
# Text Analytics on Los Angeles Restaurant Reviews
# =============================================================================
# Requires: reviews_clean and tokens_lemma from prior steps

suppressPackageStartupMessages({
  library(tidyverse)
  library(word2vec)
  library(text2vec)
  library(umap)
  library(ggrepel)
  library(patchwork)
})

RESULTS_DIR <- "results"
dir.create(RESULTS_DIR, showWarnings = FALSE)

# =============================================================================
# 1. Word2Vec (skip-gram, 100 dims)
# =============================================================================
corpus <- reviews_clean$text_clean

set.seed(3847)
w2v_model <- word2vec(
  x         = corpus,
  type      = "skip-gram",
  dim       = 100,
  window    = 5,
  iter      = 20,
  min_count = 3,
  threads   = 4
)

w2v_embeddings <- as.matrix(w2v_model)
cat("Word2Vec vocabulary:", nrow(w2v_embeddings), "words x",
    ncol(w2v_embeddings), "dims\n")

# Nearest-neighbor helper
get_neighbors <- function(model, term, n = 5) {
  res <- tryCatch(
    predict(model, term, type = "nearest", top_n = n + 1),
    error = function(e) NULL
  )
  if (is.null(res)) return(NULL)
  res[[1]] |> dplyr::filter(term2 != term) |> head(n) |> dplyr::mutate(query = term)
}

key_terms <- c("food", "service", "delicious", "wait", "price",
               "atmosphere", "friendly", "fresh", "disappoint", "recommend")

nearest_w2v <- purrr::map_dfr(key_terms, get_neighbors, model = w2v_model, n = 5)
cat("\nWord2Vec nearest neighbors:\n")
print(dplyr::select(nearest_w2v, query, term2, similarity) |>
        dplyr::mutate(similarity = round(similarity, 3)) |>
        as.data.frame())

# =============================================================================
# 2. GloVe (100 dims, window = 5)
# =============================================================================
tokens_list <- text2vec::space_tokenizer(reviews_clean$text_clean)
it          <- text2vec::itoken(tokens_list, progressbar = FALSE)
vocab       <- text2vec::create_vocabulary(it) |>
               text2vec::prune_vocabulary(term_count_min = 3)
vectorizer  <- text2vec::vocab_vectorizer(vocab)
tcm         <- text2vec::create_tcm(it, vectorizer, skip_grams_window = 5,
                                    progressbar = FALSE)

set.seed(3847)
glove    <- text2vec::GlobalVectors$new(rank = 100, x_max = 10,
                                        learning_rate = 0.1)
wv_main  <- glove$fit_transform(tcm, n_iter = 20, convergence_tol = 0.001,
                                 n_threads = 4)
glove_embeddings <- wv_main + t(glove$components)
cat("\nGloVe vocabulary:", nrow(glove_embeddings), "words x",
    ncol(glove_embeddings), "dims\n")

# Cosine similarity helper
cosine_sim <- function(mat, term, top_n = 5) {
  if (!term %in% rownames(mat)) return(NULL)
  vec  <- mat[term, , drop = FALSE]
  sims <- text2vec::sim2(mat, vec, method = "cosine", norm = "l2")
  data.frame(word = rownames(sims), similarity = as.numeric(sims)) |>
    dplyr::filter(word != term) |>
    dplyr::slice_max(similarity, n = top_n)
}

# =============================================================================
# 3. Document embeddings: average Word2Vec vectors per review
# =============================================================================
review_tokens <- tokens_lemma |>
  dplyr::filter(word %in% rownames(w2v_embeddings)) |>
  dplyr::group_by(comment_id) |>
  dplyr::summarise(words = list(unique(word)), .groups = "drop")

doc_embeddings <- purrr::map(review_tokens$words, function(ws) {
  colMeans(w2v_embeddings[ws, , drop = FALSE])
}) |> do.call(rbind, args = _)

rownames(doc_embeddings) <- review_tokens$comment_id
cat("\nDocument embeddings:", nrow(doc_embeddings), "reviews x",
    ncol(doc_embeddings), "dims\n")

# Save for clustering and ML steps
saveRDS(doc_embeddings,   file.path("data", "doc_embeddings_w2v.rds"))
saveRDS(w2v_embeddings,   file.path("data", "w2v_embeddings.rds"))
saveRDS(glove_embeddings, file.path("data", "glove_embeddings.rds"))

# =============================================================================
# 4. UMAP visualizations
# =============================================================================
focus_words <- tokens_lemma |>
  dplyr::count(word, sort = TRUE) |>
  dplyr::filter(n >= 30) |>
  dplyr::pull(word)
focus_words <- focus_words[focus_words %in% rownames(w2v_embeddings)]
focus_mat   <- w2v_embeddings[focus_words, ]

set.seed(3847)
umap_words <- umap::umap(focus_mat, n_neighbors = 12, min_dist = 0.1)

semantic_cats <- tibble::tribble(
  ~word,         ~category,
  "food",        "Core",        "service",     "Core",
  "restaurant",  "Core",        "delicious",   "Food quality",
  "tasty",       "Food quality","flavorful",   "Food quality",
  "fresh",       "Food quality","yummy",        "Food quality",
  "flavor",      "Food quality","dish",         "Food quality",
  "chicken",     "Food quality","bite",         "Food quality",
  "friendly",    "Service",     "attentive",   "Service",
  "helpful",     "Service",     "staff",       "Service",
  "server",      "Service",     "rude",        "Service",
  "wait",        "Experience",  "time",        "Experience",
  "line",        "Experience",  "reservation", "Experience",
  "atmosphere",  "Ambiance",    "ambiance",    "Ambiance",
  "cozy",        "Ambiance",    "beautiful",   "Ambiance",
  "price",       "Value",       "expensive",   "Value",
  "worth",       "Value",       "recommend",   "Sentiment",
  "love",        "Sentiment",   "disappoint",  "Sentiment",
  "amazing",     "Sentiment",   "excellent",   "Sentiment"
)

umap_word_df <- data.frame(
  word = focus_words,
  x    = umap_words$layout[, 1],
  y    = umap_words$layout[, 2]
) |>
  dplyr::left_join(semantic_cats, by = "word") |>
  dplyr::mutate(category = tidyr::replace_na(category, "Other"))

p_words <- ggplot2::ggplot(umap_word_df,
                            ggplot2::aes(x = x, y = y, color = category)) +
  ggplot2::geom_point(data = dplyr::filter(umap_word_df, category == "Other"),
                      color = "grey85", size = 1.5) +
  ggplot2::geom_point(data = dplyr::filter(umap_word_df, category != "Other"),
                      size = 2.5) +
  ggrepel::geom_text_repel(
    data = dplyr::filter(umap_word_df, category != "Other"),
    ggplot2::aes(label = word), size = 3, max.overlaps = 30,
    segment.color = "grey60") +
  ggplot2::scale_color_brewer(palette = "Set1") +
  ggplot2::labs(title = "UMAP projection of Word2Vec embeddings",
                subtitle = "Semantic neighborhoods of restaurant review vocabulary",
                x = "UMAP 1", y = "UMAP 2", color = "Category")

ggplot2::ggsave(file.path(RESULTS_DIR, "umap_word_embeddings.png"),
                p_words, width = 9, height = 7, dpi = 150)

# Document UMAP
doc_meta <- review_tokens |>
  dplyr::mutate(comment_id = as.integer(comment_id)) |>
  dplyr::left_join(dplyr::select(reviews_clean, comment_id, sentiment_label,
                                  price_tier, StarRating),
                   by = "comment_id")

set.seed(3847)
umap_docs   <- umap::umap(doc_embeddings, n_neighbors = 15, min_dist = 0.05)
umap_doc_df <- data.frame(
  x               = umap_docs$layout[, 1],
  y               = umap_docs$layout[, 2],
  sentiment_label = doc_meta$sentiment_label,
  price_tier      = doc_meta$price_tier
)

p_sent <- ggplot2::ggplot(umap_doc_df,
                           ggplot2::aes(x, y, color = sentiment_label)) +
  ggplot2::geom_point(alpha = 0.4, size = 0.9) +
  ggplot2::scale_color_manual(
    values = c(positive = "#4E79A7", negative = "#E15759")) +
  ggplot2::labs(title = "Review embeddings — sentiment",
                x = "UMAP 1", y = "UMAP 2", color = NULL) +
  ggplot2::theme(legend.position = "top")

p_price <- dplyr::filter(umap_doc_df, price_tier != "Unknown") |>
  ggplot2::ggplot(ggplot2::aes(x, y, color = price_tier)) +
  ggplot2::geom_point(alpha = 0.4, size = 0.9) +
  ggplot2::scale_color_brewer(palette = "Set2") +
  ggplot2::labs(title = "Review embeddings — price tier",
                x = "UMAP 1", y = "UMAP 2", color = NULL) +
  ggplot2::theme(legend.position = "top")

ggplot2::ggsave(file.path(RESULTS_DIR, "umap_doc_embeddings.png"),
                p_sent + p_price, width = 11, height = 5, dpi = 150)

cat("All outputs saved to", RESULTS_DIR, "and data/\n")

