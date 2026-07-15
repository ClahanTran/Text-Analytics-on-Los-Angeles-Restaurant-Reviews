# =============================================================================
# Step 9: Statistical / ML Modeling
# Text Analytics on Los Angeles Restaurant Reviews
# =============================================================================
# Load user configuration (DATA_PATH, RESULTS_DIR, SEED, model params)
if (!exists('DATA_PATH')) source(file.path(getwd(), 'config.R'))

# Requires: sentiment_full, lda_gamma, doc_embeddings, reviews_clean,
#           cluster_df from prior steps

suppressPackageStartupMessages({
  library(tidyverse)
  library(glmnet)
  library(ranger)
  library(scales)
  library(patchwork)
})

RESULTS_DIR <- "results"
dir.create(RESULTS_DIR, showWarnings = FALSE)

# =============================================================================
# 1. Build feature matrix
#    - Sentiment scores (AFINN, Bing, NRC emotions, sentimentr)
#    - LDA topic proportions (6 topics)
#    - Word2Vec embedding PCA (top 20 PCs)
# =============================================================================

# Topic proportions (wide format)
topic_wide <- lda_gamma |>
  dplyr::mutate(topic_label =
    stringr::str_replace_all(topic_label, "[^a-zA-Z0-9]", "_")) |>
  dplyr::select(-topic) |>
  tidyr::pivot_wider(names_from = topic_label, values_from = gamma,
                     values_fill = 0)

# Sentiment features
sentiment_feats <- sentiment_full |>
  dplyr::select(comment_id, afinn_score, afinn_mean, bing_score,
                ave_sentiment, review_length,
                fear, joy, sadness, trust, anger, disgust,
                anticipation, surprise) |>
  dplyr::mutate(dplyr::across(dplyr::everything(), ~tidyr::replace_na(., 0)))

# PCA on Word2Vec document embeddings (20 components)
emb_pca   <- prcomp(doc_embeddings, center = TRUE, scale. = TRUE)
pca_scores <- dplyr::as_tibble(emb_pca$x[, 1:20]) |>
  purrr::set_names(paste0("emb_pc", 1:20)) |>
  dplyr::mutate(comment_id = as.integer(rownames(doc_embeddings)))

# Join and impute
features <- sentiment_feats |>
  dplyr::left_join(topic_wide,  by = "comment_id") |>
  dplyr::left_join(pca_scores,  by = "comment_id") |>
  dplyr::left_join(dplyr::select(reviews_clean, comment_id,
                                  StarRating, price_tier),
                   by = "comment_id") |>
  dplyr::filter(!is.na(StarRating), !is.na(emb_pc1)) |>
  dplyr::mutate(dplyr::across(dplyr::where(is.numeric), ~tidyr::replace_na(., 0)))

pred_cols <- dplyr::select(features, -comment_id, -StarRating, -price_tier) |>
  names()

cat("Feature matrix:", nrow(features), "rows x", length(pred_cols),
    "predictors\n")

# =============================================================================
# 2. Train / test split (80 / 20)
# =============================================================================
set.seed(3847)
train_idx <- sample(nrow(features), floor(0.8 * nrow(features)))
train <- features[train_idx, ]
test  <- features[-train_idx, ]

X_train <- as.matrix(train[, pred_cols])
X_test  <- as.matrix(test[,  pred_cols])
y_train_reg <- train$StarRating
y_test_reg  <- test$StarRating

# =============================================================================
# 3. TASK 1 — Regression: predict StarRating
#    Note: star ratings are restaurant-level aggregates (compressed 3.5-5.0),
#    so modest R² is expected.
# =============================================================================

# LASSO
set.seed(3847)
lasso_reg  <- glmnet::cv.glmnet(X_train, y_train_reg, alpha = 1, nfolds = 5)
lasso_pred <- as.numeric(predict(lasso_reg, X_test, s = "lambda.min"))
lasso_rmse <- sqrt(mean((lasso_pred - y_test_reg)^2))
lasso_r2   <- cor(lasso_pred, y_test_reg)^2
cat("\nLASSO regression  — RMSE:", round(lasso_rmse, 4),
    "| R2:", round(lasso_r2, 4), "\n")

# Random Forest
set.seed(3847)
rf_reg  <- ranger::ranger(StarRating ~ .,
             data = train[, c("StarRating", pred_cols)],
             num.trees = 500, importance = "impurity", seed = 3847)
rf_pred <- predict(rf_reg, test[, pred_cols])$predictions
rf_rmse <- sqrt(mean((rf_pred - y_test_reg)^2))
rf_r2   <- cor(rf_pred, y_test_reg)^2
cat("RF regression     — RMSE:", round(rf_rmse, 4),
    "| R2:", round(rf_r2, 4), "\n")
cat("Baseline RMSE (mean):", round(sqrt(mean(
  (mean(y_train_reg) - y_test_reg)^2)), 4), "\n")

# =============================================================================
# 4. TASK 2 — Classification: predict price tier (drop Unknown)
# =============================================================================
clf_data <- features |>
  dplyr::filter(price_tier != "Unknown") |>
  dplyr::mutate(price_tier = factor(price_tier,
                 levels = c("Moderate","Upscale","Fine Dining")))

set.seed(3847)
clf_idx   <- sample(nrow(clf_data), floor(0.8 * nrow(clf_data)))
clf_train <- clf_data[clf_idx, ]
clf_test  <- clf_data[-clf_idx, ]

X_clf_tr <- as.matrix(clf_train[, pred_cols])
X_clf_te <- as.matrix(clf_test[,  pred_cols])
y_tr <- clf_train$price_tier
y_te <- clf_test$price_tier

baseline_acc <- mean(y_te == names(which.max(table(y_tr))))
cat("\nBaseline accuracy (majority class):", round(baseline_acc, 3), "\n")

# LASSO multinomial
set.seed(3847)
lasso_clf      <- glmnet::cv.glmnet(X_clf_tr, y_tr,
                    family = "multinomial", alpha = 1, nfolds = 5)
lasso_clf_pred <- as.vector(predict(lasso_clf, X_clf_te,
                    s = "lambda.min", type = "class"))
cat("LASSO classification accuracy:",
    round(mean(lasso_clf_pred == y_te), 3), "\n")

# Random Forest
set.seed(3847)
rf_clf      <- ranger::ranger(price_tier ~ .,
                 data = clf_train[, c("price_tier", pred_cols)],
                 num.trees = 500, importance = "impurity",
                 probability = FALSE, seed = 3847)
rf_clf_pred <- predict(rf_clf, clf_test[, pred_cols])$predictions
rf_acc      <- mean(rf_clf_pred == y_te)
cat("RF classification accuracy:", round(rf_acc, 3), "\n")

cat("\nRF confusion matrix:\n")
print(table(Predicted = rf_clf_pred, Actual = y_te))

# =============================================================================
# 5. Feature importance plots
# =============================================================================
reg_imp <- tibble::tibble(
  feature    = names(rf_reg$variable.importance),
  importance = rf_reg$variable.importance
) |> dplyr::slice_max(importance, n = 20) |>
  dplyr::mutate(feature = forcats::fct_reorder(feature, importance))

p_imp_reg <- ggplot2::ggplot(reg_imp,
               ggplot2::aes(x = importance, y = feature)) +
  ggplot2::geom_col(fill = "#4E79A7") +
  ggplot2::labs(title = "RF importance — star rating",
                x = "Impurity reduction", y = NULL)

clf_imp <- tibble::tibble(
  feature    = names(rf_clf$variable.importance),
  importance = rf_clf$variable.importance
) |> dplyr::slice_max(importance, n = 20) |>
  dplyr::mutate(feature = forcats::fct_reorder(feature, importance))

p_imp_clf <- ggplot2::ggplot(clf_imp,
               ggplot2::aes(x = importance, y = feature)) +
  ggplot2::geom_col(fill = "#F28E2B") +
  ggplot2::labs(title = "RF importance — price tier",
                x = "Impurity reduction", y = NULL)

ggplot2::ggsave(file.path(RESULTS_DIR, "ml_feature_importance.png"),
                p_imp_reg + p_imp_clf, width = 11, height = 6, dpi = 150)

# Predicted vs actual (regression)
p_pred <- tibble::tibble(actual = y_test_reg, predicted = rf_pred) |>
  ggplot2::ggplot(ggplot2::aes(x = actual, y = predicted)) +
  ggplot2::geom_jitter(alpha = 0.3, width = 0.02, color = "#4E79A7") +
  ggplot2::geom_abline(slope = 1, intercept = 0,
                        color = "#E15759", linewidth = 1) +
  ggplot2::geom_smooth(method = "lm", se = FALSE, color = "grey40",
                        linetype = "dashed") +
  ggplot2::labs(title = "RF regression: predicted vs. actual star rating",
                subtitle = paste0("RMSE = ", round(rf_rmse, 3),
                                   " | R2 = ", round(rf_r2, 3),
                                   " | Baseline = ", round(
                                     sqrt(mean((mean(y_train_reg) -
                                                  y_test_reg)^2)), 3)),
                x = "Actual", y = "Predicted")

ggplot2::ggsave(file.path(RESULTS_DIR, "ml_regression_fit.png"),
                p_pred, width = 6, height = 5, dpi = 150)

cat("\nAll ML outputs saved to", RESULTS_DIR, "\n")

