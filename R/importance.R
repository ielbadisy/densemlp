#' Permutation variable importance
#'
#' @param object A fitted `densemlp_fit` object.
#' @param new_data Evaluation data frame.
#' @param truth Ground truth outcome values.
#' @param metric Metric name. Defaults to accuracy for classification and RMSE
#'   for regression.
#' @param seed Random seed used for shuffling.
#'
#' @return A `densemlp_importance` object.
#' @export
perm_importance <- function(object, new_data, truth, metric = NULL, seed = object$seed) {
  metric <- metric %||% if (identical(object$task, "regression")) "rmse" else "accuracy"
  set.seed(seed)
  baseline <- score_metric(object, new_data, truth, metric)
  scores <- lapply(object$feature_names, function(feature) {
    shuffled <- new_data
    shuffled[[feature]] <- sample(shuffled[[feature]])
    permuted <- score_metric(object, shuffled, truth, metric)
    importance <- if (identical(metric, "rmse")) permuted - baseline else baseline - permuted
    data.frame(feature = feature, importance = importance)
  })
  out <- do.call(rbind, scores)
  out <- out[order(out$importance, decreasing = TRUE), ]
  rownames(out) <- NULL
  structure(
    list(data = out, metric = metric, baseline = baseline),
    class = "densemlp_importance"
  )
}

#' @keywords internal
score_metric <- function(object, new_data, truth, metric) {
  if (identical(object$task, "regression")) {
    pred <- stats::predict(object, new_data, type = "response")
    metrics <- densemlp_metrics(truth, pred, task = "regression")
    return(metrics[[metric]])
  }
  if (identical(metric, "accuracy")) {
    pred <- stats::predict(object, new_data, type = "class")
    return(densemlp_metrics(truth, pred, task = "classification")$accuracy)
  }
  prob <- stats::predict(object, new_data, type = "prob")
  pred <- factor(colnames(prob)[max.col(prob)], levels = levels(as.factor(truth)))
  metrics <- densemlp_metrics(truth, pred, task = "classification", prob = prob)
  metrics[[metric]]
}

#' Plot permutation importance
#'
#' @param x A `densemlp_importance` object.
#' @param ... Unused.
#'
#' @return A ggplot object.
#' @export
#' @importFrom ggplot2 aes geom_col labs theme_minimal
plot.densemlp_importance <- function(x, ...) {
  ggplot2::ggplot(x$data, ggplot2::aes(x = stats::reorder(feature, importance), y = importance)) +
    ggplot2::geom_col(fill = "#0C7BDC") +
    ggplot2::labs(x = NULL, y = sprintf("Importance (%s)", x$metric)) +
    ggplot2::theme_minimal()
}
