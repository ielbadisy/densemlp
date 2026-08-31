#' Permutation variable importance for a fitted densemlp model
#'
#' Model-agnostic permutation importance: for each predictor, the column is
#' randomly shuffled and the drop in predictive performance (relative to the
#' unpermuted baseline) is recorded. Larger values mean the model relied more
#' on that predictor.
#'
#' @param object A fitted `densemlp` object (single model; ensembles are not
#'   supported).
#' @param new_data Evaluation predictor data.
#' @param truth Ground-truth outcome for `new_data`: a numeric vector
#'   (regression), a factor/character (classification), or a
#'   `survival::Surv()` / two-column `(time, event)` object (survival).
#' @param metric Metric name understood by [densemlp_metrics()]. Defaults to
#'   `"rmse"` (regression), `"accuracy"` (classification) or `"concordance"`
#'   (survival).
#' @param seed Random seed used for the column shuffles.
#'
#' @return A `densemlp_importance` object: a list with `data` (a data frame
#'   of `feature` / `importance`, ordered by decreasing importance),
#'   `metric`, `baseline` and `task`.
#' @examples
#' set.seed(1)
#' x <- data.frame(a = rnorm(120), b = rnorm(120), c = rnorm(120))
#' y <- x$a - 0.5 * x$b + rnorm(120, sd = 0.1)
#' fit <- densemlp(x, y, epochs = 40, hidden_units = c(16))
#' perm_importance(fit, x, y)
#' @export
perm_importance <- function(object, new_data, truth, metric = NULL,
                            seed = object$seed) {
  if (!inherits(object, "densemlp")) {
    stop("`object` must be a <densemlp> fit.", call. = FALSE)
  }
  if (!is.null(object$members)) {
    stop("`perm_importance()` does not support ensembles; use an individual member.", call. = FALSE)
  }
  task <- object$task
  metric <- metric %||% switch(task,
    regression = "rmse",
    survival = "concordance",
    "accuracy"
  )
  error_metric <- metric %in% c("rmse", "nrmse", "log_loss")

  if (is.null(seed)) seed <- 1L
  set.seed(seed)
  features <- object$blueprint$feature_names
  baseline <- score_densemlp_metric(object, new_data, truth, metric, task)

  rows <- lapply(features, function(feature) {
    shuffled <- new_data
    shuffled[[feature]] <- sample(shuffled[[feature]])
    permuted <- score_densemlp_metric(object, shuffled, truth, metric, task)
    importance <- if (error_metric) permuted - baseline else baseline - permuted
    data.frame(feature = feature, importance = importance)
  })
  out <- do.call(rbind, rows)
  out <- out[order(out$importance, decreasing = TRUE), , drop = FALSE]
  rownames(out) <- NULL
  structure(
    list(data = out, metric = metric, baseline = baseline, task = task),
    class = "densemlp_importance"
  )
}

#' @keywords internal
score_densemlp_metric <- function(object, new_data, truth, metric, task) {
  if (identical(task, "regression")) {
    pred <- stats::predict(object, new_data, type = "response")
    return(densemlp_metrics(truth, pred, task = "regression")[[metric]])
  }
  if (identical(task, "survival")) {
    risk <- stats::predict(object, new_data, type = "response")
    return(densemlp_metrics(truth, risk, task = "survival")[[metric]])
  }
  if (metric %in% c("macro_auc", "log_loss")) {
    prob <- stats::predict(object, new_data, type = "prob")
    pred <- factor(colnames(prob)[max.col(prob, ties.method = "first")],
                   levels = colnames(prob))
    return(densemlp_metrics(truth, pred, task = "multiclass", prob = prob)[[metric]])
  }
  pred <- stats::predict(object, new_data, type = "class")
  densemlp_metrics(truth, pred, task = "multiclass")[[metric]]
}

#' @rdname perm_importance
#' @param x A `densemlp_importance` object.
#' @param ... Unused.
#' @export
print.densemlp_importance <- function(x, ...) {
  cat(sprintf("<densemlp_importance> metric: %s | baseline: %.4f\n",
              x$metric, x$baseline))
  print(utils::head(x$data, 20L))
  invisible(x)
}

#' Plot permutation importance
#'
#' @param x A `densemlp_importance` object.
#' @param top Number of top features to show.
#' @param ... Passed to [graphics::barplot()].
#'
#' @return `x`, invisibly.
#' @export
plot.densemlp_importance <- function(x, top = 20L, ...) {
  d <- utils::head(x$data, top)
  d <- d[order(d$importance), , drop = FALSE]
  graphics::barplot(
    stats::setNames(d$importance, d$feature),
    horiz = TRUE, las = 1,
    xlab = sprintf("Importance (%s)", x$metric), ...
  )
  invisible(x)
}
