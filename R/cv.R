#' Cross-validate a densemlp model
#'
#' Fits [densemlp()] on each of `folds` training splits and evaluates it on the
#' held-out fold via [densemlp_metrics()].
#'
#' @param x Predictor data.frame or matrix.
#' @param y Outcome vector.
#' @param task `"auto"` infers the task from `y`, as in [densemlp()]. Pass
#'   `"survival"` explicitly for a survival outcome.
#' @param folds Number of cross-validation folds.
#' @param seed Random seed for fold assignment; fold `k` fits with
#'   `seed = seed + k`.
#' @param ncores Number of cores used to fit folds in parallel (see
#'   [densemlp()]'s `ncores`).
#' @param verbose Print per-fold progress.
#' @param ... Additional arguments passed to [densemlp()] for every fold (e.g.
#'   `hidden_units`, `epochs`, `lr`).
#'
#' @return A list of class `densemlp_cv` with `fold_metrics` (one row per fold),
#'   `summary` (mean and SD per metric across folds), and `task`.
#' @examples
#' set.seed(1)
#' x <- data.frame(a = rnorm(60), b = rnorm(60))
#' y <- x$a - 0.5 * x$b + rnorm(60, sd = 0.1)
#' cv <- cv_densemlp(x, y, folds = 3, epochs = 20, hidden_units = c(8))
#' cv$summary
#' @export
cv_densemlp <- function(x, y, task = c("auto", "regression", "binary", "multiclass", "survival"),
                     folds = 5L, seed = 1L, ncores = 1L, verbose = FALSE, ...) {
  task <- match.arg(task)
  folds <- as.integer(folds)
  if (length(folds) != 1L || is.na(folds) || folds < 2L) {
    stop("`folds` must be at least 2.", call. = FALSE)
  }
  n <- NROW(x)
  if (folds > n) stop("`folds` cannot exceed the number of rows.", call. = FALSE)

  resolved_task <- task
  if (identical(task, "auto")) {
    resolved_task <- if (is.numeric(y)) "regression" else if (nlevels(factor(y)) <= 2L) "binary" else "multiclass"
  }

  set.seed(seed)
  fold_id <- sample(rep_len(seq_len(folds), n))

  fit_fold <- function(k) {
    test_idx <- which(fold_id == k)
    train_idx <- which(fold_id != k)
    fit <- densemlp(
      x = x[train_idx, , drop = FALSE], y = densemlp_index_outcome(y, train_idx), task = task,
      seed = seed + k, verbose = FALSE, ...
    )
    truth <- densemlp_index_outcome(y, test_idx)
    newdata <- x[test_idx, , drop = FALSE]
    if (fit$task %in% c("regression", "survival")) {
      estimate <- stats::predict(fit, newdata, type = "response")
      prob <- NULL
    } else {
      estimate <- stats::predict(fit, newdata, type = "class")
      prob <- stats::predict(fit, newdata, type = "prob")
    }
    metrics <- densemlp_metrics(truth, estimate, fit$task, prob)
    if (isTRUE(verbose)) {
      message(sprintf("Fold %d/%d done", k, folds))
    }
    as.data.frame(c(list(fold = k), metrics))
  }

  fold_results <- densemlp_parallel_lapply(seq_len(folds), fit_fold, ncores)
  fold_metrics <- do.call(rbind, fold_results)
  rownames(fold_metrics) <- NULL

  metric_names <- setdiff(names(fold_metrics), "fold")
  summary_df <- do.call(rbind, lapply(metric_names, function(m) {
    data.frame(metric = m, mean = mean(fold_metrics[[m]], na.rm = TRUE),
               sd = stats::sd(fold_metrics[[m]], na.rm = TRUE))
  }))
  rownames(summary_df) <- NULL

  structure(
    list(fold_metrics = fold_metrics, summary = summary_df, task = resolved_task, folds = folds),
    class = "densemlp_cv"
  )
}

#' @export
print.densemlp_cv <- function(x, ...) {
  cat(sprintf("densemlp cross-validation (%d folds, task: %s)\n", x$folds, x$task))
  cat("\nPer-fold metrics:\n")
  print(x$fold_metrics, row.names = FALSE)
  cat("\nSummary:\n")
  print(x$summary, row.names = FALSE)
  invisible(x)
}
