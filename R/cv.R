#' Cross-validate a dense multilayer perceptron
#'
#' Fits `densemlp()` on each of `folds` training splits and evaluates it on
#' the held-out fold via `densemlp_metrics()`, using the same task-aware
#' defaults as `densemlp()`/`tune_densemlp()`.
#'
#' @param formula A formula specification.
#' @param data A data frame used with `formula`.
#' @param x Predictor data frame or matrix.
#' @param y Outcome vector.
#' @param task Optional task override. `"auto"` infers the task from the
#'   outcome.
#' @param folds Number of cross-validation folds.
#' @param seed Random seed for fold assignment and model fitting.
#' @param verbose Print per-fold progress.
#' @param ... Additional arguments passed to `densemlp()` for every fold
#'   (e.g. `hidden_units`, `epochs`, `lr`).
#'
#' @return A list of class `densemlp_cv` with `fold_metrics` (a data frame,
#'   one row per fold), `summary` (mean and SD per metric across folds), and
#'   `task`.
#' @export
cv_densemlp <- function(formula = NULL,
                        data = NULL,
                        x = NULL,
                        y = NULL,
                        task = c("auto", "classification", "regression"),
                        folds = 5,
                        seed = 1,
                        verbose = FALSE,
                        ...) {
  task <- match.arg(task)
  folds <- normalize_positive_integer(folds, "folds")
  if (folds < 2L) {
    abort("`folds` must be at least 2.")
  }

  using_formula <- !is.null(formula) || !is.null(data)
  using_xy <- !is.null(x) || !is.null(y)
  if (using_formula && using_xy) {
    abort("Use either the formula interface or the x/y interface, not both.")
  }
  if (using_formula) {
    if (is.null(formula) || is.null(data)) {
      abort("Both `formula` and `data` are required.")
    }
    mf <- stats::model.frame(formula, data = data, na.action = stats::na.pass)
    truth_all <- mf[[1L]]
    n <- nrow(data)
  } else if (using_xy) {
    if (is.null(x) || is.null(y)) {
      abort("Both `x` and `y` are required.")
    }
    truth_all <- y
    n <- NROW(x)
  } else {
    abort("Supply either `formula` and `data`, or `x` and `y`.")
  }
  if (folds > n) {
    abort("`folds` cannot exceed the number of rows.")
  }

  resolved_task <- infer_task(truth_all, task)

  set.seed(seed)
  fold_id <- sample(rep_len(seq_len(folds), n))

  fold_results <- vector("list", folds)
  for (k in seq_len(folds)) {
    test_idx <- which(fold_id == k)
    train_idx <- which(fold_id != k)

    if (using_formula) {
      train_data <- data[train_idx, , drop = FALSE]
      test_data <- data[test_idx, , drop = FALSE]
      fit <- densemlp(
        formula = formula, data = train_data, task = task,
        seed = seed + k, verbose = FALSE, ...
      )
      test_mf <- stats::model.frame(formula, data = test_data, na.action = stats::na.pass)
      truth <- test_mf[[1L]]
      newdata_pred <- test_data
    } else {
      x_train <- x[train_idx, , drop = FALSE]
      y_train <- y[train_idx]
      x_test <- x[test_idx, , drop = FALSE]
      truth <- y[test_idx]
      fit <- densemlp(
        x = x_train, y = y_train, task = task,
        seed = seed + k, verbose = FALSE, ...
      )
      newdata_pred <- x_test
    }

    pred_type <- if (identical(fit$task, "regression")) "response" else "class"
    estimate <- stats::predict(fit, new_data = newdata_pred, type = pred_type)
    prob <- if (identical(fit$task, "classification")) {
      stats::predict(fit, new_data = newdata_pred, type = "prob")
    } else {
      NULL
    }

    metrics <- densemlp_metrics(truth = truth, estimate = estimate, task = fit$task, prob = prob)

    if (isTRUE(verbose)) {
      cat(sprintf("Fold %d/%d done\n", k, folds))
      utils::flush.console()
    }

    fold_results[[k]] <- as.data.frame(c(list(fold = k), metrics))
  }

  fold_metrics <- do.call(rbind, fold_results)
  rownames(fold_metrics) <- NULL

  metric_names <- setdiff(names(fold_metrics), "fold")
  summary_df <- do.call(rbind, lapply(metric_names, function(m) {
    data.frame(
      metric = m,
      mean = mean(fold_metrics[[m]], na.rm = TRUE),
      sd = stats::sd(fold_metrics[[m]], na.rm = TRUE)
    )
  }))
  rownames(summary_df) <- NULL

  structure(
    list(
      fold_metrics = fold_metrics,
      summary = summary_df,
      task = resolved_task,
      folds = folds
    ),
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
