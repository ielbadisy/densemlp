#' Tune an MLP over a task-aware hyperparameter grid
#'
#' @param formula A formula specification.
#' @param data A data frame used with `formula`.
#' @param x Predictor data frame or matrix.
#' @param y Outcome vector.
#' @param task Modeling task. `"auto"` infers from the outcome.
#' @param grid A named list of candidate values.
#' @param metric Optional ranking metric. Use `"accuracy"` for classification
#'   and `"rmse"` or `"valid_loss"` for regression.
#' @param validation Validation fraction for each fit.
#' @param early_stopping Enable early stopping during tuning.
#' @param patience Early stopping patience.
#' @param min_delta Minimum validation loss improvement.
#' @param min_epochs Minimum number of epochs before early stopping can trigger.
#' @param seed Base random seed.
#' @param repeats Number of repeated seeds per candidate.
#' @param verbose Print per-candidate progress.
#' @param device Device to use.
#' @param refit Refit the best configuration on the supplied data.
#'
#' @return A list with ranked tuning results and, when `refit = TRUE`, the best
#'   fitted model.
#' @export
tune_mlp <- function(formula = NULL,
                     data = NULL,
                     x = NULL,
                     y = NULL,
                     task = c("auto", "classification", "regression"),
                     grid = NULL,
                     metric = NULL,
                     validation = 0.2,
                     early_stopping = TRUE,
                     patience = 10,
                     min_delta = 0,
                     min_epochs = NULL,
                     seed = 1,
                     repeats = 3,
                     verbose = FALSE,
                     device = c("auto", "cpu", "cuda"),
                     refit = TRUE) {
  task <- match.arg(task)
  device <- match.arg(device)
  repeats <- normalize_positive_integer(repeats, "repeats")
  validation <- check_scalar_number(validation, "validation", lower = 0, upper = 1, lower_closed = FALSE, upper_closed = FALSE)
  patience <- normalize_nonnegative_integer(patience, "patience")
  min_delta <- check_scalar_number(min_delta, "min_delta", lower = 0)
  if (!is.null(min_epochs)) {
    min_epochs <- normalize_positive_integer(min_epochs, "min_epochs")
  }

  using_formula <- !is.null(formula) || !is.null(data)
  using_xy <- !is.null(x) || !is.null(y)
  if (using_formula) {
    mf <- stats::model.frame(formula, data = data, na.action = stats::na.pass)
    truth <- mf[[1L]]
  } else if (using_xy) {
    truth <- y
  } else {
    abort("Supply either `formula` and `data`, or `x` and `y`.")
  }

  resolved_task <- infer_task(truth, task)
  if (is.null(grid)) {
    grid <- if (identical(resolved_task, "regression")) {
      list(
        hidden_units = list(c(64, 32), c(128, 64), c(128, 64, 32)),
        activation = c("relu", "gelu"),
        dropout = list(c(0.05, 0.05), c(0.1, 0.1), c(0.1, 0.1, 0.05)),
        batch_norm = c(TRUE),
        residual = c(FALSE, TRUE),
        gated = c(FALSE, TRUE),
        input_projection = c(32L, 64L, NA_integer_),
        epochs = c(120L, 180L),
        batch_size = c(32L, 64L),
        lr = c(5e-4, 1e-3),
        optimizer = c("adam"),
        lr_schedule = c("cosine", "step"),
        weight_decay = c(1e-5, 1e-4),
        loss = c("mse"),
        label_smoothing = c(0),
        focal_gamma = c(2)
      )
    } else {
      list(
        hidden_units = list(c(64, 32), c(128, 64), c(128, 64, 32)),
        activation = c("relu", "gelu"),
        dropout = list(c(0.05, 0.05), c(0.1, 0.1), c(0.15, 0.1, 0.05)),
        batch_norm = c(TRUE),
        residual = c(FALSE, TRUE),
        gated = c(FALSE, TRUE),
        input_projection = c(32L, 64L, NA_integer_),
        epochs = c(100L, 150L),
        batch_size = c(32L, 64L),
        lr = c(5e-4, 1e-3, 3e-3),
        optimizer = c("adam"),
        lr_schedule = c("cosine", "step"),
        weight_decay = c(1e-5, 1e-4),
        loss = c("bce_with_logits", "focal"),
        label_smoothing = c(0, 0.05),
        focal_gamma = c(2, 3)
      )
    }
  }

  if (is.null(names(grid)) || any(names(grid) == "")) {
    abort("`grid` must be a named list.")
  }

  grid_defaults <- list(
    hidden_units = list(c(64, 32)),
    activation = "relu",
    dropout = list(c(0, 0)),
    batch_norm = TRUE,
    residual = FALSE,
    gated = FALSE,
    input_projection = NA_integer_,
    epochs = if (identical(resolved_task, "regression")) 120L else 100L,
    batch_size = 32L,
    lr = 1e-3,
    optimizer = "adam",
    lr_schedule = "cosine",
    weight_decay = 0,
    loss = if (identical(resolved_task, "regression")) "mse" else "bce_with_logits",
    label_smoothing = 0,
    focal_gamma = 2
  )
  grid_full <- grid_defaults
  for (nm in names(grid)) {
    grid_full[[nm]] <- grid[[nm]]
  }

  candidates <- expand.grid(
    activation = grid_full$activation,
    batch_norm = grid_full$batch_norm,
    residual = grid_full$residual,
    gated = grid_full$gated,
    input_projection = grid_full$input_projection,
    epochs = grid_full$epochs,
    batch_size = grid_full$batch_size,
    lr = grid_full$lr,
    optimizer = grid_full$optimizer,
    lr_schedule = grid_full$lr_schedule,
    weight_decay = grid_full$weight_decay,
    loss = grid_full$loss,
    label_smoothing = grid_full$label_smoothing,
    focal_gamma = grid_full$focal_gamma,
    hidden_index = seq_along(grid_full$hidden_units),
    dropout_index = seq_along(grid_full$dropout),
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )

  metric <- metric %||% if (identical(resolved_task, "regression")) "rmse" else "accuracy"
  results <- vector("list", nrow(candidates))

  for (i in seq_len(nrow(candidates))) {
    candidate <- candidates[i, , drop = FALSE]
    hidden_units <- grid_full$hidden_units[[candidate$hidden_index]]
    dropout <- grid_full$dropout[[candidate$dropout_index]]
    rep_scores <- numeric(repeats)

    for (rep_idx in seq_len(repeats)) {
      fit <- mlp(
        formula = formula,
        data = data,
        x = x,
        y = y,
        task = task,
        hidden_units = hidden_units,
        activation = candidate$activation,
        dropout = dropout,
        batch_norm = candidate$batch_norm,
        residual = candidate$residual,
        gated = candidate$gated,
        input_projection = if (is.na(candidate$input_projection)) NULL else as.integer(candidate$input_projection),
        epochs = candidate$epochs,
        batch_size = candidate$batch_size,
        lr = candidate$lr,
        optimizer = candidate$optimizer,
        lr_schedule = candidate$lr_schedule,
        weight_decay = candidate$weight_decay,
        validation = validation,
        early_stopping = early_stopping,
        patience = patience,
        min_delta = min_delta,
        min_epochs = min_epochs %||% max(10L, floor(candidate$epochs * 0.2)),
        loss = candidate$loss,
        label_smoothing = candidate$label_smoothing,
        focal_gamma = candidate$focal_gamma,
        seed = seed + (i - 1L) * repeats + rep_idx - 1L,
        verbose = FALSE,
        device = device
      )

      rep_scores[[rep_idx]] <- if (identical(metric, "accuracy")) {
        fit$best_validation_metric
      } else if (identical(metric, "rmse")) {
        sqrt(fit$best_validation_loss) * (fit$outcome_scale$scale %||% 1)
      } else if (identical(metric, "valid_loss")) {
        fit$best_validation_loss
      } else {
        abort("Unsupported `metric`. Use `accuracy`, `rmse`, or `valid_loss`.")
      }
    }

    score_mean <- mean(rep_scores)
    score_sd <- stats::sd(rep_scores)
    if (isTRUE(verbose)) {
      cat(sprintf(
        "Candidate %d/%d: %s = %.4f (+/- %.4f)\n",
        i, nrow(candidates), metric, score_mean, score_sd
      ))
      utils::flush.console()
    }

    results[[i]] <- data.frame(
      hidden_units = paste(hidden_units, collapse = "-"),
      dropout = paste(dropout, collapse = "-"),
      activation = candidate$activation,
      batch_norm = candidate$batch_norm,
      residual = candidate$residual,
      gated = candidate$gated,
      input_projection = if (is.na(candidate$input_projection)) NA else candidate$input_projection,
      epochs = candidate$epochs,
      batch_size = candidate$batch_size,
      lr = candidate$lr,
      optimizer = candidate$optimizer,
      lr_schedule = candidate$lr_schedule,
      weight_decay = candidate$weight_decay,
      loss = candidate$loss,
      label_smoothing = candidate$label_smoothing,
      focal_gamma = candidate$focal_gamma,
      metric = metric,
      score = score_mean,
      score_sd = score_sd,
      repeats = repeats,
      stringsAsFactors = FALSE
    )
  }

  results_df <- do.call(rbind, results)
  decreasing <- identical(metric, "accuracy")
  results_df <- results_df[order(results_df$score, decreasing = decreasing), , drop = FALSE]
  rownames(results_df) <- NULL
  best_row <- results_df[1, , drop = FALSE]

  best_fit <- NULL
  if (isTRUE(refit)) {
    best_hidden <- as.integer(strsplit(best_row$hidden_units, "-", fixed = TRUE)[[1L]])
    best_dropout <- as.numeric(strsplit(best_row$dropout, "-", fixed = TRUE)[[1L]])
    best_fit <- mlp(
      formula = formula,
      data = data,
      x = x,
      y = y,
      task = task,
      hidden_units = best_hidden,
      activation = best_row$activation,
      dropout = best_dropout,
      batch_norm = best_row$batch_norm,
      residual = best_row$residual,
      gated = best_row$gated,
      input_projection = if (is.na(best_row$input_projection)) NULL else as.integer(best_row$input_projection),
      epochs = best_row$epochs,
      batch_size = best_row$batch_size,
      lr = best_row$lr,
      optimizer = best_row$optimizer,
      lr_schedule = best_row$lr_schedule,
      weight_decay = best_row$weight_decay,
      validation = validation,
      early_stopping = early_stopping,
      patience = patience,
      min_delta = min_delta,
      min_epochs = min_epochs %||% max(10L, floor(best_row$epochs * 0.2)),
      loss = best_row$loss,
      label_smoothing = best_row$label_smoothing,
      focal_gamma = best_row$focal_gamma,
      seed = seed,
      verbose = FALSE,
      device = device
    )
  }

  structure(
    list(
      results = results_df,
      best_config = best_row,
      best_fit = best_fit,
      metric = metric
    ),
    class = "mlp_tuned"
  )
}
