#' Tune a densemlp model over a hyperparameter grid
#'
#' Fits [densemlp()] for every combination in `grid` (each repeated `repeats`
#' times with successive seeds), ranks candidates by their best internal
#' validation loss, and optionally refits the best configuration on the
#' full data.
#'
#' @param x Predictor data.frame or matrix.
#' @param y Outcome vector.
#' @param task `"auto"` infers the task from `y`, as in [densemlp()].
#' @param grid A named list of candidate values. Supported names:
#'   `hidden_units` (a list of integer vectors), `dropout` (a list of
#'   numeric vectors, recycled to `hidden_units` length), `residual`,
#'   `gated`, `ema_decay`, `lr_schedule`, `epochs`, `batch_size`, `lr`.
#'   Any name omitted falls back to a single-value default. `interaction`
#'   is intentionally not tunable here: it is numerically unstable on
#'   small/wide data (see package NEWS) and is left at `FALSE`.
#' @param validation Validation fraction used for every candidate fit.
#' @param seed Base random seed.
#' @param repeats Number of repeated seeds per candidate.
#' @param ncores Number of cores used to fit candidates in parallel (see
#'   [densemlp()]'s `ncores`).
#' @param verbose Print per-candidate progress.
#' @param refit Refit the best configuration on the supplied data.
#'
#' @return A list of class `densemlp_tuned` with `results` (one row per
#'   candidate, ranked best first by mean validation loss), `best_config`,
#'   and, when `refit = TRUE`, `best_fit`.
#' @examples
#' set.seed(1)
#' x <- data.frame(a = rnorm(80), b = rnorm(80))
#' y <- x$a - 0.5 * x$b + rnorm(80, sd = 0.1)
#' tuned <- tune_densemlp(
#'   x, y, repeats = 1,
#'   grid = list(hidden_units = list(c(8), c(16, 8)), epochs = c(20))
#' )
#' tuned$best_config
#' @export
tune_densemlp <- function(x, y, task = c("auto", "regression", "binary", "multiclass", "survival"),
                       grid = NULL, validation = 0.2, seed = 1L, repeats = 3L,
                       ncores = 1L, verbose = FALSE, refit = TRUE) {
  task <- match.arg(task)
  repeats <- as.integer(repeats)
  if (length(repeats) != 1L || is.na(repeats) || repeats < 1L) {
    stop("`repeats` must be a positive integer.", call. = FALSE)
  }

  grid_defaults <- list(
    hidden_units = list(c(32, 16)),
    dropout = list(0),
    residual = FALSE,
    gated = FALSE,
    ema_decay = 0,
    lr_schedule = "none",
    epochs = 100L,
    batch_size = 32L,
    lr = 1e-3
  )
  if (!is.null(grid)) {
    if (is.null(names(grid)) || any(names(grid) == "")) {
      stop("`grid` must be a named list.", call. = FALSE)
    }
    unknown <- setdiff(names(grid), names(grid_defaults))
    if (length(unknown) > 0L) {
      stop(sprintf("Unsupported `grid` name(s): %s.", paste(unknown, collapse = ", ")), call. = FALSE)
    }
    for (nm in names(grid)) grid_defaults[[nm]] <- grid[[nm]]
  }
  grid_full <- grid_defaults

  candidates <- expand.grid(
    residual = grid_full$residual, gated = grid_full$gated,
    ema_decay = grid_full$ema_decay, lr_schedule = grid_full$lr_schedule,
    epochs = grid_full$epochs, batch_size = grid_full$batch_size, lr = grid_full$lr,
    hidden_index = seq_along(grid_full$hidden_units),
    dropout_index = seq_along(grid_full$dropout),
    KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE
  )

  fit_candidate <- function(i) {
    candidate <- candidates[i, , drop = FALSE]
    hidden_units <- grid_full$hidden_units[[candidate$hidden_index]]
    dropout <- grid_full$dropout[[candidate$dropout_index]]
    rep_scores <- vapply(seq_len(repeats), function(rep_idx) {
      fit <- densemlp(
        x = x, y = y, task = task, hidden_units = hidden_units, dropout = dropout,
        residual = candidate$residual, gated = candidate$gated,
        ema_decay = candidate$ema_decay, lr_schedule = candidate$lr_schedule,
        epochs = candidate$epochs, batch_size = candidate$batch_size, lr = candidate$lr,
        validation = validation, seed = seed + (i - 1L) * repeats + rep_idx - 1L, verbose = FALSE
      )
      fit$valid_history[fit$best_epoch]
    }, numeric(1))

    if (isTRUE(verbose)) {
      message(sprintf("Candidate %d/%d: valid_loss = %.4f (+/- %.4f)",
                       i, nrow(candidates), mean(rep_scores), stats::sd(rep_scores)))
    }
    data.frame(
      hidden_units = paste(hidden_units, collapse = "-"),
      dropout = paste(dropout, collapse = "-"),
      residual = candidate$residual, gated = candidate$gated,
      ema_decay = candidate$ema_decay, lr_schedule = candidate$lr_schedule,
      epochs = candidate$epochs, batch_size = candidate$batch_size, lr = candidate$lr,
      score = mean(rep_scores), score_sd = stats::sd(rep_scores), repeats = repeats,
      stringsAsFactors = FALSE
    )
  }

  results <- densemlp_parallel_lapply(seq_len(nrow(candidates)), fit_candidate, ncores)
  results_df <- do.call(rbind, results)
  results_df <- results_df[order(results_df$score), , drop = FALSE]
  rownames(results_df) <- NULL
  best_row <- results_df[1, , drop = FALSE]

  best_fit <- NULL
  if (isTRUE(refit)) {
    best_fit <- densemlp(
      x = x, y = y, task = task,
      hidden_units = as.integer(strsplit(best_row$hidden_units, "-", fixed = TRUE)[[1L]]),
      dropout = as.numeric(strsplit(best_row$dropout, "-", fixed = TRUE)[[1L]]),
      residual = best_row$residual, gated = best_row$gated,
      ema_decay = best_row$ema_decay, lr_schedule = best_row$lr_schedule,
      epochs = best_row$epochs, batch_size = best_row$batch_size, lr = best_row$lr,
      validation = validation, seed = seed, verbose = FALSE
    )
  }

  structure(
    list(results = results_df, best_config = best_row, best_fit = best_fit, metric = "valid_loss"),
    class = "densemlp_tuned"
  )
}

#' @export
print.densemlp_tuned <- function(x, ...) {
  cat("densemlp hyperparameter tuning (ranked by valid_loss, lower is better)\n\n")
  print(utils::head(x$results, 5), row.names = FALSE)
  cat("\nBest config:\n")
  print(x$best_config, row.names = FALSE)
  invisible(x)
}
