#' @keywords internal
normalize_verbose <- function(verbose) {
  if (is.logical(verbose) && length(verbose) == 1L && !is.na(verbose)) {
    return(if (isTRUE(verbose)) 1L else 0L)
  }
  if (is.numeric(verbose) && length(verbose) == 1L && !is.na(verbose) &&
      verbose == as.integer(verbose) && verbose >= 0 && verbose <= 2) {
    return(as.integer(verbose))
  }
  abort("`verbose` must be `FALSE`, `TRUE`, or an integer in {0, 1, 2}.")
}

#' @keywords internal
normalize_log_every <- function(log_every) {
  normalize_positive_integer(log_every, "log_every")
}

#' @keywords internal
format_epoch_id <- function(epoch, epochs) {
  width <- nchar(as.character(epochs))
  sprintf(paste0("%0", width, "d/%d"), epoch, epochs)
}

#' @keywords internal
format_metric_name <- function(metric) {
  switch(
    metric,
    accuracy = "acc",
    metric
  )
}

#' @keywords internal
format_learning_rate <- function(lr, digits = 5) {
  formatC(lr, format = "f", digits = digits)
}

#' @keywords internal
format_train_header <- function(config) {
  c(
    "Training dense multilayer perceptron",
    sprintf("Task: %s", config$task),
    sprintf("Optimizer: %s", paste0(toupper(substr(config$optimizer, 1, 1)), substr(config$optimizer, 2, nchar(config$optimizer)))),
    sprintf("Learning rate: %s", trimws(formatC(config$lr, format = "fg", digits = 6))),
    sprintf("Epochs: %d", config$epochs),
    sprintf("Batch size: %d", config$batch_size)
  )
}

#' @keywords internal
format_epoch_log <- function(epoch, epochs, train_loss, valid_loss = NULL,
                             valid_metric = NULL, metric_name = NULL,
                             lr = NULL, show_lr = FALSE, epoch_time = NULL) {
  parts <- c(
    sprintf("Epoch %s", format_epoch_id(epoch, epochs)),
    sprintf("train_loss: %.4f", train_loss)
  )

  if (!is.null(valid_loss)) {
    parts <- c(parts, sprintf("valid_loss: %.4f", valid_loss))
  }

  if (!is.null(valid_metric)) {
    label <- if (is.null(metric_name)) "valid_metric" else sprintf("valid_%s", format_metric_name(metric_name))
    parts <- c(parts, sprintf("%s: %.4f", label, valid_metric))
  }

  if (isTRUE(show_lr) && !is.null(lr)) {
    parts <- c(parts, sprintf("lr: %s", format_learning_rate(lr)))
  }

  if (!is.null(epoch_time)) {
    parts <- c(parts, sprintf("time: %.2fs", epoch_time))
  }

  paste(parts, collapse = " | ")
}

#' @keywords internal
should_log_epoch <- function(epoch, epochs, log_every) {
  epoch == 1L || epoch == epochs || (epoch %% log_every == 0L)
}
