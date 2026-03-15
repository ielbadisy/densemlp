#' @keywords internal
`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}

#' @keywords internal
abort <- function(..., call. = FALSE) {
  stop(..., call. = call.)
}

#' @keywords internal
set_reproducible_seed <- function(seed) {
  set.seed(seed)
  torch::torch_manual_seed(seed)
}

#' @keywords internal
normalize_hidden_units <- function(hidden_units) {
  if (!is.numeric(hidden_units) || length(hidden_units) < 1L) {
    abort("`hidden_units` must be a numeric vector with at least one element.")
  }
  as.integer(hidden_units)
}

#' @keywords internal
normalize_dropout <- function(dropout, hidden_units) {
  if (!is.numeric(dropout) || any(dropout < 0) || any(dropout >= 1)) {
    abort("`dropout` must be numeric values in [0, 1).")
  }
  if (length(dropout) == 1L) {
    return(rep(dropout, length(hidden_units)))
  }
  if (length(dropout) != length(hidden_units)) {
    abort("`dropout` must be length 1 or match `hidden_units`.")
  }
  as.numeric(dropout)
}

#' @keywords internal
normalize_loss <- function(loss, task, outcome_type) {
  if (is.null(loss)) {
    if (identical(task, "regression")) {
      return("mse")
    }
    if (identical(outcome_type, "binary")) {
      return("bce_with_logits")
    }
    return("cross_entropy")
  }

  loss <- match.arg(
    loss,
    c("mse", "bce_with_logits", "cross_entropy", "focal")
  )

  if (identical(task, "regression") && !identical(loss, "mse")) {
    abort("Regression currently supports only `loss = 'mse'`.")
  }
  if (!identical(task, "regression") &&
      identical(outcome_type, "multiclass") &&
      identical(loss, "focal")) {
    abort("`loss = 'focal'` is currently supported only for binary classification.")
  }

  loss
}

#' @keywords internal
as_data_frame_strict <- function(x, arg = "x") {
  if (is.data.frame(x)) {
    return(x)
  }
  if (is.matrix(x)) {
    return(as.data.frame(x, stringsAsFactors = FALSE))
  }
  abort(sprintf("`%s` must be a data frame or matrix.", arg))
}

#' @keywords internal
binary_log_loss <- function(truth, prob, eps = 1e-7) {
  prob <- pmin(pmax(prob, eps), 1 - eps)
  mean(-(truth * log(prob) + (1 - truth) * log(1 - prob)))
}

#' @keywords internal
multiclass_log_loss <- function(truth, prob, eps = 1e-7) {
  prob <- pmin(pmax(prob, eps), 1 - eps)
  idx <- cbind(seq_along(truth), truth)
  mean(-log(prob[idx]))
}

#' @keywords internal
default_metric_name <- function(task, n_classes = NULL) {
  if (identical(task, "regression")) {
    return("rmse")
  }
  if (!is.null(n_classes) && n_classes > 2L) {
    return("accuracy")
  }
  "accuracy"
}

utils::globalVariables(c(
  "epoch", "train_loss", "valid_loss", "feature", "importance"
))
