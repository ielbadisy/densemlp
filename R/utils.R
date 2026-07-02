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
  if (is.null(seed)) {
    return(invisible(NULL))
  }

  set.seed(seed)

  if (requireNamespace("torch", quietly = TRUE)) {
    if (isTRUE(torch::torch_is_installed())) {
      torch::torch_manual_seed(seed)
    }
  }

  invisible(NULL)
}

#' @keywords internal
torch_cuda_is_available_safe <- function() {
  isTRUE(tryCatch(torch::cuda_is_available(), error = function(e) FALSE))
}

#' @keywords internal
resolve_device <- function(device) {
  if (identical(device, "auto")) {
    if (torch_cuda_is_available_safe()) "cuda" else "cpu"
  } else {
    device
  }
}

#' @keywords internal
normalize_hidden_units <- function(hidden_units) {
  if (!is.numeric(hidden_units) || length(hidden_units) < 1L ||
      any(!is.finite(hidden_units)) || any(hidden_units < 1) ||
      any(hidden_units != as.integer(hidden_units))) {
    abort("`hidden_units` must be positive integer layer sizes.")
  }
  as.integer(hidden_units)
}

#' @keywords internal
normalize_dropout <- function(dropout, hidden_units) {
  if (!is.numeric(dropout) || length(dropout) < 1L ||
      any(!is.finite(dropout)) || any(dropout < 0) || any(dropout >= 1)) {
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
normalize_positive_integer <- function(x, arg) {
  if (!is.numeric(x) || length(x) != 1L || !is.finite(x) ||
      x < 1 || x != as.integer(x)) {
    abort(sprintf("`%s` must be a positive integer.", arg))
  }
  as.integer(x)
}

#' @keywords internal
normalize_nonnegative_integer <- function(x, arg) {
  if (!is.numeric(x) || length(x) != 1L || !is.finite(x) ||
      x < 0 || x != as.integer(x)) {
    abort(sprintf("`%s` must be a non-negative integer.", arg))
  }
  as.integer(x)
}

#' @keywords internal
check_scalar_number <- function(x, arg, lower = -Inf, upper = Inf,
                                lower_closed = TRUE, upper_closed = TRUE) {
  if (!is.numeric(x) || length(x) != 1L || !is.finite(x)) {
    abort(sprintf("`%s` must be a finite numeric scalar.", arg))
  }
  lower_ok <- if (isTRUE(lower_closed)) x >= lower else x > lower
  upper_ok <- if (isTRUE(upper_closed)) x <= upper else x < upper
  if (!lower_ok || !upper_ok) {
    left <- if (isTRUE(lower_closed)) "[" else "("
    right <- if (isTRUE(upper_closed)) "]" else ")"
    abort(sprintf("`%s` must be in %s%s, %s%s.", arg, left, lower, upper, right))
  }
  x
}

#' @keywords internal
normalize_input_projection <- function(input_projection) {
  if (is.null(input_projection)) {
    return(NULL)
  }
  normalize_positive_integer(input_projection, "input_projection")
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
