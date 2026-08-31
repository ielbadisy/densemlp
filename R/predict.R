#' Predict from a fitted densemlp model
#'
#' @param object A fitted `densemlp` object.
#' @param newdata New predictor data, in the same representation used to fit.
#' @param type `"response"` (regression: unscaled prediction; binary:
#'   predicted probability of the second factor level; survival: a linear
#'   risk score, higher = riskier, for either `loss`), `"class"`
#'   (classification only: predicted factor label), `"prob"`
#'   (classification only: a matrix of class probabilities), or
#'   `"survival"` (survival with `loss = "brier"` only: the full
#'   `n_bins`-column survival-probability matrix).
#' @param ... Unused.
#' @return A numeric vector (`"response"`), a factor (`"class"`), or a
#'   matrix (`"prob"` or `"survival"`).
#' @export
predict.densemlp <- function(object, newdata, type = c("response", "class", "prob", "survival"), ...) {
  type <- match.arg(type)
  if (identical(type, "survival") && object$task_code != 3L) {
    stop("`type = \"survival\"` is only defined for `task = \"survival\"`.", call. = FALSE)
  }
  if (!is.null(object$members)) {
    if (object$task_code == 0L) {
      if (type != "response") stop("`type` must be \"response\" for regression.", call. = FALSE)
      predictions <- lapply(object$members, stats::predict, newdata = newdata, type = "response")
      return(Reduce(`+`, predictions) / length(predictions))
    }
    if (object$task_code == 3L) {
      if (type != "response") {
        stop("`type = \"survival\"` is not supported for ensembles (members may use different time grids); use \"response\" for the averaged risk score.", call. = FALSE)
      }
      predictions <- lapply(object$members, stats::predict, newdata = newdata, type = "response")
      return(Reduce(`+`, predictions) / length(predictions))
    }
    if (object$task_code == 1L) {
      probabilities <- lapply(object$members, stats::predict, newdata = newdata, type = "response")
      prob1 <- Reduce(`+`, probabilities) / length(probabilities)
      prob_mat <- cbind(1 - prob1, prob1)
      colnames(prob_mat) <- object$levels
      if (type == "prob") return(prob_mat)
      if (type == "response") return(prob1)
      return(factor(object$levels[ifelse(prob1 >= 0.5, 2L, 1L)], levels = object$levels))
    }
    if (type == "response") {
      stop("`type = \"response\"` is not defined for multiclass; use \"class\" or \"prob\".", call. = FALSE)
    }
    probabilities <- lapply(object$members, stats::predict, newdata = newdata, type = "prob")
    prob_mat <- Reduce(`+`, probabilities) / length(probabilities)
    if (type == "prob") return(prob_mat)
    return(factor(object$levels[max.col(prob_mat, ties.method = "first")], levels = object$levels))
  }

  xmat <- densemlp_apply_blueprint(object$blueprint, newdata)
  residual_weights <- object$residual_weights
  residual_biases <- object$residual_biases
  if (is.null(residual_weights)) {
    residual_weights <- lapply(object$gamma, function(x) matrix(numeric(), 0, 0))
    residual_biases <- lapply(object$gamma, function(x) numeric())
  }
  gate_weights <- object$gate_weights
  gate_biases <- object$gate_biases
  if (is.null(gate_weights)) {
    gate_weights <- lapply(object$gamma, function(x) matrix(numeric(), 0, 0))
    gate_biases <- lapply(object$gamma, function(x) numeric())
  }
  interaction_weights <- object$interaction_weights
  interaction_biases <- object$interaction_biases
  if (is.null(interaction_weights)) {
    interaction_weights <- numeric()
    interaction_biases <- numeric()
  }
  proj_weights <- object$proj_weights
  proj_biases <- object$proj_biases
  if (is.null(proj_weights)) {
    proj_weights <- matrix(numeric(), 0, 0)
    proj_biases <- numeric()
  }
  batch_norm <- if (is.null(object$batch_norm)) TRUE else isTRUE(object$batch_norm)
  input_projection <- if (is.null(object$input_projection)) 0L else as.integer(object$input_projection)
  raw <- densemlp_predict_cpp(
    xmat, object$weights, object$biases,
    object$gamma, object$beta, object$running_mean, object$running_var,
    residual_weights, residual_biases,
    gate_weights, gate_biases, interaction_weights, interaction_biases,
    proj_weights, proj_biases,
    object$task_code, isTRUE(object$residual), isTRUE(object$gated),
    isTRUE(object$interaction), batch_norm, input_projection
  )

  if (object$task_code == 0L) {
    if (type != "response") stop("`type` must be \"response\" for regression.", call. = FALSE)
    return(as.numeric(raw[, 1]) * object$y_scale + object$y_center)
  }

  if (object$task_code == 3L) {
    if (identical(object$loss, "cox")) {
      if (type != "response") stop("`type` must be \"response\" for a Cox-loss survival model.", call. = FALSE)
      return(as.numeric(raw[, 1]))
    }
    # Brier-loss discrete-time head: `raw` holds per-bin hazard logits;
    # sigmoid + cumulative product gives the (monotonically non-increasing)
    # survival curve S(t_k | x).
    hazard <- 1 / (1 + exp(-raw))
    surv <- t(apply(1 - hazard, 1, cumprod))
    colnames(surv) <- sprintf("t=%.4g", object$survival_breaks)
    if (type == "survival") return(surv)
    if (type != "response") stop("`type` must be \"response\" or \"survival\" for a Brier-loss survival model.", call. = FALSE)
    eps <- 1e-8
    return(-log(pmax(surv[, ncol(surv)], eps)))
  }

  if (object$task_code == 1L) {
    prob1 <- as.numeric(raw[, 1])
    prob_mat <- cbind(1 - prob1, prob1)
    colnames(prob_mat) <- object$levels
    if (type == "prob") return(prob_mat)
    if (type == "response") return(prob1)
    return(factor(object$levels[ifelse(prob1 >= 0.5, 2L, 1L)], levels = object$levels))
  }

  prob_mat <- raw
  colnames(prob_mat) <- object$levels
  if (type == "prob") return(prob_mat)
  if (type == "response") stop("`type = \"response\"` is not defined for multiclass; use \"class\" or \"prob\".", call. = FALSE)
  factor(object$levels[max.col(prob_mat, ties.method = "first")], levels = object$levels)
}

#' @rdname densemlp
#' @param x A `densemlp` object.
#' @param ... Unused.
#' @export
print.densemlp <- function(x, ...) {
  cat("<densemlp>\n", sep = "")
  cat("  task: ", x$task, "\n", sep = "")
  if (identical(x$task, "survival")) {
    cat("  survival loss: ", if (is.null(x$loss)) "cox" else x$loss, "\n", sep = "")
  }
  cat("  hidden_units: ", paste(x$hidden_units, collapse = "-"), "\n", sep = "")
  cat("  ensemble: ", if (is.null(x$ensemble)) 1L else x$ensemble, "\n", sep = "")
  cat("  batch_norm: ", if (is.null(x$batch_norm)) TRUE else isTRUE(x$batch_norm), "\n", sep = "")
  cat("  input_projection: ", if (is.null(x$input_projection)) "none" else x$input_projection, "\n", sep = "")
  cat("  residual: ", isTRUE(x$residual), "\n", sep = "")
  cat("  gated: ", isTRUE(x$gated), "\n", sep = "")
  cat("  interaction: ", isTRUE(x$interaction), "\n", sep = "")
  cat("  dropout: ", paste(if (is.null(x$dropout)) 0 else x$dropout, collapse = "-"), "\n", sep = "")
  cat("  EMA decay: ", if (is.null(x$ema_decay)) 0 else x$ema_decay, "\n", sep = "")
  cat("  lr schedule: ", if (is.null(x$lr_schedule)) "none" else x$lr_schedule, "\n", sep = "")
  if (!is.null(x$members)) {
    cat("  epochs run: ", paste(x$epochs_run, collapse = ", "), " (members)\n", sep = "")
    return(invisible(x))
  }
  cat("  epochs run: ", x$epochs_run, if (x$epochs_run < x$epochs) " (early stopped)" else "", "\n", sep = "")
  cat("  best epoch: ", x$best_epoch, "\n", sep = "")
  cat("  final train loss: ", utils::tail(x$train_history, 1), "\n", sep = "")
  if (isTRUE(x$validation > 0)) cat("  final valid loss: ", utils::tail(x$valid_history, 1), "\n", sep = "")
  invisible(x)
}
