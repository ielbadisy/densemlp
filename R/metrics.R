# Harrell's concordance index: the fraction of comparable, correctly
# ordered (time, risk-score) pairs among all comparable pairs. A pair
# (i, j) is comparable when the earlier time is an event; a higher risk
# score should predict a shorter survival time.
concordance_index <- function(time, event, risk) {
  n <- length(time)
  concordant <- 0
  comparable <- 0
  for (i in seq_len(n)) {
    if (event[i] != 1) next
    later <- which(time > time[i] | (time == time[i] & event == 0))
    later <- later[later != i]
    if (length(later) == 0L) next
    comparable <- comparable + length(later)
    concordant <- concordant + sum(risk[i] > risk[later]) + 0.5 * sum(risk[i] == risk[later])
  }
  if (comparable == 0L) return(NA_real_)
  concordant / comparable
}

#' Integrated Brier score for a fitted Brier-loss survival densemlp model
#'
#' The IPCW (Graf et al.) integrated Brier score, evaluated at the model's
#' own time grid (`object$survival_breaks`), using a Kaplan-Meier estimate
#' of the censoring distribution fit on `y` (i.e. on whatever data is
#' passed in -- pass the held-out set's own outcome for an honest
#' out-of-sample estimate). Lower is better; this is exactly the objective
#' [densemlp()] minimizes when fit with `task = "survival", loss = "brier"`.
#'
#' @param object A `densemlp` object fit with `task = "survival", loss = "brier"`.
#' @param newdata Predictor data to evaluate on.
#' @param y The corresponding survival outcome (`survival::Surv()` or a
#'   two-column `(time, event)` matrix/data.frame).
#'
#' @return A single numeric integrated Brier score.
#' @export
densemlp_integrated_brier_score <- function(object, newdata, y) {
  if (object$task_code != 3L || !identical(object$loss, "brier")) {
    stop("`densemlp_integrated_brier_score()` requires a model fit with `task = \"survival\", loss = \"brier\"`.", call. = FALSE)
  }
  te <- densemlp_as_time_event_matrix(y)
  breaks <- object$survival_breaks
  km <- densemlp_censoring_km(te[, 1], te[, 2], breaks)
  surv <- stats::predict(object, newdata, type = "survival")
  eps <- 1e-8

  bs <- vapply(seq_along(breaks), function(k) {
    time_le <- te[, 1] <= breaks[k]
    is_event <- time_le & te[, 2] == 1
    is_alive <- te[, 1] > breaks[k]
    contrib <- numeric(nrow(te))
    contrib[is_event] <- surv[is_event, k]^2 / pmax(km$at_time[is_event], eps)
    contrib[is_alive] <- (1 - surv[is_alive, k])^2 / pmax(km$grid[k], eps)
    mean(contrib)
  }, numeric(1))

  mean(bs)
}

#' Prediction metrics for a fitted densemlp model
#'
#' @param truth Observed outcome values, or, for `task = "survival"`, a
#'   `survival::Surv()` object or a two-column `(time, event)`
#'   matrix/data.frame.
#' @param estimate Predicted values (`type = "response"` for regression and
#'   survival, `type = "class"` for classification).
#' @param task `"regression"`, `"binary"`, `"multiclass"`, or `"survival"`.
#' @param prob Optional matrix of class probabilities (classification only),
#'   used to compute `macro_auc` and `log_loss`.
#'
#' @return A named list of metrics: `rmse`, `nrmse`, `rsq` for regression,
#'   `accuracy`, `balanced_accuracy`, `macro_auc`, `log_loss` for
#'   classification, or `concordance` (Harrell's C-index) for survival.
#' @export
densemlp_metrics <- function(truth, estimate, task, prob = NULL) {
  if (identical(task, "survival")) {
    te <- densemlp_as_time_event_matrix(truth)
    return(list(concordance = concordance_index(te[, 1], te[, 2], as.numeric(estimate))))
  }
  if (identical(task, "regression")) {
    truth <- as.numeric(truth)
    estimate <- as.numeric(estimate)
    rmse <- sqrt(mean((estimate - truth)^2))
    sd_truth <- stats::sd(truth)
    return(list(
      rmse = rmse,
      nrmse = if (sd_truth > 0) rmse / sd_truth else NA_real_,
      rsq = 1 - sum((truth - estimate)^2) / sum((truth - mean(truth))^2)
    ))
  }

  truth <- factor(truth)
  estimate <- factor(estimate, levels = levels(truth))
  accuracy <- mean(estimate == truth)
  recall <- vapply(levels(truth), function(level) {
    idx <- truth == level
    if (!any(idx)) return(NA_real_)
    mean(estimate[idx] == level)
  }, numeric(1))
  balanced_accuracy <- mean(recall, na.rm = TRUE)

  macro_auc <- NA_real_
  log_loss <- NA_real_
  if (!is.null(prob)) {
    prob <- prob[, levels(truth), drop = FALSE]
    auc_binary <- function(p, truth01) {
      n1 <- sum(truth01 == 1L); n0 <- sum(truth01 == 0L)
      if (n1 == 0L || n0 == 0L) return(NA_real_)
      ranks <- rank(p)
      (sum(ranks[truth01 == 1L]) - n1 * (n1 + 1) / 2) / (n1 * n0)
    }
    macro_auc <- mean(vapply(seq_along(levels(truth)), function(k) {
      auc_binary(prob[, k], as.integer(truth == levels(truth)[k]))
    }, numeric(1)), na.rm = TRUE)
    selected <- prob[cbind(seq_along(truth), as.integer(truth))]
    log_loss <- -mean(log(pmax(selected, 1e-12)))
  }

  list(
    accuracy = accuracy,
    balanced_accuracy = balanced_accuracy,
    macro_auc = macro_auc,
    log_loss = log_loss
  )
}
