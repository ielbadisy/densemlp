# Harrell's concordance index: the fraction of comparable, correctly
# ordered (time, risk-score) pairs among all comparable pairs. A pair
# (i, j) is comparable when the earlier time is an event; a higher risk
# score should predict a shorter survival time.
# With case weights `w`, each pair counts w_i * w_j.
concordance_index <- function(time, event, risk, w = NULL) {
  n <- length(time)
  concordant <- 0
  comparable <- 0
  for (i in seq_len(n)) {
    if (event[i] != 1) next
    later <- which(time > time[i] | (time == time[i] & event == 0))
    later <- later[later != i]
    if (length(later) == 0L) next
    if (is.null(w)) {
      comparable <- comparable + length(later)
      concordant <- concordant + sum(risk[i] > risk[later]) + 0.5 * sum(risk[i] == risk[later])
    } else {
      wp <- w[i] * w[later]
      comparable <- comparable + sum(wp)
      concordant <- concordant + sum(wp * ((risk[i] > risk[later]) + 0.5 * (risk[i] == risk[later])))
    }
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
#' @param task `"regression"`, `"binary"`, `"multiclass"`, `"survival"`, or
#'   `"count"` (scored like regression).
#' @param prob Optional matrix of class probabilities (classification only),
#'   used to compute `macro_auc` and `log_loss`.
#' @param weights Optional non-negative case weights, one per observation.
#'   When given, every metric is its weighted version (weighted means,
#'   weighted AUC and concordance counting each pair as the product of its
#'   two weights), e.g. an importance-weighted estimate of target-population
#'   performance under covariate shift.
#'
#' @return A named list of metrics: `rmse`, `nrmse`, `rsq` for regression,
#'   `accuracy`, `balanced_accuracy`, `macro_auc`, `log_loss` for
#'   classification, or `concordance` (Harrell's C-index) for survival.
#' @export
densemlp_metrics <- function(truth, estimate, task, prob = NULL, weights = NULL) {
  if (!is.null(weights)) {
    if (!is.numeric(weights) || length(weights) != NROW(truth) ||
        any(!is.finite(weights)) || any(weights < 0) || !any(weights > 0)) {
      stop("`weights` must be finite, non-negative numbers, one per observation, not all zero.", call. = FALSE)
    }
    return(densemlp_weighted_metrics(truth, estimate, task, prob, as.numeric(weights)))
  }
  if (identical(task, "survival")) {
    te <- densemlp_as_time_event_matrix(truth)
    return(list(concordance = concordance_index(te[, 1], te[, 2], as.numeric(estimate))))
  }
  if (task %in% c("regression", "count")) {
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

# Weighted counterparts of densemlp_metrics(), used when `weights` is given.
densemlp_weighted_metrics <- function(truth, estimate, task, prob, w) {
  wmean <- function(v) sum(w * v) / sum(w)
  if (identical(task, "survival")) {
    te <- densemlp_as_time_event_matrix(truth)
    return(list(concordance = concordance_index(te[, 1], te[, 2], as.numeric(estimate), w)))
  }
  if (task %in% c("regression", "count")) {
    truth <- as.numeric(truth)
    estimate <- as.numeric(estimate)
    rmse <- sqrt(wmean((estimate - truth)^2))
    sd_truth <- sqrt(wmean((truth - wmean(truth))^2))
    return(list(
      rmse = rmse,
      nrmse = if (sd_truth > 0) rmse / sd_truth else NA_real_,
      rsq = 1 - sum(w * (truth - estimate)^2) / sum(w * (truth - wmean(truth))^2)
    ))
  }

  truth <- factor(truth)
  estimate <- factor(estimate, levels = levels(truth))
  accuracy <- wmean(estimate == truth)
  recall <- vapply(levels(truth), function(level) {
    idx <- truth == level
    if (!any(w[idx] > 0)) return(NA_real_)
    sum(w[idx] * (estimate[idx] == level)) / sum(w[idx])
  }, numeric(1))
  balanced_accuracy <- mean(recall, na.rm = TRUE)

  macro_auc <- NA_real_
  log_loss <- NA_real_
  if (!is.null(prob)) {
    prob <- prob[, levels(truth), drop = FALSE]
    # Weighted Mann-Whitney AUC: pairs (positive, negative) weighted w_i * w_j,
    # ties count one half; computed per distinct score value.
    auc_binary <- function(p, truth01) {
      w1 <- ifelse(truth01 == 1L, w, 0)
      w0 <- ifelse(truth01 == 0L, w, 0)
      if (sum(w1) == 0 || sum(w0) == 0) return(NA_real_)
      key <- sort(unique(p))
      g <- match(p, key)
      pos <- tapply(w1, factor(g, levels = seq_along(key)), sum, default = 0)
      neg <- tapply(w0, factor(g, levels = seq_along(key)), sum, default = 0)
      neg_below <- cumsum(neg) - neg
      sum(pos * (neg_below + 0.5 * neg)) / (sum(w1) * sum(w0))
    }
    macro_auc <- mean(vapply(seq_along(levels(truth)), function(k) {
      auc_binary(prob[, k], as.integer(truth == levels(truth)[k]))
    }, numeric(1)), na.rm = TRUE)
    selected <- prob[cbind(seq_along(truth), as.integer(truth))]
    log_loss <- -wmean(log(pmax(selected, 1e-12)))
  }

  list(
    accuracy = accuracy,
    balanced_accuracy = balanced_accuracy,
    macro_auc = macro_auc,
    log_loss = log_loss
  )
}
