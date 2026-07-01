#' Compute densemlp metrics
#'
#' @param truth Ground truth values.
#' @param estimate Predicted classes for classification or numeric predictions
#'   for regression.
#' @param task Optional task override. When `NULL`, the task is inferred from
#'   `truth` and `estimate`.
#' @param prob Optional class probabilities.
#'
#' @return A named list of metrics.
#' @export
densemlp_metrics <- function(truth, estimate, task = NULL, prob = NULL) {
  if (is.null(task)) {
    task <- if (is.numeric(truth) && is.numeric(estimate)) "regression" else "classification"
  }
  if (identical(task, "regression")) {
    truth <- as.numeric(truth)
    estimate <- as.numeric(estimate)
    rss <- sum((truth - estimate)^2)
    tss <- sum((truth - mean(truth))^2)
    return(list(
      rmse = sqrt(mean((truth - estimate)^2)),
      mae = mean(abs(truth - estimate)),
      rsq = if (tss == 0) NA_real_ else 1 - rss / tss
    ))
  }

  truth <- as.factor(truth)
  estimate <- factor(estimate, levels = levels(truth))
  accuracy <- mean(truth == estimate)

  if (length(levels(truth)) == 2L && !is.null(prob)) {
    positive <- prob[, ncol(prob)]
    truth01 <- as.integer(truth == levels(truth)[2L])
    return(list(
      accuracy = accuracy,
      log_loss = binary_log_loss(truth01, positive),
      brier = mean((positive - truth01)^2)
    ))
  }

  if (!is.null(prob)) {
    truth_index <- as.integer(truth)
    return(list(
      accuracy = accuracy,
      log_loss = multiclass_log_loss(truth_index, as.matrix(prob))
    ))
  }

  list(accuracy = accuracy)
}
