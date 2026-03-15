#' @keywords internal
predict_from_matrix <- function(model, matrix, outcome, device, type) {
  model$eval()
  x_tensor <- torch::torch_tensor(matrix, dtype = torch::torch_float(), device = device)
  logits <- torch::with_no_grad({
    model(x_tensor)
  })
  logits <- as.array(logits$to(device = "cpu"))

  if (identical(outcome$task, "regression")) {
    pred <- as.numeric(logits)
    center <- outcome$y_center %||% 0
    scale <- outcome$y_scale %||% 1
    return(pred * scale + center)
  }

  if (identical(outcome$outcome_type, "binary")) {
    prob <- stats::plogis(as.numeric(logits))
    prob_mat <- cbind(1 - prob, prob)
    colnames(prob_mat) <- outcome$levels
    if (identical(type, "prob")) {
      return(prob_mat)
    }
    return(factor(outcome$levels[ifelse(prob >= 0.5, 2L, 1L)], levels = outcome$levels))
  }

  shifted <- logits - apply(logits, 1, max)
  exp_logits <- exp(shifted)
  prob_mat <- exp_logits / rowSums(exp_logits)
  colnames(prob_mat) <- outcome$levels
  if (identical(type, "prob")) {
    return(prob_mat)
  }
  factor(outcome$levels[max.col(prob_mat)], levels = outcome$levels)
}

#' Predict from a fitted MLP
#'
#' @param object A fitted `mlp_fit` object.
#' @param new_data New predictor data.
#' @param type Prediction type.
#' @param ... Unused.
#'
#' @return Predictions in a task-appropriate format.
#' @export
predict.mlp_fit <- function(object, new_data, type = NULL, ...) {
  if (is.null(type)) {
    type <- if (identical(object$task, "regression")) "response" else "class"
  }
  allowed <- if (identical(object$task, "regression")) "response" else c("class", "prob")
  if (!type %in% allowed) {
    abort(sprintf(
      "Invalid `type` for this model. Allowed values: %s.",
      paste(allowed, collapse = ", ")
    ))
  }
  processed <- apply_blueprint(object$blueprint, new_data)
  outcome <- list(
    task = object$task,
    outcome_type = object$outcome_type,
    levels = object$levels,
    y_center = object$outcome_scale$center,
    y_scale = object$outcome_scale$scale
  )
  predict_from_matrix(object$network, processed$matrix, outcome, object$device, type)
}
