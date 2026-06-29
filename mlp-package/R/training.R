#' @keywords internal
compute_validation_metric <- function(task, outcome_type, truth, logits) {
  if (identical(task, "regression")) {
    preds <- as.numeric(logits)
    return(-sqrt(mean((truth - preds)^2)))
  }

  if (identical(outcome_type, "binary")) {
    prob <- stats::plogis(as.numeric(logits))
    pred <- ifelse(prob >= 0.5, 2L, 1L)
    return(mean(pred == truth))
  }

  pred <- max.col(logits)
  mean(pred == truth)
}

#' @keywords internal
binary_focal_loss <- function(logits, targets, gamma = 2, alpha = 0.25) {
  bce <- torch::nnf_binary_cross_entropy_with_logits(logits, targets, reduction = "none")
  prob <- torch::torch_sigmoid(logits)
  pt <- prob * targets + (1 - prob) * (1 - targets)
  alpha_weight <- alpha * targets + (1 - alpha) * (1 - targets)
  loss <- alpha_weight * (1 - pt)$pow(gamma) * bce
  loss$mean()
}

#' @keywords internal
make_criterion <- function(task, outcome_type, loss_name, label_smoothing = 0, focal_gamma = 2) {
  if (identical(task, "regression")) {
    return(function(logits, targets) torch::nnf_mse_loss(logits, targets))
  }

  if (identical(outcome_type, "binary")) {
    if (identical(loss_name, "focal")) {
      return(function(logits, targets) binary_focal_loss(logits, targets, gamma = focal_gamma))
    }
    if (label_smoothing > 0) {
      return(function(logits, targets) {
        smoothed <- targets * (1 - label_smoothing) + 0.5 * label_smoothing
        torch::nnf_binary_cross_entropy_with_logits(logits, smoothed)
      })
    }
    return(function(logits, targets) torch::nnf_binary_cross_entropy_with_logits(logits, targets))
  }

  if (label_smoothing > 0) {
    return(function(logits, targets) {
      torch::nnf_cross_entropy(logits, targets, label_smoothing = label_smoothing)
    })
  }

  function(logits, targets) torch::nnf_cross_entropy(logits, targets)
}

#' @keywords internal
clone_state_dict <- function(model) {
  state <- model$state_dict()
  lapply(state, function(param) param$clone())
}

#' @keywords internal
fit_network <- function(x, y, task, outcome_type, hidden_units, activation,
                        dropout, batch_norm, residual, gated, input_projection,
                        output_dim, epochs, batch_size, lr, optimizer, lr_schedule,
                        weight_decay, validation, early_stopping, patience,
                        min_delta, min_epochs, seed, verbose, device,
                        loss_name, label_smoothing = 0, focal_gamma = 2) {
  set_reproducible_seed(seed)

  n <- nrow(x)
  valid_n <- max(1L, floor(n * validation))
  if (valid_n >= n) {
    valid_n <- max(1L, n - 1L)
  }
  indices <- sample.int(n)
  valid_idx <- indices[seq_len(valid_n)]
  train_idx <- indices[-seq_len(valid_n)]
  if (length(train_idx) < 1L) {
    abort("Training data is empty after validation split.")
  }

  x_train <- torch::torch_tensor(x[train_idx, , drop = FALSE], dtype = torch::torch_float(), device = device)
  x_valid <- torch::torch_tensor(x[valid_idx, , drop = FALSE], dtype = torch::torch_float(), device = device)

  if (identical(task, "regression")) {
    y_train <- torch::torch_tensor(matrix(y[train_idx], ncol = 1), dtype = torch::torch_float(), device = device)
    y_valid <- torch::torch_tensor(matrix(y[valid_idx], ncol = 1), dtype = torch::torch_float(), device = device)
  } else if (identical(outcome_type, "binary")) {
    y_train <- torch::torch_tensor(matrix(y[train_idx] - 1, ncol = 1), dtype = torch::torch_float(), device = device)
    y_valid <- torch::torch_tensor(matrix(y[valid_idx] - 1, ncol = 1), dtype = torch::torch_float(), device = device)
  } else {
    y_train <- torch::torch_tensor(y[train_idx], dtype = torch::torch_long(), device = device)
    y_valid <- torch::torch_tensor(y[valid_idx], dtype = torch::torch_long(), device = device)
  }

  model <- mlp_module(
    input_dim = ncol(x),
    hidden_units = hidden_units,
    output_dim = output_dim,
    activation = activation,
    dropout = dropout,
    batch_norm = batch_norm,
    residual = residual,
    gated = gated,
    input_projection = input_projection
  )
  model$to(device = device)
  opt <- build_optimizer(optimizer, model$parameters, lr, weight_decay)
  scheduler <- build_scheduler(lr_schedule, opt, epochs)
  criterion <- make_criterion(task, outcome_type, loss_name, label_smoothing, focal_gamma)

  history <- vector("list", epochs)
  best_loss <- Inf
  best_metric <- -Inf
  best_epoch <- 1L
  wait <- 0L
  best_state <- NULL

  if (isTRUE(verbose)) {
    cat(sprintf("Training %d epochs\n", epochs))
    utils::flush.console()
  }

  for (epoch in seq_len(epochs)) {
    model$train()
    batch_order <- sample.int(length(train_idx))
    batches <- split(batch_order, ceiling(seq_along(batch_order) / batch_size))
    train_losses <- numeric(length(batches))

    for (i in seq_along(batches)) {
      batch_ids <- batches[[i]]
      batch_x <- x_train[batch_ids, ]
      if (identical(task, "regression") || identical(outcome_type, "binary")) {
        batch_y <- y_train[batch_ids, ]
      } else {
        batch_y <- y_train[batch_ids]
      }
      opt$zero_grad()
      logits <- model(batch_x)
      loss <- criterion(logits, batch_y)
      loss$backward()
      torch::nn_utils_clip_grad_norm_(model$parameters, max_norm = 5)
      opt$step()
      train_losses[[i]] <- as.numeric(loss$item())
    }

    model$eval()
    torch::with_no_grad({
      valid_logits <- model(x_valid)
      valid_loss <- as.numeric(criterion(valid_logits, y_valid)$item())
      train_loss <- mean(train_losses)
      valid_metric <- compute_validation_metric(
        task = task,
        outcome_type = outcome_type,
        truth = y[valid_idx],
        logits = as.array(valid_logits$to(device = "cpu"))
      )
      current_lr <- opt$param_groups[[1]]$lr
      history[[epoch]] <- data.frame(
        epoch = epoch,
        train_loss = train_loss,
        valid_loss = valid_loss,
        valid_metric = valid_metric,
        learning_rate = current_lr
      )
      improved <- valid_loss < (best_loss - min_delta)
      if (improved) {
        best_loss <- valid_loss
        best_metric <- valid_metric
        best_epoch <- epoch
        wait <- 0L
        best_state <- clone_state_dict(model)
      } else if (epoch >= min_epochs) {
        wait <- wait + 1L
      }
      if (isTRUE(verbose)) {
        cat(sprintf(
          "Epoch %d/%d - train_loss: %.4f - valid_loss: %.4f - valid_metric: %.4f - lr: %.5f\n",
          epoch, epochs, train_loss, valid_loss, valid_metric, current_lr
        ))
        utils::flush.console()
      }
    })

    if (!is.null(scheduler)) {
      scheduler$step()
    }

    if (isTRUE(early_stopping) && epoch >= min_epochs && wait >= patience) {
      break
    }
  }

  if (!is.null(best_state)) {
    model$load_state_dict(best_state)
  }

  history <- Filter(Negate(is.null), history)

  list(
    model = model,
    history = do.call(rbind, history),
    best_epoch = best_epoch,
    best_loss = best_loss,
    best_metric = best_metric
  )
}
