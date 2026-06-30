#' Fit a tabular multilayer perceptron
#'
#' @param formula A formula specification.
#' @param data A data frame used with `formula`.
#' @param x Predictor data frame or matrix. Retained for backward compatibility
#'   with the x/y interface.
#' @param y Outcome vector. Retained for backward compatibility with the x/y
#'   interface.
#' @param task Modeling task. `"auto"` infers from the outcome.
#' @param hidden_units Hidden layer sizes.
#' @param activation Activation function.
#' @param dropout Dropout probability.
#' @param batch_norm Use batch normalization in hidden layers.
#' @param residual Use residual skip connections between hidden blocks.
#' @param gated Use learned gating inside hidden blocks.
#' @param input_projection Optional input projection dimension before hidden
#'   blocks.
#' @param epochs Number of epochs.
#' @param batch_size Mini-batch size.
#' @param lr Learning rate.
#' @param optimizer Optimizer name.
#' @param lr_schedule Learning-rate schedule.
#' @param weight_decay Weight decay.
#' @param validation Validation fraction.
#' @param early_stopping Enable early stopping.
#' @param patience Early stopping patience.
#' @param min_delta Minimum validation loss improvement.
#' @param min_epochs Minimum number of epochs before early stopping can trigger.
#' @param loss Loss function. `"focal"` is available for binary classification.
#' @param label_smoothing Label smoothing for classification losses.
#' @param focal_gamma Focal-loss focusing parameter.
#' @param metrics Reserved for future custom metrics.
#' @param seed Random seed.
#' @param verbose Verbosity level. `FALSE` or `0` silences output, `TRUE` or
#'   `1` prints a standard log, and `2` prints a detailed log.
#' @param log_every Epoch logging frequency.
#' @param device Device to use.
#'
#' @return An `mlp_fit` object.
#' @export
mlp <- function(formula = NULL,
                data = NULL,
                x = NULL,
                y = NULL,
                task = c("auto", "classification", "regression"),
                hidden_units = c(64, 32),
                activation = c("relu", "tanh", "gelu"),
                dropout = 0,
                batch_norm = TRUE,
                residual = FALSE,
                gated = FALSE,
                input_projection = NULL,
                epochs = 100,
                batch_size = 32,
                lr = 1e-3,
                optimizer = c("adam", "sgd"),
                lr_schedule = c("none", "cosine", "step"),
                weight_decay = 0,
                validation = 0.2,
                early_stopping = TRUE,
                patience = 10,
                min_delta = 0,
                min_epochs = max(10L, floor(epochs * 0.2)),
                loss = NULL,
                label_smoothing = 0,
                focal_gamma = 2,
                metrics = NULL,
                seed = 1,
                verbose = TRUE,
                log_every = 1,
                device = c("auto", "cpu", "cuda")) {
  activation <- match.arg(activation)
  optimizer <- match.arg(optimizer)
  lr_schedule <- match.arg(lr_schedule)
  device <- match.arg(device)
  task <- match.arg(task)
  hidden_units <- normalize_hidden_units(hidden_units)
  dropout <- normalize_dropout(dropout, hidden_units)
  input_projection <- normalize_input_projection(input_projection)
  epochs <- normalize_positive_integer(epochs, "epochs")
  batch_size <- normalize_positive_integer(batch_size, "batch_size")
  lr <- check_scalar_number(lr, "lr", lower = 0, lower_closed = FALSE)
  weight_decay <- check_scalar_number(weight_decay, "weight_decay", lower = 0)
  validation <- check_scalar_number(validation, "validation", lower = 0, upper = 1, lower_closed = FALSE, upper_closed = FALSE)
  patience <- normalize_nonnegative_integer(patience, "patience")
  min_delta <- check_scalar_number(min_delta, "min_delta", lower = 0)
  min_epochs <- normalize_positive_integer(min_epochs, "min_epochs")
  label_smoothing <- check_scalar_number(label_smoothing, "label_smoothing", lower = 0, upper = 1, upper_closed = FALSE)
  focal_gamma <- check_scalar_number(focal_gamma, "focal_gamma", lower = 0)

  using_formula <- !is.null(formula) || !is.null(data)
  using_xy <- !is.null(x) || !is.null(y)
  if (using_formula && using_xy) {
    abort("Use either the formula interface or the x/y interface, not both.")
  }
  if (!using_formula && !using_xy) {
    abort("Supply either `formula` and `data`, or `x` and `y`.")
  }

  if (using_formula) {
    if (is.null(formula) || is.null(data)) {
      abort("Both `formula` and `data` are required.")
    }
    mf <- stats::model.frame(formula, data = data, na.action = stats::na.pass)
    outcome_name <- names(mf)[1L]
    y <- mf[[1L]]
    x <- mf[-1L]
  } else {
    if (is.null(x) || is.null(y)) {
      abort("Both `x` and `y` are required.")
    }
    x <- as_data_frame_strict(x)
    outcome_name <- deparse(substitute(y))
  }

  inferred_task <- infer_task(y, task)
  outcome <- prepare_outcome(y, inferred_task)
  loss_name <- normalize_loss(loss, outcome$task, outcome$outcome_type)
  blueprint <- train_blueprint(x)
  processed <- apply_blueprint(blueprint, x)

  resolved_device <- if (identical(device, "auto")) {
    if (isTRUE(torch::cuda_is_available())) "cuda" else "cpu"
  } else {
    device
  }

  trained <- fit_network(
    x = processed$matrix,
    y = outcome$y_train,
    task = outcome$task,
    outcome_type = outcome$outcome_type,
    hidden_units = hidden_units,
    activation = activation,
    dropout = dropout,
    batch_norm = batch_norm,
    residual = residual,
    gated = gated,
    input_projection = input_projection,
    output_dim = outcome$output_dim,
    epochs = epochs,
    batch_size = batch_size,
    lr = lr,
    optimizer = optimizer,
    lr_schedule = lr_schedule,
    weight_decay = weight_decay,
    validation = validation,
    early_stopping = early_stopping,
    patience = patience,
    min_delta = min_delta,
    min_epochs = min_epochs,
    seed = seed,
    verbose = verbose,
    log_every = log_every,
    device = resolved_device,
    loss_name = loss_name,
    label_smoothing = label_smoothing,
    focal_gamma = focal_gamma
  )

  fitted_values <- if (identical(outcome$task, "regression")) {
    predict_from_matrix(trained$model, processed$matrix, outcome, resolved_device, type = "response")
  } else {
    predict_from_matrix(trained$model, processed$matrix, outcome, resolved_device, type = "class")
  }

  fit <- list(
    call = match.call(),
    task = outcome$task,
    outcome_name = outcome_name,
    outcome_type = outcome$outcome_type,
    levels = outcome$levels,
    feature_names = blueprint$feature_names,
    blueprint = blueprint,
    preprocessor = blueprint,
    network_spec = list(
      hidden_units = hidden_units,
      activation = activation,
      dropout = dropout,
      batch_norm = batch_norm,
      residual = residual,
      gated = gated,
      input_projection = input_projection,
      lr_schedule = lr_schedule
    ),
    network = trained$model,
    training_history = trained$history,
    best_epoch = trained$best_epoch,
    best_validation_loss = trained$best_loss,
    best_validation_metric = trained$best_metric,
    metrics = metrics %||% default_metric_name(outcome$task, length(outcome$levels)),
    loss_name = loss_name,
    n_obs = nrow(processed$matrix),
    n_features = ncol(processed$matrix),
    fitted = fitted_values,
    seed = seed,
    encoded_feature_names = processed$encoded_names,
    device = resolved_device,
    outcome_scale = list(center = outcome$y_center, scale = outcome$y_scale)
  )
  class(fit) <- "mlp_fit"
  fit
}
