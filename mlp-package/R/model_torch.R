#' @keywords internal
activation_module <- function(activation) {
  activation <- match.arg(activation, c("relu", "tanh", "gelu"))
  switch(
    activation,
    relu = torch::nn_relu(),
    tanh = torch::nn_tanh(),
    gelu = torch::nn_gelu()
  )
}

#' @keywords internal
initialize_linear <- function(layer, activation = "relu") {
  if (identical(activation, "tanh")) {
    torch::nn_init_xavier_uniform_(layer$weight)
  } else {
    torch::nn_init_kaiming_uniform_(layer$weight, nonlinearity = "relu")
  }
  if (!is.null(layer$bias)) {
    torch::nn_init_zeros_(layer$bias)
  }
  invisible(layer)
}

#' @keywords internal
make_dropout_module <- function(dropout_rate) {
  if (dropout_rate > 0) {
    torch::nn_dropout(p = dropout_rate)
  } else {
    torch::nn_identity()
  }
}

#' @keywords internal
mlp_block <- torch::nn_module(
  "mlp_block",
  initialize = function(input_dim, output_dim, activation, dropout_rate, batch_norm,
                        residual = FALSE, gated = FALSE) {
    self$linear <- torch::nn_linear(input_dim, output_dim)
    initialize_linear(self$linear, activation = activation)
    self$batch_norm <- if (isTRUE(batch_norm)) {
      torch::nn_batch_norm1d(output_dim)
    } else {
      torch::nn_identity()
    }
    self$activation <- activation_module(activation)
    self$dropout <- make_dropout_module(dropout_rate)
    self$gated <- isTRUE(gated)
    self$residual <- isTRUE(residual)
    self$gate <- if (self$gated) {
      gate_layer <- torch::nn_linear(output_dim, output_dim)
      initialize_linear(gate_layer, activation = activation)
      gate_layer
    } else {
      NULL
    }
    self$residual_proj <- if (self$residual && input_dim != output_dim) {
      proj <- torch::nn_linear(input_dim, output_dim)
      initialize_linear(proj, activation = activation)
      proj
    } else {
      NULL
    }
  },
  forward = function(x) {
    out <- self$linear(x)
    out <- self$batch_norm(out)
    out <- self$activation(out)
    if (self$gated) {
      out <- out * torch::torch_sigmoid(self$gate(out))
    }
    out <- self$dropout(out)
    if (self$residual) {
      skip <- if (is.null(self$residual_proj)) x else self$residual_proj(x)
      out <- out + skip
    }
    out
  }
)

#' @keywords internal
mlp_module <- torch::nn_module(
  "mlp_module",
  initialize = function(input_dim, hidden_units, output_dim, activation, dropout,
                        batch_norm, residual = FALSE, gated = FALSE,
                        input_projection = NULL) {
    dropout <- normalize_dropout(dropout, hidden_units)
    self$input_projection <- if (!is.null(input_projection) && input_projection > 0L) {
      proj <- torch::nn_linear(input_dim, as.integer(input_projection))
      initialize_linear(proj, activation = activation)
      proj
    } else {
      NULL
    }

    current_dim <- if (is.null(self$input_projection)) input_dim else as.integer(input_projection)
    self$blocks <- torch::nn_module_list()
    for (i in seq_along(hidden_units)) {
      block <- mlp_block(
        input_dim = current_dim,
        output_dim = hidden_units[[i]],
        activation = activation,
        dropout_rate = dropout[[i]],
        batch_norm = batch_norm,
        residual = residual,
        gated = gated
      )
      self$blocks$append(block)
      current_dim <- hidden_units[[i]]
    }

    self$output <- torch::nn_linear(current_dim, output_dim)
    initialize_linear(self$output, activation = activation)
  },
  forward = function(x) {
    if (!is.null(self$input_projection)) {
      x <- self$input_projection(x)
    }
    for (block in self$blocks) {
      x <- block(x)
    }
    self$output(x)
  }
)

#' @keywords internal
build_optimizer <- function(optimizer, parameters, lr, weight_decay) {
  optimizer <- match.arg(optimizer, c("adam", "sgd"))
  if (identical(optimizer, "adam")) {
    torch::optim_adam(parameters, lr = lr, weight_decay = weight_decay)
  } else {
    torch::optim_sgd(parameters, lr = lr, weight_decay = weight_decay, momentum = 0.9)
  }
}

#' @keywords internal
build_scheduler <- function(schedule, optimizer, epochs) {
  schedule <- match.arg(schedule, c("none", "cosine", "step"))
  if (identical(schedule, "none")) {
    return(NULL)
  }
  if (identical(schedule, "cosine")) {
    return(torch::lr_cosine_annealing_lr(optimizer, T_max = max(2L, epochs)))
  }
  torch::lr_step_lr(optimizer, step_size = max(5L, floor(epochs / 3)), gamma = 0.5)
}
