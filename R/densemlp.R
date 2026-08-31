#' Fit a fast dense multilayer perceptron
#'
#' A compact feedforward network for regression and classification on
#' tabular data. Forward propagation, backpropagation, and Adam optimization
#' are all implemented natively in C++ via `RcppArmadillo` -- no `torch` /
#' `libtorch` dependency.
#'
#' @param x Predictor data.frame or matrix (`x`/`y` interface).
#' @param y Outcome (`x`/`y` interface): numeric for regression, a
#'   factor/character (2 levels for binary, 3+ for multiclass) for
#'   classification, or, for `task = "survival"`, a [survival::Surv()]
#'   object or a two-column matrix/data.frame giving `(time, event)`
#'   (`event` coded `1` = event, `0` = censored).
#' @param task `"auto"` infers the task from `y` (a [survival::Surv()]
#'   response, from either interface, is always detected as `"survival"`);
#'   otherwise one of `"regression"`, `"binary"`, `"multiclass"`,
#'   `"survival"`.
#' @param loss For `task = "survival"` only: `"cox"` (batch-wise Breslow-tie
#'   Cox partial likelihood, a single linear risk score) or `"brier"`
#'   (IPCW integrated Brier score over a discrete-time grid of `n_bins`
#'   hazard outputs; see the "Survival" section below). Ignored otherwise.
#' @param n_bins For `task = "survival", loss = "brier"` only: number of
#'   discrete-time bins (quantile cutpoints of the observed follow-up
#'   times).
#' @param hidden_units Integer vector of hidden layer sizes.
#' @param epochs Maximum number of training epochs (an upper bound when
#'   `early_stopping` is used).
#' @param batch_size Mini-batch size.
#' @param lr Adam learning rate.
#' @param residual Logical; add a residual skip to every hidden block. A
#'   learned linear projection is used when the block dimensions differ.
#' @param gated Logical; use a learned sigmoid gate in every hidden block.
#' @param dropout Dropout probability, either one value or one per hidden
#'   layer. Values must be in `[0, 1)`.
#' @param batch_norm Logical; apply batch normalization inside every hidden
#'   block (`Linear -> BatchNorm -> ReLU`). When `FALSE`, hidden blocks are
#'   `Linear -> ReLU` with no normalization and no learned BN affine
#'   parameters.
#' @param input_projection Optional positive integer. When set, a plain
#'   linear layer (no activation, no batch normalization) maps the encoded
#'   predictors to `input_projection` dimensions before the first hidden
#'   block. `NULL` (default) disables it. Cannot be combined with
#'   `interaction`.
#' @param interaction Logical; prepend an efficient learned cross-feature
#'   layer that models explicit second-order interactions in `O(p)` parameters.
#' @param ema_decay Exponential moving-average decay for model parameters.
#'   Set to `0` to disable; values such as `0.99` enable EMA evaluation and
#'   best-epoch restoration.
#' @param ensemble Number of internally fitted members whose predictions are
#'   averaged. `1` fits a single model and preserves the standard behavior.
#' @param ensemble_bootstrap Logical; bootstrap rows independently for each
#'   member when `ensemble > 1`.
#' @param lr_schedule Learning-rate schedule: `"none"`, cosine annealing over
#'   `epochs`, or `"step"` decay by 0.5 every `max(5, floor(epochs / 3))`
#'   epochs.
#' @param validation Validation fraction held out for early stopping.
#'   Set to `0` to disable (trains for the full `epochs`, no BN running-stat
#'   evaluation set).
#' @param early_stopping Logical; stop once validation loss stops improving.
#'   Ignored if `validation = 0`.
#' @param patience Number of non-improving epochs to wait before stopping.
#' @param min_delta Minimum validation loss improvement to reset patience.
#' @param min_epochs Minimum number of epochs before early stopping can
#'   trigger. Defaults to `max(10, floor(epochs * 0.2))`.
#' @param seed Integer seed.
#' @param verbose Logical; print training/validation loss every 10 epochs.
#' @param ncores Number of cores used to fit ensemble members in parallel
#'   when `ensemble > 1` (via `parallel::mclapply` on Unix-alikes, serially
#'   on Windows). Ignored when `ensemble = 1`.
#' @param formula A formula, e.g. `y ~ .` or, for survival,
#'   `survival::Surv(time, status) ~ .`, as an alternative to `x`/`y`. Use
#'   either `formula`/`data` or `x`/`y`, not both. A formula may be given as
#'   the first positional argument -- `densemlp(y ~ ., df)`,
#'   `densemlp(formula = y ~ ., data = df)` and `densemlp(x, y)` all work.
#' @param data A data.frame used with `formula`.
#'
#' @details
#' Hidden layers are `Linear -> BatchNorm -> ReLU -> optional gate -> optional
#' dropout -> optional residual` (batch statistics during training, running
#' mean/var at prediction time, exponential decay 0.9), with He initialization
#' for the linear weights. Set `batch_norm = FALSE` to drop the normalization
#' step, leaving `Linear -> ReLU` hidden blocks. Residual blocks use a learned
#' linear projection when dimensions differ. With `input_projection = k`, a
#' bare linear layer maps the encoded inputs to `k` dimensions before the
#' first hidden block (mutually exclusive with `interaction`). The optional
#' interaction layer is a one-layer cross network initialized as the
#' identity. With `ema_decay > 0`, validation and
#' final predictions use moving-average parameters. With `ensemble > 1`, fully
#' fitted internal members are trained with successive seeds and optionally
#' bootstrapped rows, and their response probabilities/predictions are averaged
#' transparently. The output layer is plain
#' `Linear -> task activation` (no BN): linear (regression, MSE loss),
#' sigmoid (binary, binary cross-entropy), softmax (multiclass, categorical
#' cross-entropy), or a linear risk score (survival, Cox partial
#' likelihood); the first three share the output-gradient simplification
#' `dZ = (yhat - y) / n`, while survival uses its own closed-form Cox
#' gradient (see the "Survival" section below). With `validation > 0` and
#' `early_stopping = TRUE` (both on by default), the parameters (weights,
#' biases, and BN affine/running-stat parameters) from the best validation
#' epoch are restored at the end.
#'
#' @section Survival:
#' `task = "survival"` supports two losses, both trained batch-wise against
#' a `(time, event)` outcome:
#' * `loss = "cox"` (the default) trains a single linear risk score using a
#'   batch-wise Breslow-tie Cox partial log-likelihood -- the same
#'   "risk set = current mini-batch" approximation used by DeepSurv/pycox,
#'   which turns Cox regression into an ordinary per-batch loss compatible
#'   with masked-free Adam training, BN, and early stopping unchanged. For a
#'   batch sorted by descending time, with each observation's risk set
#'   approximated by the sorted prefix of observations with a time at least
#'   as large, the loss is the negative mean, over events, of the risk
#'   score minus the log of the cumulative sum of exponentiated risk scores
#'   over its risk set -- computed via a single cumulative sum with a
#'   closed-form O(n) gradient (no autodiff).
#' * `loss = "brier"` trains a discrete-time hazard head with `n_bins`
#'   outputs (a quantile grid of the observed follow-up times) directly
#'   against the IPCW (Graf et al.) integrated Brier score. Bin `k`'s
#'   sigmoid output is a conditional hazard, and the survival probability
#'   at that bin is the running product of one minus every hazard up to
#'   and including it, which is monotonically non-increasing by
#'   construction. Each bin's Brier term
#'   is inverse-probability-of-censoring weighted using a Kaplan-Meier
#'   estimate of the *censoring* distribution fit once on the training
#'   data; subjects censored before a bin are excluded from that bin's
#'   term, per Graf's correction. The gradient is closed-form (no
#'   autodiff): a bin's survival probability depends on every hazard
#'   logit at or before it, so each bin's Brier-score gradient is
#'   distributed back across all of them.
#'
#' `predict(fit, newdata, type = "response")` returns a linear risk score
#' for either loss (for `"brier"`, the negative log of the predicted
#' survival probability at the final time bin, so higher is still
#' riskier); with
#' `loss = "brier"`, `type = "survival"` additionally returns the full
#' `n_bins`-column survival-probability matrix. [densemlp_metrics()] reports
#' Harrell's concordance index for `task = "survival"` regardless of loss.
#'
#' @return A `densemlp` object.
#' @examples
#' set.seed(1)
#' x <- data.frame(a = rnorm(100), b = rnorm(100))
#' y <- x$a - 0.5 * x$b + rnorm(100, sd = 0.1)
#' fit <- densemlp(x, y, epochs = 50, hidden_units = c(16))
#' predict(fit, x[1:5, ])
#' @export
densemlp <- function(x = NULL, y = NULL,
                     task = c("auto", "regression", "binary", "multiclass", "survival"),
                     loss = c("cox", "brier"), n_bins = 10L,
                     hidden_units = c(32, 16), epochs = 100L, batch_size = 32L,
                     lr = 1e-3, validation = 0.2, early_stopping = TRUE,
                     patience = 10L, min_delta = 0, min_epochs = max(10L, floor(epochs * 0.2)),
                     residual = FALSE, gated = FALSE, dropout = 0,
                     batch_norm = TRUE, input_projection = NULL,
                     interaction = FALSE, ema_decay = 0,
                     ensemble = 1L, ensemble_bootstrap = TRUE,
                     lr_schedule = c("none", "cosine", "step"),
                     seed = 1L, verbose = FALSE, ncores = 1L,
                     formula = NULL, data = NULL) {
  task <- match.arg(task)
  loss <- match.arg(loss)
  lr_schedule <- match.arg(lr_schedule)

  # Allow a formula in the first positional slot: `densemlp(y ~ ., data)`.
  # A formula is never valid predictor data, so this is unambiguous.
  if (inherits(x, "formula") && is.null(formula)) {
    formula <- x
    x <- NULL
    if (is.null(data) && !is.null(y)) {
      data <- y
      y <- NULL
    }
  }

  using_formula <- !is.null(formula) || !is.null(data)
  using_xy <- !is.null(x) || !is.null(y)
  if (using_formula && using_xy) {
    stop("Use either the `formula`/`data` interface or the `x`/`y` interface, not both.", call. = FALSE)
  }
  if (!using_formula && !using_xy) {
    stop("Supply either `formula` and `data`, or `x` and `y`.", call. = FALSE)
  }
  outcome_name <- NULL
  if (using_formula) {
    if (is.null(formula) || is.null(data)) {
      stop("Both `formula` and `data` are required.", call. = FALSE)
    }
    mf <- stats::model.frame(formula, data = data, na.action = stats::na.pass)
    outcome_name <- names(mf)[1L]
    y <- stats::model.response(mf)
    x <- mf[-1L]
  } else {
    if (is.null(x) || is.null(y)) {
      stop("Both `x` and `y` are required.", call. = FALSE)
    }
    outcome_name <- deparse(substitute(y))
  }

  if (identical(task, "auto")) {
    if (inherits(y, "Surv")) {
      task <- "survival"
    } else if (is.numeric(y)) {
      task <- "regression"
    } else {
      nlev <- nlevels(as.factor(y))
      task <- if (nlev <= 2L) "binary" else "multiclass"
    }
  }
  if (!is.numeric(hidden_units) || length(hidden_units) < 1L ||
      any(!is.finite(hidden_units)) || any(hidden_units < 1) ||
      any(hidden_units != as.integer(hidden_units))) {
    stop("`hidden_units` must contain positive integers.", call. = FALSE)
  }
  if (length(epochs) != 1L || !is.finite(epochs) || epochs < 1 || epochs != as.integer(epochs)) {
    stop("`epochs` must be a positive integer.", call. = FALSE)
  }
  if (length(batch_size) != 1L || !is.finite(batch_size) || batch_size < 1 ||
      batch_size != as.integer(batch_size)) {
    stop("`batch_size` must be a positive integer.", call. = FALSE)
  }
  if (length(lr) != 1L || !is.finite(lr) || lr <= 0) {
    stop("`lr` must be a positive number.", call. = FALSE)
  }
  if (length(validation) != 1L || !is.finite(validation) || validation < 0 || validation >= 1) {
    stop("`validation` must be in [0, 1).", call. = FALSE)
  }
  if (!is.numeric(dropout) || length(dropout) < 1L || any(!is.finite(dropout)) ||
      any(dropout < 0) || any(dropout >= 1)) {
    stop("`dropout` must contain values in [0, 1).", call. = FALSE)
  }
  if (length(dropout) == 1L) dropout <- rep(dropout, length(hidden_units))
  if (length(dropout) != length(hidden_units)) {
    stop("`dropout` must have length 1 or match `hidden_units`.", call. = FALSE)
  }
  if (length(batch_norm) != 1L || !is.logical(batch_norm) || is.na(batch_norm)) {
    stop("`batch_norm` must be `TRUE` or `FALSE`.", call. = FALSE)
  }
  if (!is.null(input_projection)) {
    if (length(input_projection) != 1L || !is.finite(input_projection) ||
        input_projection < 1 || input_projection != as.integer(input_projection)) {
      stop("`input_projection` must be `NULL` or a positive integer.", call. = FALSE)
    }
    if (isTRUE(interaction)) {
      stop("`input_projection` and `interaction` cannot be used together.", call. = FALSE)
    }
  }
  if (length(ema_decay) != 1L || !is.finite(ema_decay) || ema_decay < 0 || ema_decay >= 1) {
    stop("`ema_decay` must be in [0, 1).", call. = FALSE)
  }
  if (length(ensemble) != 1L || !is.finite(ensemble) || ensemble < 1 ||
      ensemble != as.integer(ensemble)) {
    stop("`ensemble` must be a positive integer.", call. = FALSE)
  }
  if (length(n_bins) != 1L || !is.finite(n_bins) || n_bins < 2 || n_bins != as.integer(n_bins)) {
    stop("`n_bins` must be an integer >= 2.", call. = FALSE)
  }
  if (NROW(x) != NROW(y)) stop("`x` and `y` must have the same number of rows.", call. = FALSE)

  if (ensemble > 1L) {
    x_members <- as.data.frame(x, stringsAsFactors = FALSE)
    x_members[] <- lapply(x_members, function(column) {
      if (is.character(column)) factor(column) else column
    })
    outcome_levels <- if (task %in% c("regression", "survival")) NULL else unique(as.character(y))
    member_index <- vector("list", ensemble)
    for (member in seq_len(ensemble)) {
      member_seed <- as.integer(seed) + member - 1L
      if (isTRUE(ensemble_bootstrap)) {
        set.seed(member_seed)
        index <- sample.int(NROW(x_members), NROW(x_members), replace = TRUE)
        if (!is.null(outcome_levels)) {
          sampled_levels <- unique(as.character(y[index]))
          missing_levels <- setdiff(outcome_levels, sampled_levels)
          for (j in seq_along(missing_levels)) {
            index[j] <- which(as.character(y) == missing_levels[j])[1L]
          }
        }
      } else {
        index <- seq_len(NROW(x_members))
      }
      member_index[[member]] <- index
    }
    fit_member <- function(member) {
      index <- member_index[[member]]
      densemlp(
        x = x_members[index, , drop = FALSE], y = densemlp_index_outcome(y, index), task = task,
        loss = loss, n_bins = n_bins,
        hidden_units = hidden_units, epochs = epochs, batch_size = batch_size,
        lr = lr, validation = validation, early_stopping = early_stopping,
        patience = patience, min_delta = min_delta, min_epochs = min_epochs,
        residual = residual, gated = gated, dropout = dropout,
        batch_norm = batch_norm, input_projection = input_projection,
        interaction = interaction, ema_decay = ema_decay,
        ensemble = 1L, ensemble_bootstrap = FALSE, lr_schedule = lr_schedule,
        seed = as.integer(seed) + member - 1L, verbose = verbose
      )
    }
    members <- densemlp_parallel_lapply(seq_len(ensemble), fit_member, ncores)
    out <- list(
      call = match.call(), members = members, ensemble = as.integer(ensemble),
      ensemble_bootstrap = isTRUE(ensemble_bootstrap), task = task, loss = loss,
      outcome_name = outcome_name,
      task_code = members[[1L]]$task_code, levels = members[[1L]]$levels,
      hidden_units = as.integer(hidden_units), epochs = as.integer(epochs),
      epochs_run = vapply(members, `[[`, integer(1), "epochs_run"),
      best_epoch = vapply(members, `[[`, integer(1), "best_epoch"),
      residual = isTRUE(residual), gated = isTRUE(gated), dropout = dropout,
      batch_norm = isTRUE(batch_norm),
      input_projection = if (is.null(input_projection)) NULL else as.integer(input_projection),
      interaction = isTRUE(interaction), ema_decay = ema_decay,
      lr_schedule = lr_schedule, validation = validation, seed = as.integer(seed)
    )
    class(out) <- "densemlp"
    return(out)
  }

  blueprint <- densemlp_train_blueprint(x)
  xmat <- densemlp_apply_blueprint(blueprint, x)

  task_code <- switch(task, regression = 0L, binary = 1L, multiclass = 2L, survival = 3L)
  levels_y <- NULL
  y_center <- 0
  y_scale <- 1
  output_dim <- 1L

  if (task_code == 0L) {
    y_num <- as.numeric(y)
    y_center <- mean(y_num)
    y_scale <- stats::sd(y_num)
    if (!is.finite(y_scale) || y_scale == 0) y_scale <- 1
    ymat <- matrix((y_num - y_center) / y_scale, ncol = 1)
  } else if (task_code == 1L) {
    y_fac <- factor(y)
    levels_y <- levels(y_fac)
    if (length(levels_y) != 2L) stop("`task = \"binary\"` requires a 2-level outcome.", call. = FALSE)
    ymat <- matrix(as.numeric(y_fac) - 1, ncol = 1)
  } else if (task_code == 2L) {
    y_fac <- factor(y)
    levels_y <- levels(y_fac)
    k <- length(levels_y)
    output_dim <- k
    ymat <- matrix(0, nrow = length(y_fac), ncol = k)
    ymat[cbind(seq_along(y_fac), as.integer(y_fac))] <- 1
  }

  survival_loss_code <- 0L
  breaks <- numeric(0)
  ghat_grid <- numeric(0)

  if (task_code == 3L) {
    time_event <- densemlp_as_time_event_matrix(y)
    if (any(time_event[, 1] <= 0)) stop("Survival times must be strictly positive.", call. = FALSE)
    if (!all(time_event[, 2] %in% c(0, 1))) stop("Survival `event` must be coded 0 (censored) or 1 (event).", call. = FALSE)
    if (sum(time_event[, 2]) < 1) stop("Survival outcome must contain at least one event.", call. = FALSE)

    if (identical(loss, "cox")) {
      ymat <- time_event
      output_dim <- 1L
    } else {
      survival_loss_code <- 1L
      breaks <- densemlp_time_grid(time_event[, 1], as.integer(n_bins))
      output_dim <- length(breaks)
      km <- densemlp_censoring_km(time_event[, 1], time_event[, 2], breaks)
      ghat_grid <- km$grid
      ymat <- cbind(time_event, ghat_at_time = km$at_time)
    }
  }

  # arma::arma_rng::set_seed() wraps R's own RNG stream, so reproducibility across
  # calls requires the R-level generator to be seeded too -- otherwise training
  # draws (init weights, dropout masks, train/validation shuffling) depend on
  # whatever global RNG state happens to precede this call.
  set.seed(as.integer(seed))
  fit <- withCallingHandlers(
    densemlp_train_cpp(
      X = xmat, Y = ymat, hidden_units = as.integer(hidden_units), task = task_code,
      output_dim = as.integer(output_dim),
      epochs = as.integer(epochs), batch_size = as.integer(batch_size), lr = lr,
      validation = validation, early_stopping = isTRUE(early_stopping),
      patience = as.integer(patience), min_delta = min_delta, min_epochs = as.integer(min_epochs),
      residual = isTRUE(residual), gated = isTRUE(gated), dropout = as.numeric(dropout),
      interaction = isTRUE(interaction), batch_norm = isTRUE(batch_norm),
      input_projection = if (is.null(input_projection)) 0L else as.integer(input_projection),
      ema_decay = ema_decay, lr_schedule = lr_schedule,
      seed = as.integer(seed), verbose = isTRUE(verbose),
      survival_loss = survival_loss_code, breaks_in = breaks, ghat_grid_in = ghat_grid
    ),
    warning = function(w) {
      if (grepl("RNG seed has to be set at the R level", conditionMessage(w))) {
        invokeRestart("muffleWarning")
      }
    }
  )

  out <- list(
    call = match.call(),
    outcome_name = outcome_name,
    blueprint = blueprint,
    weights = fit$weights,
    biases = fit$biases,
    gamma = fit$gamma,
    beta = fit$beta,
    running_mean = fit$running_mean,
    running_var = fit$running_var,
    residual_weights = fit$residual_weights,
    residual_biases = fit$residual_biases,
    gate_weights = fit$gate_weights,
    gate_biases = fit$gate_biases,
    interaction_weights = fit$interaction_weights,
    interaction_biases = fit$interaction_biases,
    proj_weights = fit$proj_weights,
    proj_biases = fit$proj_biases,
    train_history = fit$train_history,
    valid_history = fit$valid_history,
    lr_history = fit$lr_history,
    best_epoch = fit$best_epoch,
    epochs_run = fit$epochs_run,
    task = task,
    task_code = task_code,
    loss = loss,
    n_bins = as.integer(n_bins),
    survival_breaks = breaks,
    levels = levels_y,
    y_center = y_center,
    y_scale = y_scale,
    hidden_units = as.integer(hidden_units),
    epochs = as.integer(epochs),
    batch_size = as.integer(batch_size),
    lr = lr,
    validation = validation,
    early_stopping = isTRUE(early_stopping),
    residual = isTRUE(residual),
    gated = isTRUE(gated),
    dropout = dropout,
    batch_norm = isTRUE(batch_norm),
    input_projection = if (is.null(input_projection)) NULL else as.integer(input_projection),
    interaction = isTRUE(interaction),
    ema_decay = ema_decay,
    ensemble = 1L,
    lr_schedule = lr_schedule,
    seed = as.integer(seed)
  )
  class(out) <- "densemlp"
  out
}
