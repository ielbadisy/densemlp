benchmark_large_cv <- function(seed = 42, folds = 5, repeats = 3) {
  stopifnot(
    requireNamespace("AmesHousing", quietly = TRUE),
    requireNamespace("glmnet", quietly = TRUE),
    requireNamespace("mlbench", quietly = TRUE),
    requireNamespace("ranger", quietly = TRUE)
  )

  score_classification <- function(truth, pred) {
    mean(truth == pred)
  }

  score_regression <- function(truth, pred) {
    sqrt(mean((truth - pred)^2))
  }

  make_folds <- function(n, seed, folds, repeats) {
    out <- vector("list", repeats)
    for (r in seq_len(repeats)) {
      set.seed(seed + r - 1L)
      ids <- sample(rep(seq_len(folds), length.out = n))
      out[[r]] <- ids
    }
    out
  }

  evaluate_case <- function(data, formula, task, family, mlp_grid, ranger_args = list()) {
    fold_ids <- make_folds(nrow(data), seed = seed, folds = folds, repeats = repeats)
    results <- vector("list", folds * repeats * 3L)
    idx_out <- 1L

    for (r in seq_len(repeats)) {
      ids <- fold_ids[[r]]
      for (k in seq_len(folds)) {
        train <- data[ids != k, , drop = FALSE]
        test <- data[ids == k, , drop = FALSE]

        tuned <- mlp::tune_mlp(
          formula = formula,
          data = train,
          task = task,
          grid = mlp_grid,
          validation = 0.2,
          patience = 8,
          seed = seed + (r * 100L) + k,
          refit = TRUE
        )
        mlp_fit <- tuned$best_fit

        x_train <- stats::model.matrix(formula, train)[, -1, drop = FALSE]
        x_test <- stats::model.matrix(formula, test)[, -1, drop = FALSE]
        y_train <- stats::model.response(stats::model.frame(formula, train))
        y_test <- stats::model.response(stats::model.frame(formula, test))

        ranger_fit <- do.call(ranger::ranger, c(list(formula = formula, data = train), ranger_args))
        glmnet_fit <- glmnet::glmnet(x_train, y_train, family = family)

        if (identical(task, "regression")) {
          fold_res <- data.frame(
            repeat_id = r,
            fold_id = k,
            model = c("mlp", "ranger", "glmnet"),
            metric = "rmse",
            value = c(
              score_regression(y_test, stats::predict(mlp_fit, test, type = "response")),
              score_regression(y_test, stats::predict(ranger_fit, test)$predictions),
              score_regression(y_test, as.numeric(stats::predict(glmnet_fit, newx = x_test, s = glmnet_fit$lambda[30])))
            )
          )
        } else {
          glm_prob <- as.numeric(stats::predict(glmnet_fit, newx = x_test, type = "response", s = glmnet_fit$lambda[30]))
          glm_pred <- factor(
            ifelse(glm_prob >= 0.5, levels(y_train)[2L], levels(y_train)[1L]),
            levels = levels(y_train)
          )
          fold_res <- data.frame(
            repeat_id = r,
            fold_id = k,
            model = c("mlp", "ranger", "glmnet"),
            metric = "accuracy",
            value = c(
              score_classification(y_test, stats::predict(mlp_fit, test, type = "class")),
              score_classification(y_test, stats::predict(ranger_fit, test)$predictions),
              score_classification(y_test, glm_pred)
            )
          )
        }

        results[[idx_out]] <- fold_res
        idx_out <- idx_out + 1L
      }
    }

    raw <- do.call(rbind, results)
    summary <- stats::aggregate(
      value ~ model + metric,
      data = raw,
      FUN = function(x) c(mean = mean(x), sd = stats::sd(x))
    )
    summary <- do.call(data.frame, summary)
    names(summary) <- c("model", "metric", "mean", "sd")
    summary <- summary[order(summary$mean, decreasing = identical(summary$metric[[1L]], "accuracy")), , drop = FALSE]
    rownames(summary) <- NULL

    list(raw = raw, summary = summary)
  }

  data("PimaIndiansDiabetes2", package = "mlbench")
  pima <- stats::na.omit(PimaIndiansDiabetes2)
  ames <- AmesHousing::make_ames()

  list(
    pima = evaluate_case(
      data = pima,
      formula = diabetes ~ .,
      task = "classification",
      family = "binomial",
      mlp_grid = list(
        hidden_units = list(c(16), c(32, 16)),
        activation = c("relu", "tanh"),
        lr = c(1e-3, 5e-3),
        batch_size = c(16),
        epochs = c(50)
      ),
      ranger_args = list(probability = FALSE)
    ),
    ames = evaluate_case(
      data = ames,
      formula = Sale_Price ~ .,
      task = "regression",
      family = "gaussian",
      mlp_grid = list(
        hidden_units = list(c(32, 16), c(64, 32)),
        activation = c("relu"),
        dropout = c(0, 0.05),
        lr = c(1e-3),
        batch_size = c(32),
        epochs = c(60),
        weight_decay = c(0, 1e-4)
      )
    )
  )
}
