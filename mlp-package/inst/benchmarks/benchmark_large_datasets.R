benchmark_large_datasets <- function(seed = 42) {
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

  run_case <- function(data, formula, task, family, mlp_grid, ranger_args = list()) {
    set.seed(seed)
    idx <- sample.int(nrow(data), floor(0.8 * nrow(data)))
    train <- data[idx, , drop = FALSE]
    test <- data[-idx, , drop = FALSE]

    tuned <- mlp::tune_mlp(
      formula = formula,
      data = train,
      task = task,
      grid = mlp_grid,
      validation = 0.25,
      patience = 8,
      seed = seed,
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
      scores <- data.frame(
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
      scores <- data.frame(
        model = c("mlp", "ranger", "glmnet"),
        metric = "accuracy",
        value = c(
          score_classification(y_test, stats::predict(mlp_fit, test, type = "class")),
          score_classification(y_test, stats::predict(ranger_fit, test)$predictions),
          score_classification(y_test, glm_pred)
        )
      )
    }

    list(best = tuned$best_config, scores = scores)
  }

  data("PimaIndiansDiabetes2", package = "mlbench")
  pima <- stats::na.omit(PimaIndiansDiabetes2)

  ames <- AmesHousing::make_ames()

  pima_case <- run_case(
    data = pima,
    formula = diabetes ~ .,
    task = "classification",
    family = "binomial",
    mlp_grid = list(
      hidden_units = list(c(16), c(32, 16), c(64, 32)),
      activation = c("relu", "tanh"),
      lr = c(1e-3, 5e-3),
      batch_size = c(16),
      epochs = c(60)
    ),
    ranger_args = list(probability = FALSE)
  )

  ames_case <- run_case(
    data = ames,
    formula = Sale_Price ~ .,
    task = "regression",
    family = "gaussian",
    mlp_grid = list(
      hidden_units = list(c(32, 16), c(64, 32)),
      activation = c("relu"),
      dropout = c(0, 0.05),
      lr = c(1e-3, 5e-3),
      batch_size = c(32),
      epochs = c(80),
      weight_decay = c(0, 1e-4)
    )
  )

  list(
    pima = pima_case,
    ames = ames_case
  )
}
