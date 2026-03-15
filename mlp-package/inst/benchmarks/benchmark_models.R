benchmark_mlp_models <- function(seed = 42) {
  stopifnot(
    requireNamespace("glmnet", quietly = TRUE),
    requireNamespace("ranger", quietly = TRUE)
  )

  score_classification <- function(truth, pred) {
    mean(truth == pred)
  }

  score_regression <- function(truth, pred) {
    sqrt(mean((truth - pred)^2))
  }

  run_case <- function(data, formula, task, mlp_args = list(), mlp_grid = NULL, ranger_args = list(), family = NULL) {
    set.seed(seed)
    n <- nrow(data)
    idx <- sample.int(n, floor(0.8 * n))
    train <- data[idx, , drop = FALSE]
    test <- data[-idx, , drop = FALSE]

    x_train <- stats::model.matrix(formula, train)[, -1, drop = FALSE]
    x_test <- stats::model.matrix(formula, test)[, -1, drop = FALSE]
    y_train <- stats::model.response(stats::model.frame(formula, train))
    y_test <- stats::model.response(stats::model.frame(formula, test))

    mlp_time <- system.time({
      if (is.null(mlp_grid)) {
        mlp_fit <- do.call(mlp::mlp, c(list(formula = formula, data = train, task = task, verbose = FALSE, epochs = 50, patience = 5), mlp_args))
      } else {
        tuned <- mlp::tune_mlp(
          formula = formula,
          data = train,
          task = task,
          grid = mlp_grid,
          validation = 0.25,
          patience = 8,
          seed = seed,
          verbose = FALSE,
          refit = TRUE
        )
        mlp_fit <- tuned$best_fit
      }
    })["elapsed"]

    ranger_time <- system.time({
      ranger_fit <- do.call(ranger::ranger, c(list(formula = formula, data = train), ranger_args))
    })["elapsed"]

    glmnet_time <- system.time({
      glmnet_fit <- glmnet::glmnet(x_train, y_train, family = family)
    })["elapsed"]

    if (identical(task, "regression")) {
      mlp_pred <- predict(mlp_fit, test, type = "response")
      ranger_pred <- predict(ranger_fit, test)$predictions
      glmnet_pred <- as.numeric(stats::predict(glmnet_fit, newx = x_test, s = glmnet_fit$lambda[30]))
      data.frame(
        model = c("mlp", "ranger", "glmnet"),
        metric = "rmse",
        value = c(
          score_regression(y_test, mlp_pred),
          score_regression(y_test, ranger_pred),
          score_regression(y_test, glmnet_pred)
        ),
        runtime_sec = c(mlp_time, ranger_time, glmnet_time)
      )
    } else {
      mlp_pred <- predict(mlp_fit, test, type = "class")
      ranger_pred <- predict(ranger_fit, test)$predictions
      if (identical(family, "multinomial")) {
        glmnet_pred <- colnames(stats::predict(glmnet_fit, newx = x_test, type = "response", s = glmnet_fit$lambda[30])[,,1])[max.col(stats::predict(glmnet_fit, newx = x_test, type = "response", s = glmnet_fit$lambda[30])[,,1])]
      } else {
        glm_prob <- as.numeric(stats::predict(glmnet_fit, newx = x_test, type = "response", s = glmnet_fit$lambda[30]))
        glmnet_pred <- ifelse(glm_prob >= 0.5, levels(y_train)[2], levels(y_train)[1])
      }
      data.frame(
        model = c("mlp", "ranger", "glmnet"),
        metric = "accuracy",
        value = c(
          score_classification(y_test, mlp_pred),
          score_classification(y_test, ranger_pred),
          score_classification(y_test, glmnet_pred)
        ),
        runtime_sec = c(mlp_time, ranger_time, glmnet_time)
      )
    }
  }

  binary_data <- mtcars
  binary_data$am <- factor(binary_data$am, labels = c("auto", "manual"))
  multiclass_data <- iris
  regression_data <- mtcars

  out <- rbind(
    cbind(
      dataset = "mtcars_am",
      run_case(
        binary_data,
        am ~ mpg + wt + hp + cyl,
        "classification",
        family = "binomial",
        mlp_grid = list(
          hidden_units = list(c(16), c(32, 16), c(64, 32)),
          activation = c("relu", "tanh"),
          dropout = c(0, 0.1),
          batch_size = c(8, 16),
          lr = c(1e-3, 5e-3, 1e-2),
          epochs = c(80)
        ),
        ranger_args = list(probability = FALSE)
      )
    ),
    cbind(
      dataset = "iris",
      run_case(
        multiclass_data,
        Species ~ .,
        "classification",
        family = "multinomial",
        mlp_grid = list(
          hidden_units = list(c(16), c(32, 16), c(64, 32)),
          activation = c("relu", "tanh"),
          dropout = c(0, 0.05),
          batch_size = c(8, 16),
          lr = c(1e-3, 5e-3),
          epochs = c(80)
        ),
        ranger_args = list(probability = FALSE)
      )
    ),
    cbind(
      dataset = "mtcars_mpg",
      run_case(
        regression_data,
        mpg ~ disp + hp + wt + cyl,
        "regression",
        family = "gaussian",
        mlp_grid = list(
          hidden_units = list(c(16), c(32, 16), c(64, 32), c(128, 64)),
          activation = c("relu", "tanh"),
          dropout = c(0, 0.05),
          batch_size = c(8, 16),
          lr = c(5e-4, 1e-3, 5e-3),
          epochs = c(120),
          weight_decay = c(0, 1e-4)
        )
      )
    )
  )
  rownames(out) <- NULL
  out
}
