test_that("binary classification works end to end", {
  data <- mtcars
  data$am <- factor(data$am, labels = c("auto", "manual"))
  fit <- mlp(am ~ mpg + wt + hp + cyl, data = data, epochs = 10, patience = 3, verbose = FALSE, seed = 1)
  pred <- predict(fit, data, type = "class")
  prob <- predict(fit, data, type = "prob")
  expect_s3_class(fit, "mlp_fit")
  expect_true(is.factor(pred))
  expect_equal(ncol(prob), 2)
  expect_equal(levels(pred), levels(data$am))
})

test_that("verbose training prints epoch progress to the console", {
  output <- capture.output(
    mlp(Species ~ ., data = iris, epochs = 2, patience = 1, verbose = TRUE, seed = 1)
  )

  expect_true(any(grepl("^Training 2 epochs$", output)))
  expect_true(any(grepl("^Epoch 1/2 -", output)))
})

test_that("multiclass classification preserves levels", {
  fit <- mlp(Species ~ ., data = iris, epochs = 10, patience = 3, verbose = FALSE, seed = 2)
  pred <- predict(fit, iris, type = "class")
  expect_true(is.factor(pred))
  expect_equal(levels(pred), levels(iris$Species))
})

test_that("regression returns numeric predictions", {
  fit <- mlp(mpg ~ disp + hp + wt, data = mtcars, task = "regression", epochs = 10, patience = 3, verbose = FALSE, seed = 3)
  pred <- predict(fit, mtcars, type = "response")
  metrics <- mlp_metrics(mtcars$mpg, pred, task = "regression")
  expect_true(is.numeric(pred))
  expect_true(all(is.finite(unlist(metrics))))
})

test_that("preprocessing handles unseen categories", {
  train <- data.frame(
    y = factor(c("a", "b", "a")),
    x1 = c(1, NA, 3),
    x2 = factor(c("u", "v", NA))
  )
  fit <- mlp(y ~ ., data = train, epochs = 5, patience = 2, verbose = FALSE, seed = 4)
  new_data <- data.frame(x1 = c(2, NA), x2 = c("new", NA))
  pred <- predict(fit, new_data, type = "class")
  expect_length(pred, 2)
})

test_that("outcomes must be complete", {
  data <- iris
  data$Species[1] <- NA
  expect_error(
    mlp(Species ~ ., data = data, epochs = 2, verbose = FALSE),
    "Outcome values must not be missing"
  )
})

test_that("unused outcome levels are dropped before fitting", {
  data <- iris[iris$Species != "virginica", ]
  data$Species <- factor(data$Species, levels = levels(iris$Species))
  fit <- mlp(Species ~ ., data = data, epochs = 3, patience = 1, verbose = FALSE, seed = 7)
  pred <- predict(fit, data[1:3, ], type = "class")
  prob <- predict(fit, data[1:3, ], type = "prob")
  expect_equal(levels(pred), c("setosa", "versicolor"))
  expect_equal(colnames(prob), c("setosa", "versicolor"))
})

test_that("invalid training controls fail before fitting", {
  expect_error(
    mlp(Species ~ ., data = iris, epochs = 0, verbose = FALSE),
    "`epochs` must be a positive integer",
    fixed = TRUE
  )
  expect_error(
    mlp(Species ~ ., data = iris, validation = 1, verbose = FALSE),
    "`validation` must be in (0, 1).",
    fixed = TRUE
  )
  expect_error(
    mlp(Species ~ ., data = iris, label_smoothing = 1, verbose = FALSE),
    "`label_smoothing` must be in [0, 1).",
    fixed = TRUE
  )
})

test_that("permutation importance returns expected structure", {
  fit <- mlp(Species ~ ., data = iris, epochs = 5, patience = 2, verbose = FALSE, seed = 5)
  imp <- perm_importance(fit, iris[, -5], iris$Species)
  expect_s3_class(imp, "mlp_importance")
  expect_true(all(c("feature", "importance") %in% names(imp$data)))
})

test_that("exported workflow used in getting started works", {
  fit <- mlp(
    Species ~ .,
    data = iris,
    epochs = 5,
    patience = 2,
    verbose = FALSE,
    seed = 6
  )

  pred_class <- predict(fit, iris[1:5, ], type = "class")
  pred_prob <- predict(fit, iris[1:5, ], type = "prob")
  metrics <- mlp_metrics(iris$Species, predict(fit, iris, type = "class"), task = "classification")
  history_plot <- plot_history(fit)
  auto_plot <- ggplot2::autoplot(fit)
  imp <- perm_importance(fit, iris[, -5], iris$Species)

  expect_true(is.factor(pred_class))
  expect_equal(nrow(pred_prob), 5)
  expect_true("accuracy" %in% names(metrics))
  expect_s3_class(history_plot, "ggplot")
  expect_s3_class(auto_plot, "ggplot")
  expect_s3_class(imp, "mlp_importance")
})
