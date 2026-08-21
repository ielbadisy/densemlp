test_that("cv_densemlp cross-validates a regression model via formula", {
  skip_if_no_torch_backend()
  set.seed(1)
  n <- 100
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- 2 * df$x1 - df$x2 + rnorm(n, sd = 0.3)

  cv <- cv_densemlp(y ~ x1 + x2, data = df, folds = 3, epochs = 8, seed = 5)

  expect_s3_class(cv, "densemlp_cv")
  expect_identical(cv$task, "regression")
  expect_equal(nrow(cv$fold_metrics), 3)
  expect_true(all(c("rmse", "mae", "rsq") %in% names(cv$fold_metrics)))
  expect_true(all(c("mean", "sd") %in% names(cv$summary)))
})

test_that("cv_densemlp cross-validates a classification model via formula", {
  skip_if_no_torch_backend()
  set.seed(2)
  n <- 100
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- factor(ifelse(df$x1 + df$x2 > 0, "a", "b"))

  cv <- cv_densemlp(y ~ x1 + x2, data = df, folds = 3, epochs = 8, seed = 6)

  expect_s3_class(cv, "densemlp_cv")
  expect_identical(cv$task, "classification")
  expect_true(all(c("accuracy", "log_loss", "brier") %in% names(cv$fold_metrics)))
})

test_that("cv_densemlp supports the x/y interface", {
  skip_if_no_torch_backend()
  set.seed(3)
  n <- 90
  x <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  y <- 1.5 * x$x1 + rnorm(n, sd = 0.2)

  cv <- cv_densemlp(x = x, y = y, folds = 3, epochs = 8, seed = 7)

  expect_s3_class(cv, "densemlp_cv")
  expect_equal(nrow(cv$fold_metrics), 3)
})

test_that("cv_densemlp validates folds and mutually exclusive interfaces", {
  df <- data.frame(x1 = rnorm(10), y = rnorm(10))
  expect_error(cv_densemlp(y ~ x1, data = df, folds = 1), "at least 2")
  expect_error(cv_densemlp(y ~ x1, data = df, folds = 3, x = df["x1"], y = df$y))
  expect_error(cv_densemlp(folds = 3))
})

test_that("print.densemlp_cv prints fold metrics and summary", {
  skip_if_no_torch_backend()
  set.seed(4)
  n <- 60
  df <- data.frame(x1 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)
  cv <- cv_densemlp(y ~ x1, data = df, folds = 3, epochs = 5, seed = 8)

  output <- capture.output(print(cv))
  expect_true(any(grepl("cross-validation", output)))
  expect_true(any(grepl("Per-fold metrics", output)))
  expect_true(any(grepl("Summary", output)))
})
