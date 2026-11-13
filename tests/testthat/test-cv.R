test_that("cvdensemlp cross-validates a regression fit", {
  set.seed(1)
  x <- data.frame(a = rnorm(60), b = rnorm(60))
  y <- x$a - 0.5 * x$b + rnorm(60, sd = 0.1)
  cv <- cvdensemlp(x, y, folds = 3, epochs = 15, hidden_units = c(8), validation = 0.2)
  expect_s3_class(cv, "densemlp_cv")
  expect_equal(nrow(cv$fold_metrics), 3)
  expect_true(all(c("rmse", "nrmse", "rsq") %in% names(cv$fold_metrics)))
  expect_equal(cv$task, "regression")
})

test_that("cvdensemlp cross-validates a binary classification fit", {
  set.seed(1)
  x <- data.frame(a = rnorm(60), b = rnorm(60))
  y <- factor(ifelse(x$a + rnorm(60, sd = 0.1) > 0, "yes", "no"))
  cv <- cvdensemlp(x, y, folds = 3, epochs = 15, hidden_units = c(8))
  expect_s3_class(cv, "densemlp_cv")
  expect_true(all(c("accuracy", "balanced_accuracy", "macro_auc", "log_loss") %in% names(cv$fold_metrics)))
})

test_that("cvdensemlp rejects too few folds", {
  x <- data.frame(a = rnorm(20))
  y <- rnorm(20)
  expect_error(cvdensemlp(x, y, folds = 1), "at least 2")
})
