test_that("batch_norm = FALSE fits and predicts", {
  set.seed(1)
  x <- data.frame(a = rnorm(80), b = rnorm(80), c = rnorm(80))
  y <- x$a - 0.5 * x$b + rnorm(80, sd = 0.1)
  fit <- densemlp(x, y, epochs = 40, hidden_units = c(12), batch_norm = FALSE, seed = 1)
  expect_false(isTRUE(fit$batch_norm))
  pred <- predict(fit, x)
  expect_length(pred, nrow(x))
  expect_true(all(is.finite(pred)))
})

test_that("input_projection fits, predicts and learns the projection", {
  set.seed(1)
  n <- 200
  x <- data.frame(a = rnorm(n), b = rnorm(n), c = rnorm(n))
  y <- x$a - 0.5 * x$b + 0.3 * x$c + rnorm(n, sd = 0.1)
  fit <- densemlp(x, y, epochs = 150, hidden_units = c(16), input_projection = 3,
                  lr = 5e-3, seed = 1)
  expect_identical(fit$input_projection, 3L)
  expect_equal(dim(fit$proj_weights), c(3L, 3L))
  # projection weights move away from their initialization
  expect_gt(mean(abs(fit$proj_weights - diag(3))), 0.05)
  pred <- predict(fit, x)
  rmse <- sqrt(mean((pred - y)^2))
  expect_lt(rmse, 0.5)
})

test_that("input_projection cannot be combined with interaction", {
  x <- data.frame(a = rnorm(30), b = rnorm(30))
  y <- rnorm(30)
  expect_error(
    densemlp(x, y, input_projection = 2, interaction = TRUE),
    "cannot be used together"
  )
})

test_that("input_projection validates its value", {
  x <- data.frame(a = rnorm(30), b = rnorm(30))
  y <- rnorm(30)
  expect_error(densemlp(x, y, input_projection = 0), "positive integer")
  expect_error(densemlp(x, y, input_projection = 2.5), "positive integer")
})

test_that("perm_importance ranks a relevant feature above noise", {
  set.seed(1)
  n <- 200
  x <- data.frame(sig = rnorm(n), noise1 = rnorm(n), noise2 = rnorm(n))
  y <- 2 * x$sig + rnorm(n, sd = 0.1)
  fit <- densemlp(x, y, epochs = 80, hidden_units = c(16), seed = 1)
  imp <- perm_importance(fit, x, y)
  expect_s3_class(imp, "densemlp_importance")
  expect_identical(imp$data$feature[1], "sig")
  expect_gt(imp$data$importance[1], max(imp$data$importance[-1]))
})

test_that("perm_importance works for classification", {
  set.seed(1)
  n <- 200
  x <- data.frame(sig = rnorm(n), noise = rnorm(n))
  y <- factor(ifelse(x$sig + rnorm(n, sd = 0.3) > 0, "yes", "no"))
  fit <- densemlp(x, y, epochs = 80, hidden_units = c(16), seed = 1)
  imp <- perm_importance(fit, x, y)
  expect_identical(imp$data$feature[1], "sig")
})

test_that("perm_importance rejects ensembles", {
  set.seed(1)
  x <- data.frame(a = rnorm(60), b = rnorm(60))
  y <- x$a + rnorm(60, sd = 0.1)
  fit <- densemlp(x, y, epochs = 20, hidden_units = c(8), ensemble = 2, seed = 1)
  expect_error(perm_importance(fit, x, y), "does not support ensembles")
})

test_that("plot_history returns the fit invisibly", {
  set.seed(1)
  x <- data.frame(a = rnorm(60), b = rnorm(60))
  y <- x$a + rnorm(60, sd = 0.1)
  fit <- densemlp(x, y, epochs = 20, hidden_units = c(8), seed = 1)
  tmp <- tempfile(fileext = ".pdf")
  pdf(tmp)
  out <- plot_history(fit)
  dev.off()
  unlink(tmp)
  expect_identical(out, fit)
})

test_that("batch_norm and input_projection round-trip through an ensemble", {
  set.seed(1)
  n <- 120
  x <- data.frame(a = rnorm(n), b = rnorm(n), c = rnorm(n))
  y <- x$a - 0.5 * x$b + rnorm(n, sd = 0.1)
  fit <- densemlp(x, y, epochs = 30, hidden_units = c(10), batch_norm = FALSE,
                  input_projection = 2, ensemble = 2, seed = 1)
  expect_identical(fit$input_projection, 2L)
  expect_false(isTRUE(fit$batch_norm))
  pred <- predict(fit, x)
  expect_length(pred, nrow(x))
  expect_true(all(is.finite(pred)))
})
