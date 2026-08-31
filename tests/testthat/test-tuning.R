test_that("tune_densemlp ranks candidates and refits the best one", {
  set.seed(1)
  x <- data.frame(a = rnorm(80), b = rnorm(80))
  y <- x$a - 0.5 * x$b + rnorm(80, sd = 0.1)
  tuned <- tune_densemlp(
    x, y, repeats = 1, validation = 0.2,
    grid = list(hidden_units = list(c(8), c(16, 8)), epochs = c(15))
  )
  expect_s3_class(tuned, "densemlp_tuned")
  expect_equal(nrow(tuned$results), 2)
  expect_true(all(diff(tuned$results$score) >= 0))
  expect_s3_class(tuned$best_fit, "densemlp")
})

test_that("tune_densemlp rejects unsupported grid names", {
  x <- data.frame(a = rnorm(20))
  y <- rnorm(20)
  expect_error(tune_densemlp(x, y, grid = list(activation = "relu")), "Unsupported")
})
