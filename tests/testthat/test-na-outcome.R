test_that("NA in the outcome errors early with a clear message", {
  set.seed(1)
  x <- matrix(rnorm(200), 100)
  y <- x[, 1] + rnorm(100)
  y[3] <- NA
  expect_error(densemlp(x, y, epochs = 2), "`y` contains 1 missing value")
  yb <- factor(ifelse(x[, 1] > 0, "a", "b"))
  yb[c(2, 5)] <- NA
  expect_error(densemlp(x, yb, epochs = 2), "`y` contains 2 missing value")
  d <- data.frame(y = y, x1 = x[, 1], x2 = x[, 2])
  expect_error(densemlp(y ~ ., data = d, epochs = 2), "missing value")
  skip_if_not_installed("survival")
  tt <- rexp(100); ev <- rbinom(100, 1, 0.7); ev[7] <- NA
  expect_error(densemlp(x, survival::Surv(tt, ev), epochs = 2), "missing value")
})
