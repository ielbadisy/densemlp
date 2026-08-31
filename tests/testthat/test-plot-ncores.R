test_that("densemlp ensemble trains identically with ncores = 1 and ncores > 1", {
  set.seed(1)
  x <- data.frame(a = rnorm(60), b = rnorm(60))
  y <- x$a - 0.5 * x$b + rnorm(60, sd = 0.1)
  fit1 <- densemlp(x, y, epochs = 15, hidden_units = c(8), ensemble = 3, ncores = 1, verbose = FALSE)
  fit2 <- densemlp(x, y, epochs = 15, hidden_units = c(8), ensemble = 3, ncores = 2, verbose = FALSE)
  expect_equal(
    predict(fit1, x, type = "response"),
    predict(fit2, x, type = "response")
  )
})

test_that("plot.densemlp draws without error for a single fit and rejects ensembles", {
  set.seed(1)
  x <- data.frame(a = rnorm(40), b = rnorm(40))
  y <- x$a - 0.5 * x$b + rnorm(40, sd = 0.1)
  fit <- densemlp(x, y, epochs = 15, hidden_units = c(8), verbose = FALSE)
  tmp <- tempfile(fileext = ".png")
  grDevices::png(tmp)
  expect_silent(plot(fit))
  grDevices::dev.off()
  unlink(tmp)

  ens <- densemlp(x, y, epochs = 15, hidden_units = c(8), ensemble = 2, verbose = FALSE)
  expect_error(plot(ens), "ensembles")
})
