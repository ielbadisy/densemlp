test_that("tune_densemlp returns ranked results and a fitted model", {
  tuned <- tune_densemlp(
    Species ~ .,
    data = iris,
    grid = list(
      hidden_units = list(c(8), c(16, 8)),
      activation = c("relu"),
      dropout = c(0),
      batch_size = c(8),
      lr = c(1e-3),
      epochs = c(10)
    ),
    patience = 3,
    seed = 11
  )

  expect_true(is.data.frame(tuned$results))
  expect_s3_class(tuned$best_fit, "densemlp_fit")
  expect_gte(nrow(tuned$results), 2)
})

test_that("verbose tuning prints candidate progress to the console", {
  output <- capture.output(
    tune_densemlp(
      Species ~ .,
      data = iris,
      grid = list(
        hidden_units = list(c(8)),
        activation = c("relu"),
        dropout = c(0),
        batch_size = c(8),
        lr = c(1e-3),
        epochs = c(2)
      ),
      patience = 1,
      repeats = 1,
      seed = 12,
      verbose = TRUE
    )
  )

  expect_true(any(grepl("^Candidate 1/", output)))
})
