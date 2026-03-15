test_that("tune_mlp returns ranked results and a fitted model", {
  tuned <- tune_mlp(
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
  expect_s3_class(tuned$best_fit, "mlp_fit")
  expect_gte(nrow(tuned$results), 2)
})
