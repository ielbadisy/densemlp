test_that("logging helpers format epoch ids and metric names", {
  expect_identical(format_epoch_id(1, 10), "01/10")
  expect_identical(format_epoch_id(1, 100), "001/100")
  expect_identical(format_metric_name("accuracy"), "acc")
  expect_identical(format_metric_name("auc"), "auc")
  expect_identical(format_metric_name("rmse"), "rmse")
})

test_that("logging helpers build clean epoch lines", {
  expect_identical(
    format_epoch_log(1, 10, 1.2860),
    "Epoch 01/10 | train_loss: 1.2860"
  )
  expect_identical(
    format_epoch_log(1, 10, 1.2860, 1.3458),
    "Epoch 01/10 | train_loss: 1.2860 | valid_loss: 1.3458"
  )
  expect_identical(
    format_epoch_log(1, 10, 1.2860, 1.3458, NULL, metric_name = "accuracy"),
    "Epoch 01/10 | train_loss: 1.2860 | valid_loss: 1.3458"
  )
  expect_identical(
    format_epoch_log(1, 10, 1.2860, 1.3458, valid_metric = 0.0333, metric_name = "accuracy"),
    "Epoch 01/10 | train_loss: 1.2860 | valid_loss: 1.3458 | valid_acc: 0.0333"
  )
  expect_identical(
    format_epoch_log(1, 10, 1.2860, 1.3458, valid_metric = 0.8734, metric_name = "auc"),
    "Epoch 01/10 | train_loss: 1.2860 | valid_loss: 1.3458 | valid_auc: 0.8734"
  )
  expect_identical(
    format_epoch_log(1, 10, 1.2860, 1.3458, valid_metric = 1.2450, metric_name = "rmse"),
    "Epoch 01/10 | train_loss: 1.2860 | valid_loss: 1.3458 | valid_rmse: 1.2450"
  )
  expect_false(grepl("valid_metric", format_epoch_log(1, 10, 1.2860, 1.3458, valid_metric = 0.0333, metric_name = "accuracy"), fixed = TRUE))
})

test_that("should_log_epoch respects the logging cadence", {
  expect_true(should_log_epoch(1, 100, 10))
  expect_false(should_log_epoch(2, 100, 10))
  expect_true(should_log_epoch(10, 100, 10))
  expect_true(should_log_epoch(100, 100, 10))
})

test_that("verbose zero stays silent", {
  skip_if_no_torch_backend()
  output <- capture.output(
    densemlp(Species ~ ., data = iris, epochs = 2, patience = 1, verbose = 0, seed = 1)
  )
  expect_false(any(grepl("^(Training dense multilayer perceptron|Epoch )", output)))
})

test_that("verbose one prints a header and no per-epoch learning rate", {
  skip_if_no_torch_backend()
  output <- capture.output(
    densemlp(Species ~ ., data = iris, epochs = 2, patience = 1, verbose = 1, seed = 2)
  )
  expect_true(any(grepl("^Training dense multilayer perceptron$", output)))
  expect_true(any(grepl("^Task: multiclass classification$", output)))
  expect_true(any(grepl("^Learning rate: ", output)))
  epoch_lines <- output[grepl("^Epoch ", output)]
  expect_true(length(epoch_lines) > 0)
  expect_false(any(grepl("lr:", epoch_lines, fixed = TRUE)))
})

test_that("verbose two prints per-epoch learning rate and timing", {
  skip_if_no_torch_backend()
  output <- capture.output(
    densemlp(Species ~ ., data = iris, epochs = 2, patience = 1, verbose = 2, seed = 3)
  )
  epoch_lines <- output[grepl("^Epoch ", output)]
  expect_true(any(grepl("lr:", epoch_lines, fixed = TRUE)))
  expect_true(any(grepl("time:", epoch_lines, fixed = TRUE)))
})

test_that("log_every skips intermediate epochs but keeps the first and last", {
  skip_if_no_torch_backend()
  output <- capture.output(
    densemlp(
      Species ~ .,
      data = iris,
      epochs = 20,
      patience = 20,
      early_stopping = FALSE,
      verbose = 1,
      log_every = 10,
      seed = 4
    )
  )
  epoch_lines <- output[grepl("^Epoch ", output)]
  expect_identical(
    sub(" \\|.*", "", epoch_lines),
    c("Epoch 01/20", "Epoch 10/20", "Epoch 20/20")
  )
})
