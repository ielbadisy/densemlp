test_that("densemlp fits regression and predicts", {
  set.seed(1)
  n <- 200
  x <- data.frame(a = rnorm(n), b = rnorm(n))
  y <- 2 * x$a - x$b + rnorm(n, sd = 0.1)
  fit <- densemlp(x, y, epochs = 60, hidden_units = c(16), lr = 0.01)
  pred <- predict(fit, x)
  expect_type(pred, "double")
  expect_gt(cor(pred, y), 0.8)
})

test_that("densemlp fits binary classification and predicts", {
  set.seed(2)
  n <- 300
  x <- data.frame(a = rnorm(n), b = rnorm(n))
  p <- plogis(1.5 * x$a - x$b)
  y <- factor(ifelse(runif(n) < p, "yes", "no"))
  fit <- densemlp(x, y, epochs = 60, hidden_units = c(16), lr = 0.01)
  cls <- predict(fit, x, type = "class")
  prob <- predict(fit, x, type = "prob")
  expect_s3_class(cls, "factor")
  expect_equal(ncol(prob), 2)
  expect_gt(mean(cls == y), 0.6)
})

test_that("densemlp fits a survival model and improves the Cox partial likelihood", {
  testthat::skip_if_not_installed("survival")
  set.seed(11)
  n <- 300
  x <- data.frame(a = rnorm(n), b = rnorm(n))
  risk <- 0.8 * x$a - 0.6 * x$b
  time <- stats::rexp(n, rate = exp(risk) * 0.1)
  cens <- stats::rexp(n, rate = 0.03)
  obs_time <- pmin(time, cens)
  event <- as.integer(time <= cens)
  y <- survival::Surv(obs_time, event)

  fit <- densemlp(x, y, task = "survival", hidden_units = c(8), epochs = 60,
             patience = 15, validation = 0.2, seed = 1)
  expect_s3_class(fit, "densemlp")
  expect_true(utils::tail(fit$train_history, 1) < fit$train_history[1])

  risk_score <- predict(fit, x, type = "response")
  metrics <- densemlp_metrics(y, risk_score, task = "survival")
  expect_true(metrics$concordance > 0.55)
})

test_that("densemlp survival accepts a plain two-column (time, event) matrix and rejects bad predict types", {
  set.seed(12)
  x <- data.frame(a = rnorm(150))
  y <- cbind(time = stats::rexp(150, 0.2), event = rbinom(150, 1, 0.7))
  fit <- densemlp(x, y, task = "survival", hidden_units = c(4), epochs = 10, seed = 1)
  pred <- predict(fit, x, type = "response")
  expect_true(is.numeric(pred))
  expect_error(predict(fit, x, type = "class"), "must be \"response\"")
})

test_that("densemlp fits a survival model with the Brier loss and improves the IBS", {
  testthat::skip_if_not_installed("survival")
  set.seed(13)
  n <- 300
  x <- data.frame(a = rnorm(n), b = rnorm(n))
  risk <- 0.8 * x$a - 0.6 * x$b
  time <- stats::rexp(n, rate = exp(risk) * 0.1)
  cens <- stats::rexp(n, rate = 0.03)
  obs_time <- pmin(time, cens)
  event <- as.integer(time <= cens)
  y <- survival::Surv(obs_time, event)

  fit <- densemlp(x, y, task = "survival", loss = "brier", n_bins = 8, hidden_units = c(8),
             epochs = 60, patience = 15, validation = 0.2, seed = 1)
  expect_equal(fit$loss, "brier")
  expect_true(utils::tail(fit$train_history, 1) < fit$train_history[1])

  surv <- predict(fit, x, type = "survival")
  expect_equal(ncol(surv), length(fit$survival_breaks))
  expect_true(all(diff(t(surv)) <= 1e-8)) # non-increasing survival curves

  risk_score <- predict(fit, x, type = "response")
  metrics <- densemlp_metrics(y, risk_score, task = "survival")
  expect_true(metrics$concordance > 0.55)

  ibs <- densemlp_integrated_brier_score(fit, x, y)
  expect_true(is.finite(ibs) && ibs >= 0 && ibs <= 1)
})

test_that("densemlp supports the formula interface, including Surv() responses", {
  testthat::skip_if_not_installed("survival")
  set.seed(14)
  n <- 200
  df <- data.frame(a = rnorm(n), b = rnorm(n))
  df$y <- 2 * df$a - df$b + rnorm(n, sd = 0.1)
  fit <- densemlp(formula = y ~ a + b, data = df, epochs = 60, hidden_units = c(8), lr = 0.01)
  expect_equal(fit$task, "regression")
  pred <- predict(fit, df)
  expect_gt(cor(pred, df$y), 0.7)

  risk <- 0.8 * df$a - 0.6 * df$b
  time <- stats::rexp(n, rate = exp(risk) * 0.1)
  cens <- stats::rexp(n, rate = 0.03)
  df$time <- pmin(time, cens)
  df$event <- as.integer(time <= cens)
  fit_surv <- densemlp(formula = survival::Surv(time, event) ~ a + b, data = df, epochs = 40, hidden_units = c(8))
  expect_equal(fit_surv$task, "survival")

  expect_error(densemlp(formula = y ~ a + b, data = df, x = df[, c("a", "b")], y = df$y), "not both")
  expect_error(densemlp(), "Supply either")
})

test_that("densemlp fits multiclass classification and predicts", {
  set.seed(3)
  n <- 300
  x <- data.frame(a = rnorm(n), b = rnorm(n))
  y <- factor(ifelse(x$a > 0.5, "hi", ifelse(x$a < -0.5, "lo", "mid")))
  fit <- densemlp(x, y, epochs = 80, hidden_units = c(16, 8), lr = 0.01)
  cls <- predict(fit, x, type = "class")
  expect_s3_class(cls, "factor")
  expect_equal(nlevels(cls), 3)
})

test_that("densemlp handles categorical predictors", {
  set.seed(4)
  n <- 150
  x <- data.frame(a = rnorm(n), g = factor(sample(c("x", "y", "z"), n, replace = TRUE)))
  y <- x$a + ifelse(x$g == "x", 1, 0) + rnorm(n, sd = 0.1)
  fit <- densemlp(x, y, epochs = 40, hidden_units = c(8))
  pred <- predict(fit, x)
  expect_length(pred, n)
})

test_that("densemlp early stopping triggers and stays within epochs", {
  set.seed(5)
  n <- 300
  x <- data.frame(a = rnorm(n), b = rnorm(n))
  y <- 2 * x$a - x$b + rnorm(n, sd = 0.1)
  fit <- densemlp(x, y, epochs = 100, hidden_units = c(16), lr = 0.01)
  expect_lte(fit$epochs_run, 100)
  expect_lte(fit$best_epoch, fit$epochs_run)
  expect_length(fit$train_history, fit$epochs_run)
  expect_length(fit$valid_history, fit$epochs_run)
})

test_that("densemlp validation = 0 disables early stopping and runs full epochs", {
  set.seed(6)
  n <- 150
  x <- data.frame(a = rnorm(n), b = rnorm(n))
  y <- 2 * x$a - x$b + rnorm(n, sd = 0.1)
  fit <- densemlp(x, y, epochs = 20, hidden_units = c(8), validation = 0, early_stopping = FALSE)
  expect_equal(fit$epochs_run, 20)
  expect_true(all(is.na(fit$valid_history)))
})

test_that("residual blocks train through identity and projected skips", {
  set.seed(7)
  n <- 240
  x <- data.frame(a = rnorm(n), b = rnorm(n), c = rnorm(n))
  y <- 1.5 * x$a - x$b + 0.5 * x$a * x$c + rnorm(n, sd = 0.1)

  identity_fit <- densemlp(
    x, y, hidden_units = c(12, 12), epochs = 30, lr = 0.01,
    residual = TRUE, validation = 0
  )
  projected_fit <- densemlp(
    x, y, hidden_units = c(16, 8), epochs = 30, lr = 0.01,
    residual = TRUE, validation = 0
  )

  identity_pred <- predict(identity_fit, x)
  projected_pred <- predict(projected_fit, x)
  expect_true(all(is.finite(identity_pred)))
  expect_true(all(is.finite(projected_pred)))
  expect_gt(cor(identity_pred, y), 0.8)
  expect_gt(cor(projected_pred, y), 0.8)
  expect_equal(dim(identity_fit$residual_weights[[2]]), c(0L, 0L))
  expect_equal(dim(projected_fit$residual_weights[[1]]), c(3L, 16L))
  expect_equal(dim(projected_fit$residual_weights[[2]]), c(16L, 8L))
})

test_that("learning-rate schedules are recorded correctly", {
  set.seed(8)
  x <- matrix(rnorm(240), 80, 3)
  y <- x[, 1] - x[, 2]

  none <- densemlp(x, y, hidden_units = 8, epochs = 12, validation = 0,
                  lr = 0.01, lr_schedule = "none")
  cosine <- densemlp(x, y, hidden_units = 8, epochs = 12, validation = 0,
                    lr = 0.01, lr_schedule = "cosine")
  step <- densemlp(x, y, hidden_units = 8, epochs = 12, validation = 0,
                  lr = 0.01, lr_schedule = "step")

  expect_equal(none$lr_history, rep(0.01, 12))
  expect_equal(cosine$lr_history[1], 0.01)
  expect_true(all(diff(cosine$lr_history) < 0))
  expect_equal(step$lr_history, c(rep(0.01, 5), rep(0.005, 5), rep(0.0025, 2)))
})

test_that("new training arguments are validated", {
  x <- matrix(1:20, 10, 2)
  y <- seq_len(10)
  expect_error(densemlp(x, y, hidden_units = 0), "positive integers", fixed = TRUE)
  expect_error(densemlp(x, y, validation = 1), "[0, 1)", fixed = TRUE)
  expect_error(densemlp(x, y, lr_schedule = "linear"))
  expect_error(densemlp(x, y, dropout = 1), "[0, 1)", fixed = TRUE)
  expect_error(densemlp(x, y, ema_decay = 1), "[0, 1)", fixed = TRUE)
  expect_error(densemlp(x, y, ensemble = 0), "positive integer", fixed = TRUE)
})

test_that("gating, dropout, interactions, and EMA train natively", {
  set.seed(9)
  n <- 240
  x <- matrix(rnorm(n * 4), n, 4)
  y <- x[, 1] * x[, 2] + x[, 3] - 0.5 * x[, 4] + rnorm(n, sd = 0.1)
  fit <- densemlp(
    x, y, hidden_units = c(12, 8), epochs = 35, lr = 0.01,
    gated = TRUE, dropout = c(0.05, 0.1), interaction = TRUE,
    ema_decay = 0.9, validation = 0
  )
  pred <- predict(fit, x)

  expect_true(all(is.finite(pred)))
  expect_gt(cor(pred, y), 0.7)
  expect_equal(dim(fit$gate_weights[[1]]), c(12L, 12L))
  expect_length(fit$interaction_weights, ncol(x))
  expect_equal(fit$dropout, c(0.05, 0.1))
  expect_equal(fit$ema_decay, 0.9)
})

test_that("ensemble is fitted internally and averages predictions", {
  set.seed(10)
  n <- 180
  x <- data.frame(a = rnorm(n), b = rnorm(n), group = sample(c("a", "b", "c"), n, TRUE))
  y <- factor(ifelse(x$a + 0.5 * x$b > 0, "yes", "no"))
  fit <- densemlp(
    x, y, hidden_units = 8, epochs = 25, lr = 0.01,
    ensemble = 3, ensemble_bootstrap = TRUE
  )
  response <- predict(fit, x, type = "response")
  member_response <- lapply(fit$members, predict, newdata = x, type = "response")

  expect_s3_class(fit, "densemlp")
  expect_length(fit$members, 3)
  expect_equal(fit$ensemble, 3L)
  expect_equal(response, Reduce(`+`, member_response) / 3)
  expect_equal(dim(predict(fit, x, type = "prob")), c(n, 2L))
  expect_s3_class(predict(fit, x, type = "class"), "factor")
})

test_that("ensemble aggregation supports regression and multiclass", {
  set.seed(11)
  x <- matrix(rnorm(360), 120, 3)
  y_reg <- x[, 1] - 0.5 * x[, 2] + rnorm(120, sd = 0.1)
  reg_fit <- densemlp(x, y_reg, hidden_units = 8, epochs = 20, lr = 0.01,
                     ensemble = 2, ensemble_bootstrap = FALSE)
  reg_members <- lapply(reg_fit$members, predict, newdata = x)
  expect_equal(predict(reg_fit, x), Reduce(`+`, reg_members) / 2)

  y_multi <- factor(ifelse(x[, 1] > 0.5, "high", ifelse(x[, 1] < -0.5, "low", "mid")))
  multi_fit <- densemlp(x, y_multi, hidden_units = 8, epochs = 20, lr = 0.01,
                       ensemble = 2, ensemble_bootstrap = FALSE)
  multi_prob <- predict(multi_fit, x, type = "prob")
  member_prob <- lapply(multi_fit$members, predict, newdata = x, type = "prob")
  expect_equal(multi_prob, Reduce(`+`, member_prob) / 2)
  expect_equal(rowSums(multi_prob), rep(1, nrow(x)), tolerance = 1e-8)
  expect_s3_class(predict(multi_fit, x, type = "class"), "factor")
})
