test_that("group intercepts recover additive cluster effects", {
  set.seed(1)
  n_id <- 40; n_t <- 10
  id <- rep(seq_len(n_id), each = n_t)
  alpha <- rnorm(n_id, sd = 1.5)
  x <- data.frame(x1 = runif(n_id * n_t, -2, 2))
  y <- sin(2 * x$x1) + alpha[id] + rnorm(n_id * n_t, sd = 0.3)
  fit <- densemlp(x, y, groups = list(id = id), hidden_units = c(16, 8),
                  epochs = 150, batch_size = 64, lr = 5e-3, seed = 1)
  eff <- group_effects(fit)$id
  expect_named(group_effects(fit), "id")
  expect_length(eff, n_id)
  expect_gt(stats::cor(eff, alpha), 0.95)
  # with the group levels, predictions beat the population-level ones
  rmse <- function(p) sqrt(mean((y - p)^2))
  expect_lt(rmse(predict(fit, x, groups = list(id = id))), rmse(predict(fit, x)))
  # unseen levels contribute no intercept
  expect_equal(predict(fit, x[1:3, , drop = FALSE], groups = list(id = c(999, 999, 999))),
               predict(fit, x[1:3, , drop = FALSE]))
})

test_that("group_penalty can differ across grouping factors", {
  set.seed(2)
  id <- rep(1:20, each = 8); tm <- rep(1:8, times = 20)
  x <- data.frame(x1 = rnorm(160))
  y <- x$x1 + rnorm(20, sd = 2)[id] + rnorm(8, sd = 2)[tm] + rnorm(160, sd = 0.2)
  fit <- densemlp(x, y, groups = list(id = id, time = tm), group_penalty = c(0, 1e3),
                  hidden_units = 8, epochs = 60, lr = 5e-3, seed = 1)
  eff <- group_effects(fit)
  expect_gt(stats::sd(eff$id), 10 * stats::sd(eff$time))
  expect_error(densemlp(x, y, groups = list(id = id, time = tm), group_penalty = c(0, 1, 2)),
               "one value per grouping factor")
})

test_that("count task with a log offset fits a Poisson rate", {
  set.seed(2)
  n <- 2000
  x <- data.frame(x1 = runif(n, -1, 1))
  exposure <- runif(n, 0.5, 4)
  y <- rpois(n, exposure * exp(0.3 + 0.8 * x$x1))
  fit <- densemlp(x, y, task = "count", offset = log(exposure), hidden_units = 8,
                  epochs = 100, batch_size = 128, lr = 5e-3, seed = 2)
  rate <- predict(fit, x)
  mu <- predict(fit, x, offset = log(exposure))
  expect_equal(mu, rate * exposure, tolerance = 1e-8)
  expect_lt(mean(abs(log(rate) - (0.3 + 0.8 * x$x1))), 0.15)
  expect_error(densemlp(x, -y, task = "count"), "non-negative")
})

test_that("a regression offset is added on the outcome scale", {
  set.seed(3)
  n <- 300
  x <- data.frame(x1 = rnorm(n))
  off <- rnorm(n, sd = 5)
  y <- x$x1 + off + rnorm(n, sd = 0.1)
  fit <- densemlp(x, y, offset = off, hidden_units = 8, epochs = 200, lr = 1e-2, seed = 3)
  p0 <- predict(fit, x)
  p1 <- predict(fit, x, offset = off)
  expect_equal(p1 - p0, off, tolerance = 1e-6)
  expect_lt(sqrt(mean((p1 - y)^2)), 0.5)
})

test_that("binary accepts proportions", {
  set.seed(4)
  n <- 500
  x <- data.frame(x1 = rnorm(n))
  y <- plogis(x$x1)
  fit <- densemlp(x, y, task = "binary", hidden_units = 8, epochs = 100, seed = 4)
  expect_lt(mean(abs(predict(fit, x) - y)), 0.05)
  expect_error(densemlp(x, y * 2, task = "binary"), "proportions")
})

test_that("validation row indices are honoured and losses are exact", {
  set.seed(5)
  n <- 300
  x <- data.frame(a = rnorm(n), b = rnorm(n))
  y <- factor(rbinom(n, 1, plogis(x$a)))
  hold <- 201:300
  fit <- densemlp(x, y, task = "binary", validation = hold, hidden_units = 4,
                  epochs = 30, seed = 5)
  p <- predict(fit, x[hold, ])
  yy <- as.numeric(y[hold]) - 1
  bce <- -mean(yy * log(p) + (1 - yy) * log(1 - p))
  expect_equal(bce, min(fit$valid_history), tolerance = 1e-6)
  expect_error(densemlp(x, y, validation = c(1, 1)), "distinct")
  expect_error(densemlp(x, y, validation = 1:10, ensemble = 2), "ensemble")
})

test_that("multiclass takes a matrix offset and per-class group intercepts", {
  set.seed(6)
  n <- 300
  x <- data.frame(a = rnorm(n))
  g <- sample(letters[1:5], n, replace = TRUE)
  y <- factor(sample(c("u", "v", "w"), n, replace = TRUE))
  fit <- densemlp(x, y, task = "multiclass", groups = g, offset = matrix(0, n, 3),
                  hidden_units = 4, epochs = 5, seed = 6)
  eff <- group_effects(fit)$group
  expect_equal(dim(eff), c(5L, 3L))
  expect_equal(dimnames(eff), list(letters[1:5], c("u", "v", "w")))
  pr <- predict(fit, x, type = "prob", groups = g)
  expect_equal(rowSums(pr), rep(1, n), tolerance = 1e-8)
  expect_error(densemlp(x, y, task = "multiclass", offset = rep(0, n)), "column")
})

test_that("survival rejects offset and groups; ensembles pass them through", {
  set.seed(7)
  n <- 200
  x <- data.frame(a = rnorm(n))
  tt <- cbind(time = rexp(n), event = rbinom(n, 1, 0.7))
  expect_error(densemlp(x, tt, task = "survival", groups = rep(1:2, n / 2)), "survival")
  y <- x$a + rep(c(-1, 1), n / 2)
  ens <- densemlp(x, y, groups = list(id = rep(1:2, n / 2)), ensemble = 2,
                  hidden_units = 4, epochs = 20, seed = 7)
  expect_length(group_effects(ens)$id, 2L)
  expect_length(predict(ens, x, groups = list(id = rep(1:2, n / 2))), n)
})

test_that("empty hidden_units gives a linear model", {
  set.seed(3)
  x <- data.frame(a = rnorm(300), b = rnorm(300))
  y <- 1 + 2 * x$a - x$b + rnorm(300, sd = 0.3)
  fit <- densemlp(x, y, hidden_units = integer(0), epochs = 200, lr = 1e-2)
  expect_gt(stats::cor(predict(fit, x), stats::fitted(stats::lm(y ~ a + b, x))), 0.999)
  expect_output(print(fit), "none \\(linear\\)")
})
