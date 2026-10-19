dup_fit_pair <- function(x, y, w, ...) {
  dup <- rep(seq_len(NROW(y)), w)
  args <- list(hidden_units = c(8, 4), epochs = 15, batch_size = 10000L, validation = 0,
               lr = 5e-3, residual = TRUE, gated = TRUE, seed = 4, ...)
  list(
    weighted = do.call(densemlp, c(list(x = x, y = y, weights = w), args)),
    duplicated = do.call(densemlp, c(list(x = x[dup, , drop = FALSE],
                                          y = densemlp_index_outcome(y, dup)), args))
  )
}

# Weights are rescaled to mean one, so numeric inputs are standardized with
# an n - 1 (not sum(w) - 1) denominator. Factor inputs are one-hot encoded
# without scaling, so with them and an unscaled outcome the weighted engine
# (loss, gradients, weighted batch norm) must match repeated rows exactly.
test_that("integer weights match repeated rows (full batch, batch norm on)", {
  set.seed(11)
  n <- 80
  x <- data.frame(a = factor(sample(letters[1:4], n, TRUE)),
                  b = factor(sample(c("u", "v", "z"), n, TRUE)))
  eta <- as.integer(x$a) / 2 - as.integer(x$b) / 3
  w <- sample(1:3, n, TRUE)
  outcomes <- list(
    binary = list(y = factor(ifelse(eta + rnorm(n) > 0, "y", "n")), type = "prob"),
    multiclass = list(y = factor(sample(c("p", "q", "r"), n, TRUE)), type = "prob"),
    count = list(y = rpois(n, exp(0.3 * eta)), type = "response", task = "count")
  )
  for (o in outcomes) {
    fits <- dup_fit_pair(x, o$y, w, task = if (is.null(o$task)) "auto" else o$task)
    expect_equal(predict(fits$weighted, x, type = o$type),
                 predict(fits$duplicated, x, type = o$type), tolerance = 1e-8)
  }
  skip_if_not_installed("survival")
  te <- survival::Surv(rexp(n, exp(0.5 * eta)), rbinom(n, 1, 0.7))
  fits <- dup_fit_pair(x, te, w, loss = "brier", n_bins = 7L)
  expect_equal(predict(fits$weighted, x, type = "survival"),
               predict(fits$duplicated, x, type = "survival"), tolerance = 1e-8)
})

test_that("a weighted linear model matches lm() and coxph() with weights", {
  set.seed(3)
  n <- 100
  x <- data.frame(a = rnorm(n), b = rnorm(n))
  w <- runif(n, 0, 4)
  y <- x$a - x$b^2 + rnorm(n, sd = 0.3)
  fit <- densemlp(x, y, hidden_units = integer(0), weights = w, epochs = 2000,
                  batch_size = 10000L, validation = 0, lr = 1e-2, batch_norm = FALSE)
  expect_equal(unname(predict(fit, x)),
               unname(stats::fitted(stats::lm(y ~ a + b, x, weights = w))), tolerance = 1e-6)

  skip_if_not_installed("survival")
  te <- survival::Surv(rexp(n, exp(0.5 * x$a)), rbinom(n, 1, 0.7))
  fit <- densemlp(x, te, hidden_units = integer(0), weights = w, epochs = 2000,
                  batch_size = 10000L, validation = 0, lr = 1e-2, batch_norm = FALSE)
  ref <- survival::coxph(te ~ a + b, x, weights = w, ties = "breslow")
  lp <- predict(fit, x)
  lr <- stats::predict(ref, type = "lp")
  expect_equal(unname(lp - mean(lp)), unname(lr - mean(lr)), tolerance = 1e-6)
})

test_that("only relative weights matter; constant weights reproduce the unweighted fit", {
  set.seed(2)
  x <- data.frame(a = rnorm(100), b = rnorm(100))
  y <- x$a + rnorm(100, sd = 0.2)
  p0 <- predict(densemlp(x, y, epochs = 20, seed = 2), x)
  p1 <- predict(densemlp(x, y, epochs = 20, seed = 2, weights = rep(5, 100)), x)
  expect_equal(p0, p1, tolerance = 1e-6)
  w <- runif(100, 0.1, 3)
  expect_equal(predict(densemlp(x, y, epochs = 20, weights = w), x),
               predict(densemlp(x, y, epochs = 20, weights = 10 * w), x), tolerance = 1e-8)
  expect_false(densemlp(x, y, epochs = 2)$has_weights)
})

test_that("the validation loss is the weighted mean over held-out rows", {
  set.seed(5)
  x <- data.frame(a = rnorm(100))
  y <- x$a + rnorm(100)
  w <- runif(100, 0, 3)
  vr <- 1:30
  fit <- densemlp(x, y, validation = vr, weights = w, epochs = 20, hidden_units = 4)
  r2 <- ((predict(fit, x[vr, , drop = FALSE]) - y[vr]) / fit$y_scale)^2
  expect_equal(fit$valid_history[fit$best_epoch], sum(w[vr] * r2) / sum(w[vr]))
})

test_that("zero weights drop rows from the fit", {
  set.seed(6)
  x <- data.frame(a = rnorm(60))
  y <- x$a + rnorm(60)
  w <- c(rep(1, 40), rep(0, 20))
  y_noise <- y
  y_noise[41:60] <- 1e3
  a <- densemlp(x, y, weights = w, hidden_units = integer(0), epochs = 300,
                batch_size = 10000L, validation = 0, lr = 1e-2, batch_norm = FALSE)
  b <- densemlp(x, y_noise, weights = w, hidden_units = integer(0), epochs = 300,
                batch_size = 10000L, validation = 0, lr = 1e-2, batch_norm = FALSE)
  # the outcome scale is computed with weights, so ignored rows leave it unchanged
  expect_equal(predict(a, x), predict(b, x), tolerance = 1e-10)
})

test_that("weights are validated", {
  x <- data.frame(a = rnorm(10))
  y <- rnorm(10)
  expect_error(densemlp(x, y, weights = rep(1, 9)), "one per row")
  expect_error(densemlp(x, y, weights = c(-1, rep(1, 9))), "non-negative")
  expect_error(densemlp(x, y, weights = c(NA, rep(1, 9))), "non-negative")
  expect_error(densemlp(x, y, weights = rep(0, 10)), "positive")
})

test_that("weighted metrics equal metrics on repeated rows", {
  set.seed(8)
  n <- 50
  w <- sample(1:3, n, TRUE)
  dup <- rep(seq_len(n), w)
  truth <- factor(sample(c("n", "y"), n, TRUE))
  prob <- runif(n)
  prob <- cbind(n = 1 - prob, y = prob)
  est <- factor(ifelse(prob[, "y"] > 0.5, "y", "n"), levels = c("n", "y"))
  expect_equal(densemlp_metrics(truth, est, "binary", prob, weights = w),
               densemlp_metrics(truth[dup], est[dup], "binary", prob[dup, ]))
  yr <- rnorm(n)
  er <- yr + rnorm(n)
  mw <- densemlp_metrics(yr, er, "regression", weights = w)
  md <- densemlp_metrics(yr[dup], er[dup], "regression")
  expect_equal(mw[c("rmse", "rsq")], md[c("rmse", "rsq")])
  skip_if_not_installed("survival")
  te <- survival::Surv(rexp(n), rbinom(n, 1, 0.7))
  expect_equal(densemlp_metrics(te, er, "survival", weights = w),
               densemlp_metrics(te[dup], er[dup], "survival"))
})

test_that("cv_densemlp() and tune_densemlp() pass weights through", {
  set.seed(9)
  x <- data.frame(a = rnorm(60), b = rnorm(60))
  y <- x$a + rnorm(60, sd = 0.3)
  w <- runif(60, 0.2, 3)
  cv_w <- cv_densemlp(x, y, folds = 3, epochs = 5, hidden_units = 4, weights = w)
  cv_u <- cv_densemlp(x, y, folds = 3, epochs = 5, hidden_units = 4)
  expect_false(isTRUE(all.equal(cv_w$fold_metrics, cv_u$fold_metrics)))
  expect_error(cv_densemlp(x, y, folds = 3, weights = w[-1]), "one value per row")
  tuned <- tune_densemlp(x, y, repeats = 1, weights = w,
                         grid = list(hidden_units = list(4), epochs = 5))
  expect_true(tuned$best_fit$has_weights)
})

test_that("ensembles carry weights to every member", {
  set.seed(10)
  x <- data.frame(a = rnorm(60))
  y <- x$a + rnorm(60)
  fit <- densemlp(x, y, ensemble = 2, epochs = 5, hidden_units = 4, weights = runif(60))
  expect_true(fit$has_weights)
  expect_true(all(vapply(fit$members, `[[`, logical(1), "has_weights")))
})
