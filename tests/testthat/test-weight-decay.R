test_that("native weight gradient with weight decay matches finite differences", {
  set.seed(1)
  X <- matrix(rnorm(40 * 3), 40)
  lin <- drop(X %*% c(1, -1, 0.5)) + rnorm(40)
  targets <- list(
    `0` = matrix(lin),
    `1` = matrix(as.numeric(lin > 0)),
    `2` = diag(3)[sample(1:3, 40, TRUE), ],
    `4` = matrix(rpois(40, 2))
  )
  for (task in names(targets)) {
    for (wd in c(0, 0.3)) {
      for (h in list(integer(0), c(5L, 4L))) {
        set.seed(2)
        r <- suppressWarnings(densemlp:::densemlp_gradcheck_cpp(
          X, targets[[task]], h, as.integer(task), wd, 1L, 1e-6))
        rel <- max(abs(r$analytic - r$numeric)) / max(abs(r$numeric))
        expect_lt(rel, 1e-6, label = sprintf("task %s, wd %s, %d hidden", task, wd, length(h)))
      }
    }
  }
})

test_that("weight decay shrinks the weights monotonically", {
  set.seed(3)
  n <- 300
  X <- matrix(rnorm(n * 4), n)
  y <- drop(X %*% c(2, -1, 0.5, 0)) + rnorm(n)
  sq <- function(f) sum(vapply(f$weights, function(w) sum(w^2), numeric(1)))
  norms <- vapply(c(0, 0.01, 0.1, 1), function(wd) {
    sq(densemlp(X, y, hidden_units = c(16, 8), epochs = 60, validation = 0,
                batch_norm = FALSE, weight_decay = wd, seed = 1))
  }, numeric(1))
  expect_true(all(diff(norms) < 0))
  expect_error(densemlp(X, y, weight_decay = -1), "non-negative")
})

test_that("linear model with weight decay converges to the ridge solution", {
  set.seed(4)
  n <- 300
  X <- matrix(rnorm(n * 4), n)
  y <- drop(X %*% c(2, -1, 0.5, 0)) + rnorm(n)
  for (wd in c(0, 0.5)) {
    f <- densemlp(X, y, hidden_units = integer(0), epochs = 3000, batch_size = n,
                  lr = 0.01, lr_schedule = "cosine", validation = 0,
                  batch_norm = FALSE, weight_decay = wd, seed = 1)
    Z <- densemlp:::densemlp_apply_blueprint(f$blueprint, X)
    ys <- (y - f$y_center) / f$y_scale
    Zc <- scale(Z, scale = FALSE)
    # minimizer of (1/2n)||ys - b - Zw||^2 + (wd/2)||w||^2
    ridge <- solve(crossprod(Zc) / n + wd * diag(ncol(Z)), crossprod(Zc, ys - mean(ys)) / n)
    expect_equal(drop(f$weights[[1]]), drop(ridge), tolerance = 1e-3)
  }
})
