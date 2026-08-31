test_that("densemlp_metrics computes regression and classification metrics", {
  truth_reg <- c(1, 2, 3, 4)
  est_reg <- c(1.1, 1.9, 3.2, 3.8)
  m <- densemlp_metrics(truth_reg, est_reg, "regression")
  expect_true(m$rmse > 0)
  expect_true(is.finite(m$rsq))

  truth_cls <- factor(c("a", "b", "a", "b"))
  est_cls <- factor(c("a", "b", "b", "b"), levels = levels(truth_cls))
  prob <- cbind(a = c(0.9, 0.2, 0.4, 0.1), b = c(0.1, 0.8, 0.6, 0.9))
  m2 <- densemlp_metrics(truth_cls, est_cls, "binary", prob)
  expect_equal(m2$accuracy, 0.75)
  expect_true(is.finite(m2$macro_auc))
})
