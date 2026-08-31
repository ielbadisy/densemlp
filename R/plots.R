#' Plot training history
#'
#' @param x A fitted `densemlp` object (single model; ensembles are not
#'   supported since members don't share an epoch axis).
#' @param ... Additional arguments passed to [graphics::matplot()].
#'
#' @return `x`, invisibly.
#' @export
plot.densemlp <- function(x, ...) {
  if (!is.null(x$members)) {
    stop("`plot.densemlp()` does not support ensembles; plot an individual member instead.", call. = FALSE)
  }
  train_history <- x$train_history
  valid_history <- x$valid_history
  epoch <- seq_along(train_history)
  history <- cbind(train = train_history, validation = valid_history[seq_along(train_history)])
  graphics::matplot(epoch, history, type = "l", lty = 1, col = c("#1b6ca8", "#d1495b"),
                     xlab = "Epoch", ylab = "Loss", ...)
  graphics::legend("topright", legend = colnames(history), col = c("#1b6ca8", "#d1495b"),
                    lty = 1, bty = "n")
  invisible(x)
}

#' Plot the training history of a fitted densemlp model
#'
#' A named wrapper around the [plot.densemlp()] method: draws the per-epoch
#' training and validation loss curves.
#'
#' @param object A fitted `densemlp` object (single model).
#' @param ... Passed to [graphics::matplot()].
#'
#' @return `object`, invisibly.
#' @examples
#' set.seed(1)
#' x <- data.frame(a = rnorm(80), b = rnorm(80))
#' y <- x$a - 0.5 * x$b + rnorm(80, sd = 0.1)
#' fit <- densemlp(x, y, epochs = 40, hidden_units = c(16))
#' plot_history(fit)
#' @export
plot_history <- function(object, ...) {
  if (!inherits(object, "densemlp")) {
    stop("`object` must be a <densemlp> fit.", call. = FALSE)
  }
  plot.densemlp(object, ...)
}
