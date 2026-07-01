#' Print a fitted dense multilayer perceptron
#'
#' @param x A fitted `densemlp_fit` object.
#' @param ... Unused.
#'
#' @return `x`, invisibly.
#' @export
print.densemlp_fit <- function(x, ...) {
  cat("<densemlp_fit>\n")
  cat(sprintf("Task: %s\n", x$task))
  if (!is.null(x$levels)) {
    cat(sprintf("Outcome levels: %s\n", paste(x$levels, collapse = ", ")))
  }
  cat(sprintf("Encoded features: %d\n", x$n_features))
  cat(sprintf("Best epoch: %d\n", x$best_epoch))
  invisible(x)
}
