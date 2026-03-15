#' Print a fitted MLP
#'
#' @param x A fitted `mlp_fit` object.
#' @param ... Unused.
#'
#' @return `x`, invisibly.
#' @export
print.mlp_fit <- function(x, ...) {
  cat("<mlp_fit>\n")
  cat(sprintf("Task: %s\n", x$task))
  if (!is.null(x$levels)) {
    cat(sprintf("Outcome levels: %s\n", paste(x$levels, collapse = ", ")))
  }
  cat(sprintf("Encoded features: %d\n", x$n_features))
  cat(sprintf("Best epoch: %d\n", x$best_epoch))
  invisible(x)
}
