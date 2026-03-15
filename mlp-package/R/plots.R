#' Plot training history
#'
#' @param object A fitted `mlp_fit` object.
#' @param ... Unused.
#'
#' @return A ggplot object.
#' @importFrom ggplot2 autoplot
#' @export
autoplot.mlp_fit <- function(object, ...) {
  plot_history(object)
}

#' Plot training history
#'
#' @param object A fitted `mlp_fit` object.
#'
#' @return A ggplot object.
#' @importFrom ggplot2 aes geom_line labs theme_minimal
#' @export
plot_history <- function(object) {
  history <- object$training_history
  ggplot2::ggplot(history, ggplot2::aes(x = epoch)) +
    ggplot2::geom_line(ggplot2::aes(y = train_loss, colour = "train")) +
    ggplot2::geom_line(ggplot2::aes(y = valid_loss, colour = "validation")) +
    ggplot2::labs(x = "Epoch", y = "Loss", colour = "Split") +
    ggplot2::theme_minimal()
}
