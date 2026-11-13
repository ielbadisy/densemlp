#' Deprecated functions in densemlp
#'
#' These functions were renamed to glued lowercase names and will be
#' removed in a future release. Use the new names instead.
#'
#' @name densemlp-deprecated
#' @param ... Arguments passed to the new function.
#' @return The value of the new function.
#' @keywords internal
NULL

#' @rdname densemlp-deprecated
#' @export
cv_densemlp <- function(...) {
  .Deprecated("cvdensemlp")
  cvdensemlp(...)
}

#' @rdname densemlp-deprecated
#' @export
densemlp_integrated_brier_score <- function(...) {
  .Deprecated("densemlpintegratedbrierscore")
  densemlpintegratedbrierscore(...)
}

#' @rdname densemlp-deprecated
#' @export
densemlp_metrics <- function(...) {
  .Deprecated("densemlpmetrics")
  densemlpmetrics(...)
}

#' @rdname densemlp-deprecated
#' @export
group_effects <- function(...) {
  .Deprecated("groupeffects")
  groupeffects(...)
}

#' @rdname densemlp-deprecated
#' @export
perm_importance <- function(...) {
  .Deprecated("permimportance")
  permimportance(...)
}

#' @rdname densemlp-deprecated
#' @export
plot_history <- function(...) {
  .Deprecated("plothistory")
  plothistory(...)
}

#' @rdname densemlp-deprecated
#' @export
tune_densemlp <- function(...) {
  .Deprecated("tunedensemlp")
  tunedensemlp(...)
}
