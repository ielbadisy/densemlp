skip_if_no_torch_backend <- function() {
  testthat::skip_on_cran()
  testthat::skip_if_not(
    requireNamespace("torch", quietly = TRUE) &&
      isTRUE(torch::torch_is_installed()),
    "Torch backend is not installed."
  )
}
