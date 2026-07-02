## handles silent setup
.onLoad <- function(libname, pkgname) {
  utils::globalVariables(c(
    "epoch", "train_loss", "valid_loss", "feature", "importance"
  ))
  # Never probe torch here; CRAN Windows checks can fail if we do.
}

## handles user-facing messaging
.onAttach <- function(libname, pkgname) {
  torch_pkg_present <- requireNamespace("torch", quietly = TRUE)

  if (interactive() && (!torch_pkg_present || !isTRUE(torch::torch_is_installed()))) {
    packageStartupMessage(
      "Optional Torch backend is not ready. ",
      "Install the R package 'torch' and then run torch::install_torch() ",
      "to use deep-learning features."
    )
  }

  invisible()
}

## set R + torch seeds safely
densemlp_set_seed <- function(.seed = NULL) {
  if (is.null(.seed)) {
    return(invisible(NULL))
  }

  set.seed(.seed)

  if (requireNamespace("torch", quietly = TRUE) && isTRUE(torch::torch_is_installed())) {
    torch::torch_manual_seed(.seed)
  }

  invisible(NULL)
}

## set torch CPU thread count safely (process-global)
densemlp_set_threads <- function(.threads = NULL) {
  if (is.null(.threads)) {
    return(invisible(NULL))
  }

  if (!is.numeric(.threads) || length(.threads) != 1L || is.na(.threads) ||
      .threads <= 0 || (.threads %% 1 != 0)) {
    abort("`.threads` must be a single positive integer or NULL.")
  }

  if (requireNamespace("torch", quietly = TRUE) && isTRUE(torch::torch_is_installed())) {
    torch::torch_set_num_threads(as.integer(.threads))
  }

  invisible(NULL)
}

## internal utility to choose a torch device
densemlp_get_device <- function(.device = c("auto", "cpu", "cuda")) {
  .device <- match.arg(.device)

  if (!requireNamespace("torch", quietly = TRUE)) {
    abort(
      "The 'torch' package is required to fit densemlp models.\n",
      "Please install it with: install.packages('torch') and then run torch::install_torch()."
    )
  }

  if (!isTRUE(torch::torch_is_installed())) {
    abort(
      "The Torch backend is not installed.\n",
      "Please run: torch::install_torch()."
    )
  }

  if (.device == "cpu") {
    return(torch::torch_device("cpu"))
  }

  if (.device == "cuda") {
    if (!torch::cuda_is_available()) {
      warning("CUDA was requested but is not available; falling back to CPU.")
      return(torch::torch_device("cpu"))
    }
    return(torch::torch_device("cuda"))
  }

  if (torch::cuda_is_available()) {
    torch::torch_device("cuda")
  } else {
    torch::torch_device("cpu")
  }
}
