# Apply `fun` over `x` using `ncores` cores. Falls back to serial `lapply()`
# whenever ncores <= 1, on Windows (no fork()), or when only one element is
# supplied. Each worker sets its own RNG seed internally, so results are
# reproducible regardless of the number of cores used.
`%||%` <- function(a, b) if (is.null(a)) b else a

densemlp_parallel_lapply <- function(x, fun, ncores = 1L) {
  ncores <- max(1L, as.integer(ncores))
  if (ncores > 1L && length(x) > 1L && .Platform$OS.type == "unix") {
    return(parallel::mclapply(x, fun, mc.cores = ncores))
  }
  lapply(x, fun)
}
