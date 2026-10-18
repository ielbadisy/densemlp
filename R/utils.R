# Apply `fun` over `x` using `ncores` cores. Falls back to serial `lapply()`
# whenever ncores <= 1, on Windows (no fork()), or when only one element is
# supplied. Each worker sets its own RNG seed internally, so results are
# reproducible regardless of the number of cores used.
`%||%` <- function(a, b) if (is.null(a)) b else a

densemlp_parallel_lapply <- function(x, fun, ncores = 1L) {
  ncores <- max(1L, as.integer(ncores))
  if (ncores > 1L && length(x) > 1L && .Platform$OS.type == "unix") {
    # A forked child that enters the threaded BLAS/OpenMP pool inherited from
    # the parent (MKL, OpenBLAS) can deadlock; each worker runs its BLAS on
    # one thread instead (the parallelism comes from the workers).
    worker <- function(i) {
      RhpcBLASctl::blas_set_num_threads(1L)
      RhpcBLASctl::omp_set_num_threads(1L)
      fun(i)
    }
    return(parallel::mclapply(x, worker, mc.cores = ncores))
  }
  lapply(x, fun)
}

# Normalizes `groups` to a named list of factors, one per grouping
# variable, each of length `n` with no missing values. NULL stays NULL.
densemlp_as_groups <- function(groups, n) {
  if (is.null(groups)) return(NULL)
  if (is.data.frame(groups)) groups <- as.list(groups)
  if (!is.list(groups)) groups <- list(group = groups)
  if (length(groups) == 0L) return(NULL)
  if (is.null(names(groups))) names(groups) <- paste0("group", seq_along(groups))
  empty <- !nzchar(names(groups))
  names(groups)[empty] <- paste0("group", which(empty))
  groups <- lapply(groups, function(g) if (is.factor(g)) g else factor(g))
  for (nm in names(groups)) {
    if (length(groups[[nm]]) != n) {
      stop(sprintf("Grouping factor `%s` must have one value per row.", nm), call. = FALSE)
    }
    if (anyNA(groups[[nm]])) {
      stop(sprintf("Grouping factor `%s` contains missing values.", nm), call. = FALSE)
    }
  }
  groups
}

# 0-based level codes of `groups` against the levels seen in training;
# unseen (or missing) levels get -1 and contribute no intercept.
densemlp_group_codes <- function(object, groups, n) {
  if (length(object$group_levels) == 0L) return(list())
  if (is.null(groups)) {
    return(lapply(object$group_levels, function(l) rep(-1L, n)))
  }
  if (is.data.frame(groups)) groups <- as.list(groups)
  if (!is.list(groups)) groups <- list(groups)
  if (is.null(names(groups))) names(groups) <- names(object$group_levels)[seq_along(groups)]
  lapply(names(object$group_levels), function(nm) {
    g <- groups[[nm]]
    if (is.null(g)) return(rep(-1L, n))
    if (length(g) != n) {
      stop(sprintf("Grouping factor `%s` must have one value per row of `newdata`.", nm), call. = FALSE)
    }
    code <- match(as.character(g), object$group_levels[[nm]]) - 1L
    code[is.na(code)] <- -1L
    code
  })
}

# Validates an offset and returns it as an n x output_dim numeric matrix.
densemlp_offset_matrix <- function(offset, n, output_dim) {
  offset <- as.matrix(offset)
  storage.mode(offset) <- "double"
  if (nrow(offset) != n) stop("`offset` must have one value (row) per observation.", call. = FALSE)
  if (ncol(offset) != output_dim) {
    stop(sprintf("`offset` must have %d column(s), one per output.", output_dim), call. = FALSE)
  }
  if (any(!is.finite(offset))) stop("`offset` must be finite.", call. = FALSE)
  unname(offset)
}

#' Extract the learned group intercepts
#'
#' Returns the per-level intercepts that a model fitted with `groups`
#' adds to its linear predictor, on the scale of the outcome for
#' regression and of the link otherwise.
#'
#' @param object A fitted `densemlp` object fitted with `groups`.
#' @param ... Unused.
#' @return A named list with one entry per grouping factor: a named numeric
#'   vector of intercepts (one per level), or for multiclass a matrix with
#'   one row per level and one column per class. For an ensemble, the
#'   members' intercepts are averaged.
#' @export
group_effects <- function(object, ...) {
  UseMethod("group_effects")
}

#' @rdname group_effects
#' @export
group_effects.densemlp <- function(object, ...) {
  if (!is.null(object$members)) {
    per_member <- lapply(object$members, group_effects.densemlp)
    nms <- names(per_member[[1L]])
    return(stats::setNames(lapply(nms, function(nm) {
      vals <- lapply(per_member, function(m) m[[nm]])
      Reduce(`+`, vals) / length(vals)
    }), nms))
  }
  if (length(object$group_weights) == 0L) {
    stop("This model was fitted without `groups`.", call. = FALSE)
  }
  out <- lapply(names(object$group_weights), function(nm) {
    w <- object$group_weights[[nm]] * object$y_scale
    if (ncol(w) == 1L) {
      stats::setNames(as.numeric(w), object$group_levels[[nm]])
    } else {
      dimnames(w) <- list(object$group_levels[[nm]], object$levels)
      w
    }
  })
  stats::setNames(out, names(object$group_weights))
}
