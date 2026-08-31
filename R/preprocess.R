# Coerces a survival outcome (a `survival::Surv` object, or a two-column
# matrix/data.frame of (time, event)) into a plain two-column numeric
# matrix `cbind(time, event)`. Avoids a hard dependency on the 'survival'
# package: a `Surv` object is unclassed and its first two columns used.
densemlp_as_time_event_matrix <- function(y) {
  if (inherits(y, "Surv")) {
    m <- unclass(y)
    return(matrix(as.numeric(m[, 1:2]), ncol = 2, dimnames = list(NULL, c("time", "event"))))
  }
  if (is.matrix(y) || is.data.frame(y)) {
    if (ncol(y) != 2L) {
      stop("Survival `y` must have exactly two columns: time and event.", call. = FALSE)
    }
    m <- as.matrix(y)
    storage.mode(m) <- "double"
    colnames(m) <- c("time", "event")
    return(m)
  }
  stop("`task = \"survival\"` requires `y` to be a `survival::Surv()` object or a two-column (time, event) matrix/data.frame.", call. = FALSE)
}

# A time grid of `n_bins` cutpoints for the discrete-time Brier-score
# survival head: quantiles of the observed (event or censoring) follow-up
# times, deduplicated and sorted ascending. Always includes the maximum
# observed time as the last breakpoint so every subject's risk set is fully
# resolved by the final bin.
densemlp_time_grid <- function(time, n_bins) {
  probs <- seq(1 / n_bins, 1, length.out = n_bins)
  breaks <- unique(stats::quantile(time, probs = probs, names = FALSE, type = 1))
  breaks <- sort(unique(c(breaks, max(time))))
  breaks
}

# Kaplan-Meier estimate of the *censoring* distribution (reverse KM: treat
# censoring, not the event, as the event of interest), evaluated (a) at
# each cutpoint in `breaks` and (b) at each subject's own observed time
# (needed only for subjects with event = 1, as the IPCW weight for the
# Brier score's event term). Returns a right-continuous step function value
# (last drop at or before the query time; 1 for times before the first
# observed censoring).
densemlp_censoring_km <- function(time, event, breaks) {
  ord <- order(time)
  t_sorted <- time[ord]
  cens_sorted <- 1 - event[ord]
  distinct_times <- unique(t_sorted)
  n <- length(time)
  surv <- numeric(length(distinct_times))
  s <- 1
  for (i in seq_along(distinct_times)) {
    ti <- distinct_times[i]
    n_at_risk <- sum(t_sorted >= ti)
    n_cens_events <- sum(t_sorted == ti & cens_sorted == 1)
    if (n_at_risk > 0) s <- s * (1 - n_cens_events / n_at_risk)
    surv[i] <- s
  }
  eval_at <- function(query) {
    vapply(query, function(q) {
      idx <- which(distinct_times <= q)
      if (length(idx) == 0L) return(1)
      surv[max(idx)]
    }, numeric(1))
  }
  list(
    grid = eval_at(breaks),
    at_time = eval_at(time)
  )
}

# Row-subsets an outcome `y` by `index`, handling both plain vectors/factors
# (subset with `[`) and two-column outcomes -- a `survival::Surv` object or
# a matrix/data.frame -- which need `[index, ]` to keep both columns.
densemlp_index_outcome <- function(y, index) {
  if (inherits(y, "Surv") || is.matrix(y) || is.data.frame(y)) {
    y[index, , drop = FALSE]
  } else {
    y[index]
  }
}

densemlp_train_blueprint <- function(x) {
  x <- as.data.frame(x, stringsAsFactors = FALSE)
  for (nm in names(x)) {
    if (is.character(x[[nm]])) x[[nm]] <- factor(x[[nm]])
    if (is.logical(x[[nm]])) x[[nm]] <- factor(x[[nm]], levels = c(FALSE, TRUE))
  }
  types <- vapply(x, function(col) if (is.numeric(col)) "numeric" else "categorical", character(1))

  numeric_info <- list()
  for (nm in names(types)[types == "numeric"]) {
    values <- x[[nm]]
    impute <- stats::median(values, na.rm = TRUE)
    if (!is.finite(impute)) impute <- 0
    values[is.na(values)] <- impute
    scale <- stats::sd(values)
    if (!is.finite(scale) || scale == 0) scale <- 1
    numeric_info[[nm]] <- list(impute = impute, center = mean(values), scale = scale)
  }

  categorical_info <- list()
  for (nm in names(types)[types == "categorical"]) {
    categorical_info[[nm]] <- list(levels = levels(x[[nm]]))
  }

  list(feature_names = names(x), types = types, numeric = numeric_info, categorical = categorical_info)
}

densemlp_apply_blueprint <- function(blueprint, new_data) {
  new_data <- as.data.frame(new_data, stringsAsFactors = FALSE)
  missing_cols <- setdiff(blueprint$feature_names, names(new_data))
  if (length(missing_cols) > 0L) {
    stop(sprintf("Missing predictors in `new_data`: %s.", paste(missing_cols, collapse = ", ")), call. = FALSE)
  }
  new_data <- new_data[blueprint$feature_names]

  processed <- vector("list", length(blueprint$feature_names))
  names(processed) <- blueprint$feature_names
  for (nm in blueprint$feature_names) {
    if (identical(blueprint$types[[nm]], "numeric")) {
      info <- blueprint$numeric[[nm]]
      values <- as.numeric(new_data[[nm]])
      values[is.na(values)] <- info$impute
      processed[[nm]] <- (values - info$center) / info$scale
    } else {
      info <- blueprint$categorical[[nm]]
      values <- as.character(new_data[[nm]])
      values[is.na(values) | !values %in% info$levels] <- info$levels[1]
      processed[[nm]] <- factor(values, levels = info$levels)
    }
  }
  n <- nrow(new_data)
  column_blocks <- vector("list", length(blueprint$feature_names))
  for (i in seq_along(blueprint$feature_names)) {
    nm <- blueprint$feature_names[i]
    if (identical(blueprint$types[[nm]], "numeric")) {
      column_blocks[[i]] <- matrix(processed[[nm]], nrow = n, ncol = 1L)
    } else {
      levels_nm <- blueprint$categorical[[nm]]$levels
      onehot <- matrix(0, nrow = n, ncol = length(levels_nm))
      onehot[cbind(seq_len(n), match(as.character(processed[[nm]]), levels_nm))] <- 1
      column_blocks[[i]] <- onehot
    }
  }
  mm <- do.call(cbind, column_blocks)
  unname(mm)
}
