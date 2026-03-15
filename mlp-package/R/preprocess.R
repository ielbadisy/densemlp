#' @keywords internal
normalize_predictors <- function(df) {
  out <- df
  for (nm in names(out)) {
    column <- out[[nm]]
    if (is.character(column)) {
      out[[nm]] <- factor(column)
    } else if (is.logical(column)) {
      out[[nm]] <- factor(column, levels = c(FALSE, TRUE))
    }
  }
  out
}

#' @keywords internal
train_blueprint <- function(x) {
  x <- normalize_predictors(as_data_frame_strict(x))
  types <- vapply(
    x,
    function(col) {
      if (is.numeric(col) || is.integer(col)) "numeric" else "categorical"
    },
    character(1)
  )

  numeric_info <- lapply(names(types)[types == "numeric"], function(nm) {
    values <- x[[nm]]
    imput <- stats::median(values, na.rm = TRUE)
    if (!is.finite(imput)) {
      imput <- 0
    }
    values[is.na(values)] <- imput
    scale <- stats::sd(values)
    if (!is.finite(scale) || scale == 0) {
      scale <- 1
    }
    list(name = nm, impute = imput, center = mean(values), scale = scale)
  })
  names(numeric_info) <- names(types)[types == "numeric"]

  categorical_info <- lapply(names(types)[types == "categorical"], function(nm) {
    values <- x[[nm]]
    values <- as.character(values)
    values[is.na(values)] <- "(Missing)"
    levels <- unique(c(values, "(Other)"))
    list(name = nm, levels = levels)
  })
  names(categorical_info) <- names(types)[types == "categorical"]

  blueprint <- list(
    feature_names = names(x),
    types = types,
    numeric = numeric_info,
    categorical = categorical_info
  )
  class(blueprint) <- "mlp_blueprint"
  blueprint
}

#' @keywords internal
apply_blueprint <- function(blueprint, new_data) {
  new_data <- normalize_predictors(as_data_frame_strict(new_data, "new_data"))
  missing_cols <- setdiff(blueprint$feature_names, names(new_data))
  if (length(missing_cols) > 0L) {
    abort(sprintf(
      "Missing predictors in `new_data`: %s.",
      paste(missing_cols, collapse = ", ")
    ))
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
      values[is.na(values)] <- "(Missing)"
      values[!values %in% info$levels] <- "(Other)"
      processed[[nm]] <- factor(values, levels = info$levels)
    }
  }

  processed_df <- as.data.frame(processed, stringsAsFactors = FALSE)
  categorical_names <- names(blueprint$categorical)
  contrasts_arg <- stats::setNames(
    lapply(processed_df[categorical_names], stats::contrasts, contrasts = FALSE),
    categorical_names
  )
  mm <- stats::model.matrix(
    stats::as.formula("~ . - 1"),
    data = processed_df,
    contrasts.arg = contrasts_arg
  )
  matrix_info <- list(
    matrix = unname(mm),
    encoded_names = colnames(mm)
  )
  matrix_info
}

#' @keywords internal
prepare_outcome <- function(y, task) {
  if (identical(task, "regression")) {
    if (!is.numeric(y)) {
      abort("Regression requires a numeric outcome.")
    }
    y <- as.numeric(y)
    center <- mean(y)
    scale <- stats::sd(y)
    if (!is.finite(scale) || scale == 0) {
      scale <- 1
    }
    return(list(
      task = "regression",
      y_train = (y - center) / scale,
      outcome_type = "numeric",
      levels = NULL,
      output_dim = 1L,
      y_center = center,
      y_scale = scale
    ))
  }

  if (is.logical(y)) {
    y <- factor(y, levels = c(FALSE, TRUE))
  } else if (is.character(y)) {
    y <- factor(y)
  } else if (is.numeric(y) && all(stats::na.omit(y) %in% c(0, 1))) {
    y <- factor(y, levels = c(0, 1))
  } else if (!is.factor(y)) {
    abort("Classification requires a factor, logical, character, or 0/1 outcome.")
  }

  if (length(levels(y)) < 2L) {
    abort("Classification outcomes must contain at least two classes.")
  }

  list(
    task = "classification",
    y_train = as.integer(y),
    outcome_type = if (length(levels(y)) == 2L) "binary" else "multiclass",
    levels = levels(y),
    output_dim = if (length(levels(y)) == 2L) 1L else length(levels(y)),
    y_center = NULL,
    y_scale = NULL
  )
}

#' @keywords internal
infer_task <- function(y, task) {
  task <- match.arg(task, c("auto", "classification", "regression"))
  if (!identical(task, "auto")) {
    return(task)
  }
  if (is.numeric(y) && !all(stats::na.omit(y) %in% c(0, 1))) {
    "regression"
  } else {
    "classification"
  }
}
