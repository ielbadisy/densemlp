# tabularmlp

`tabularmlp` provides formula-first multilayer perceptrons for tabular
classification and regression in R using `torch`.

## Installation

```r
# For regular use
devtools::install(".")
```

```r
# For local development without reinstalling after each edit
pkgload::load_all(".")
```

## Example

```r
library(tabularmlp)

fit <- mlp(
  Species ~ .,
  data = iris,
  epochs = 20,
  validation = 0.2,
  verbose = FALSE
)

predict(fit, iris[1:5, ], type = "class")
predict(fit, iris[1:5, ], type = "prob")
```

If you are working from the source tree, do not `source("R/mlp.R")` by itself.
`mlp()` depends on internal package helpers such as `normalize_hidden_units()`,
which are available when the package is installed or loaded with
`pkgload::load_all(".")`.

## Status

Version `0.5.0` supports binary classification, multiclass classification, and
regression for tabular data frames with automatic preprocessing.
