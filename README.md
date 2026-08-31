# densemlp

`densemlp` provides dense feed-forward neural network models (multilayer
perceptrons) for tabular **regression**, **classification** and
**survival analysis** in R. Forward propagation, backpropagation and
Adam optimization are implemented natively in C++ via `RcppArmadillo`,
with hand-derived closed-form gradients: there is **no `torch` /
`libtorch` dependency**, so the package installs and trains without a
separate deep-learning runtime.

The optional extensions - dropout, batch normalization, residual
connections, gated blocks, a learned cross-feature interaction layer, a
linear input projection, exponential moving-average weights,
learning-rate schedules and internal bootstrap ensembles - enrich the
architecture without changing its model class: the models stay dense
feed-forward neural networks.

`task` is optional. When set to `"auto"` (the default) the task is
inferred from the outcome: numeric to `"regression"`, a factor/character
to `"binary"` / `"multiclass"`, a `survival::Surv()` object to
`"survival"`.

## Installation

``` r
# CRAN
install.packages("densemlp")

# development version
remotes::install_github("ielbadisy/densemlp")
```

## Regression

``` r
library(densemlp)

set.seed(1)
n <- 300
x <- data.frame(a = rnorm(n), b = rnorm(n), c = rnorm(n))
y <- 2 * x$a - x$b + rnorm(n, sd = 0.2)

fit <- densemlp(x, y, hidden_units = c(32, 16), epochs = 60, seed = 1)
pred <- predict(fit, x)
densemlp_metrics(y, pred, task = "regression")
#> $rmse
#> [1] 0.2414633
#> 
#> $nrmse
#> [1] 0.1111125
#> 
#> $rsq
#> [1] 0.9876127
```

## Classification

``` r
fit <- densemlp(Species ~ ., data = iris, epochs = 40, seed = 1)
predict(fit, iris[c(1, 60, 120), ], type = "prob")
#>          setosa versicolor   virginica
#> [1,] 0.99224792 0.00368238 0.004069703
#> [2,] 0.05145259 0.76175233 0.186795086
#> [3,] 0.05642313 0.36431480 0.579262066
predict(fit, iris[c(1, 60, 120), ], type = "class")
#> [1] setosa     versicolor virginica 
#> Levels: setosa versicolor virginica
```

## Survival

``` r
library(survival)
data(lung, package = "survival")
lung <- na.omit(lung[, c("time", "status", "age", "sex", "ph.ecog")])
sy <- Surv(lung$time, lung$status == 2)
sx <- lung[, c("age", "sex", "ph.ecog")]

sfit <- densemlp(sx, sy, task = "survival", hidden_units = c(16, 8), epochs = 60, seed = 1)
densemlp_metrics(sy, predict(sfit, sx, type = "response"), task = "survival")
#> $concordance
#> [1] 0.5200131
```

## More

- `cv_densemlp()` - k-fold cross-validation.
- `tune_densemlp()` - grid search over the architecture and optimizer
  knobs.
- `perm_importance()` - model-agnostic permutation importance.
- `plot_history()` / `plot()` - training and validation loss curves.

See `?densemlp` and `vignette("densemlp-intro")` for the full API.
