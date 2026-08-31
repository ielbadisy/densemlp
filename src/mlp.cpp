// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// Task codes: 0 = regression (linear output, MSE), 1 = binary (sigmoid
// output, BCE), 2 = multiclass (softmax output, categorical CE), 3 =
// survival (linear risk score, batch-wise Breslow-tie Cox partial
// likelihood). The first three share the same sigmoid/softmax +
// cross-entropy gradient simplification dZ_out = (yhat - y) / n, so the
// backward pass only branches on the output activation, not the loss; task
// 3 uses its own closed-form Cox gradient (see cox_loss_and_grad below)
// wherever that simplification is used.
//
// Hidden layers are Linear -> (optional BatchNorm) -> ReLU -> optional gate
// -> optional inverted dropout -> optional residual projection. An optional
// linear input projection maps the raw inputs to a lower/other dimension
// before the first hidden layer (no activation, no BN); an optional O(p)
// cross layer models input interactions before the MLP (the two are
// mutually exclusive). Optional EMA parameters are used for validation /
// final output. The output layer is plain Linear -> task activation, no BN.
// Training supports a held-out validation split and early stopping on
// validation loss, restoring all best-epoch parameters.
//
// Task 3 (survival) trains a single linear risk score eta = f(x) against a
// two-column (time, event) target using a batch-wise Breslow-tie Cox
// partial log-likelihood -- the same "risk set = current mini-batch"
// approximation used by DeepSurv/pycox, which turns Cox regression into an
// ordinary per-batch loss compatible with the rest of this file's masked-free
// Adam training loop, BN, and early stopping unchanged. For a batch sorted
// by descending time, with risk sets R_i = {j : t_j >= t_i} approximated by
// the sorted prefix:
//   L = -(1/E) * sum_{i: event_i=1} [eta_i - log sum_{j in R_i} exp(eta_j)]
// computed via a single cumulative sum (see cox_loss_and_grad), where E is
// the number of events in the batch. Its gradient has the same closed form
// as ordinary Cox regression and is O(n) per batch.

static const double BN_EPS = 1e-5;
static const double BN_DECAY = 0.9; // running = DECAY*running + (1-DECAY)*batch

static inline mat relu(const mat &z) { return arma::clamp(z, 0.0, arma::datum::inf); }
static inline mat relu_grad(const mat &z) { return arma::conv_to<mat>::from(z > 0.0); }
static inline mat sigmoid(const mat &z) { return 1.0 / (1.0 + arma::exp(-z)); }

static mat softmax_rows(const mat &z) {
  mat shifted = z.each_col() - arma::max(z, 1);
  mat e = arma::exp(shifted);
  vec row_sums = arma::sum(e, 1);
  return e.each_col() / row_sums;
}

struct MLPParams {
  std::vector<mat> W;              // n_layers = n_hidden + 1 (last is output)
  std::vector<vec> B;
  std::vector<vec> gamma, beta;          // n_hidden BN affine params
  std::vector<vec> running_mean, running_var; // n_hidden BN running stats
  std::vector<mat> residual_W;      // empty for identity/no skip, projection otherwise
  std::vector<vec> residual_B;
  std::vector<mat> gate_W;          // empty when gating is disabled
  std::vector<vec> gate_B;
  vec interaction_W;                // one efficient cross-feature layer
  vec interaction_B;
  mat proj_W;                      // linear input projection (empty when disabled)
  vec proj_B;
};

static void deep_copy(MLPParams &dst, const MLPParams &src) { dst = src; }

static MLPParams init_params(const std::vector<uword> &sizes, unsigned int seed,
                             bool residual, bool gated, bool interaction,
                             uword proj_in) {
  arma::arma_rng::set_seed(seed);
  MLPParams p;
  size_t n_layers = sizes.size() - 1;
  if (proj_in > 0) {
    double scale = std::sqrt(2.0 / static_cast<double>(proj_in));
    p.proj_W = arma::randn(proj_in, sizes[0]) * scale;
    p.proj_B = arma::zeros(sizes[0]);
  }
  for (size_t l = 0; l < n_layers; ++l) {
    double fan_in = static_cast<double>(sizes[l]);
    double scale = std::sqrt(2.0 / fan_in);
    p.W.push_back(arma::randn(sizes[l], sizes[l + 1]) * scale);
    p.B.push_back(arma::zeros(sizes[l + 1]));
    if (l + 1 < n_layers) { // hidden layer -> has BN affine slots
      p.gamma.push_back(arma::ones(sizes[l + 1]));
      p.beta.push_back(arma::zeros(sizes[l + 1]));
      p.running_mean.push_back(arma::zeros(sizes[l + 1]));
      p.running_var.push_back(arma::ones(sizes[l + 1]));
      if (residual && sizes[l] != sizes[l + 1]) {
        p.residual_W.push_back(arma::randn(sizes[l], sizes[l + 1]) * scale);
        p.residual_B.push_back(arma::zeros(sizes[l + 1]));
      } else {
        p.residual_W.push_back(mat());
        p.residual_B.push_back(vec());
      }
      if (gated) {
        double gate_scale = std::sqrt(2.0 / static_cast<double>(sizes[l + 1]));
        p.gate_W.push_back(arma::randn(sizes[l + 1], sizes[l + 1]) * gate_scale);
        p.gate_B.push_back(arma::zeros(sizes[l + 1]));
      } else {
        p.gate_W.push_back(mat());
        p.gate_B.push_back(vec());
      }
    }
  }
  if (interaction) {
    p.interaction_W = arma::zeros(sizes[0]);
    p.interaction_B = arma::zeros(sizes[0]);
  }
  return p;
}

// Caches populated by forward_train(), consumed by backward().
struct ForwardCache {
  std::vector<mat> A;          // A[0] = (projected) input, A[l+1] = layer l's output
  std::vector<mat> Z;          // pre-BN linear output, hidden layers only
  std::vector<mat> Zbn;        // post-BN pre-ReLU (== Z when BN off), hidden layers only
  std::vector<rowvec> batch_mean, batch_var; // hidden layers only (empty when BN off)
  std::vector<mat> Xhat;       // normalized pre-affine, hidden layers only (empty when BN off)
  std::vector<mat> H;          // post-ReLU, before optional gate/dropout
  std::vector<mat> Gate;       // sigmoid gate, empty when disabled
  std::vector<mat> DropMask;   // inverted-dropout mask, empty when disabled
  mat proj_input;              // raw input, before the linear projection
  mat raw_input;               // input to the MLP proper (post-projection), pre-interaction
  vec interaction_scale;       // raw_input * interaction_W
};

static mat apply_projection(const MLPParams &p, const mat &X, bool use_proj) {
  if (!use_proj) return X;
  mat out = X * p.proj_W;
  out.each_row() += p.proj_B.t();
  return out;
}

static mat apply_interaction(const MLPParams &p, const mat &X, bool interaction) {
  if (!interaction) return X;
  vec scale = X * p.interaction_W;
  mat out = X.each_col() % scale;
  out += X;
  out.each_row() += p.interaction_B.t();
  return out;
}

static void forward_train(MLPParams &p, const mat &X, int task, bool residual,
                          bool gated, const vec &dropout, bool interaction,
                          bool batch_norm, bool use_proj, ForwardCache &c) {
  size_t n_layers = p.W.size();
  size_t n_hidden = n_layers - 1;
  c.A.clear(); c.Z.clear(); c.Zbn.clear(); c.batch_mean.clear(); c.batch_var.clear(); c.Xhat.clear();
  c.H.clear(); c.Gate.clear(); c.DropMask.clear();
  c.proj_input = X;
  mat Xin = apply_projection(p, X, use_proj);
  c.raw_input = Xin;
  if (interaction) c.interaction_scale = Xin * p.interaction_W;
  c.A.push_back(apply_interaction(p, Xin, interaction));

  for (size_t l = 0; l < n_hidden; ++l) {
    mat z = c.A.back() * p.W[l];
    z.each_row() += p.B[l].t();
    mat zbn;
    if (batch_norm) {
      rowvec bmean = arma::mean(z, 0);
      rowvec bvar = arma::var(z, 1, 0); // population variance (divide by n)
      rowvec bstd = arma::sqrt(bvar + BN_EPS);
      mat xhat = z.each_row() - bmean;
      xhat.each_row() /= bstd;
      zbn = xhat.each_row() % p.gamma[l].t();
      zbn.each_row() += p.beta[l].t();

      p.running_mean[l] = BN_DECAY * p.running_mean[l] + (1 - BN_DECAY) * bmean.t();
      p.running_var[l] = BN_DECAY * p.running_var[l] + (1 - BN_DECAY) * bvar.t();

      c.batch_mean.push_back(bmean);
      c.batch_var.push_back(bvar);
      c.Xhat.push_back(xhat);
    } else {
      zbn = z;
      c.batch_mean.push_back(rowvec());
      c.batch_var.push_back(rowvec());
      c.Xhat.push_back(mat());
    }

    c.Z.push_back(z);
    c.Zbn.push_back(zbn);
    mat h = relu(zbn);
    mat out = h;
    mat gate;
    if (gated) {
      mat gate_z = h * p.gate_W[l];
      gate_z.each_row() += p.gate_B[l].t();
      gate = sigmoid(gate_z);
      out %= gate;
    }
    mat drop_mask;
    if (dropout[l] > 0.0) {
      drop_mask = arma::conv_to<mat>::from(arma::randu<mat>(size(out)) >= dropout[l]);
      drop_mask /= (1.0 - dropout[l]);
      out %= drop_mask;
    }
    c.H.push_back(h);
    c.Gate.push_back(gate);
    c.DropMask.push_back(drop_mask);
    if (residual) {
      if (p.residual_W[l].is_empty()) {
        out += c.A.back();
      } else {
        mat skip = c.A.back() * p.residual_W[l];
        skip.each_row() += p.residual_B[l].t();
        out += skip;
      }
    }
    c.A.push_back(out);
  }

  mat zout = c.A.back() * p.W[n_hidden];
  zout.each_row() += p.B[n_hidden].t();
  mat out;
  if (task == 0 || task == 3) out = zout; // regression / survival: linear risk score
  else if (task == 1) out = sigmoid(zout);
  else out = softmax_rows(zout);
  c.A.push_back(out);
}

// Inference-mode forward: uses running mean/var instead of batch statistics,
// no caching, no running-stat updates. Used for validation loss and predict().
static mat forward_eval(const MLPParams &p, const mat &X, int task, bool residual,
                        bool gated, bool interaction, bool batch_norm, bool use_proj) {
  size_t n_layers = p.W.size();
  size_t n_hidden = n_layers - 1;
  mat a = apply_interaction(p, apply_projection(p, X, use_proj), interaction);
  for (size_t l = 0; l < n_hidden; ++l) {
    mat input = a;
    mat z = input * p.W[l];
    z.each_row() += p.B[l].t();
    mat zbn;
    if (batch_norm) {
      mat xhat = z.each_row() - p.running_mean[l].t();
      xhat.each_row() /= arma::sqrt(p.running_var[l].t() + BN_EPS);
      zbn = xhat.each_row() % p.gamma[l].t();
      zbn.each_row() += p.beta[l].t();
    } else {
      zbn = z;
    }
    a = relu(zbn);
    if (gated) {
      mat gate_z = a * p.gate_W[l];
      gate_z.each_row() += p.gate_B[l].t();
      a %= sigmoid(gate_z);
    }
    if (residual) {
      if (p.residual_W[l].is_empty()) {
        a += input;
      } else {
        mat skip = input * p.residual_W[l];
        skip.each_row() += p.residual_B[l].t();
        a += skip;
      }
    }
  }
  mat zout = a * p.W[n_hidden];
  zout.each_row() += p.B[n_hidden].t();
  if (task == 0 || task == 3) return zout; // regression / survival: linear risk score
  if (task == 1) return sigmoid(zout);
  return softmax_rows(zout);
}

// Batch-wise Breslow-tie Cox negative partial log-likelihood and its
// gradient with respect to the linear risk score eta (see the file-header
// comment for the derivation). Risk sets are approximated by the
// time-sorted batch. `time`/`event` are the two Y columns for task 3.
// Writes the per-observation gradient into `grad_eta` (original order) and
// returns the scalar loss. If the batch has no events, the loss and
// gradient are both zero (Cox contributes nothing to learn from).
static double cox_loss_and_grad(const vec &eta, const vec &time, const vec &event, vec &grad_eta) {
  uword n = eta.n_elem;
  uvec order = arma::sort_index(time, "descend");
  vec eta_s = eta(order);
  vec ev_s = event(order);
  double n_events = arma::accu(ev_s);
  grad_eta = arma::zeros(n);
  if (n_events < 1.0) return 0.0;

  double eta_max = eta_s.max();
  vec exp_eta = arma::exp(eta_s - eta_max); // shift for numerical stability
  vec cumsum_exp = arma::cumsum(exp_eta);
  vec log_risk = arma::log(cumsum_exp) + eta_max;

  double loss = -arma::accu(ev_s % (eta_s - log_risk)) / n_events;

  vec term = ev_s / cumsum_exp;
  vec rev_cumsum(n);
  double running = 0.0;
  for (int i = (int)n - 1; i >= 0; --i) {
    running += term[(uword)i];
    rev_cumsum[(uword)i] = running;
  }
  vec grad_s = (exp_eta % rev_cumsum - ev_s) / n_events;
  grad_eta(order) = grad_s;
  return loss;
}

// IPCW (Graf et al.) integrated Brier score and its gradient with respect
// to the K per-bin hazard logits Z, for a discrete-time survival head:
//   h_k = sigmoid(Z_k)            per-bin conditional hazard
//   S_k = prod_{j<=k} (1 - h_j)   survival probability at breaks[k]
// At each bin k, subjects split into three (mutually exclusive) groups:
//   - had the event at or before breaks[k]: contributes S_k^2 / Ghat(T_i)
//   - known alive past breaks[k] (T_i > breaks[k]): contributes
//     (1-S_k)^2 / Ghat(breaks[k])
//   - censored at or before breaks[k] with no event yet: excluded (Graf's
//     IPCW correction -- their status at breaks[k] is unknown)
// where Ghat is the Kaplan-Meier estimate of the *censoring* distribution
// (fixed, precomputed once on the training set in R -- it does not depend
// on the network's parameters). `Y` here is (time, event, ghat_at_time):
// ghat_at_time[i] = Ghat(T_i), needed only for observations with event=1.
// `ghat_grid[k]` = Ghat(breaks[k]). IBS = mean_k BS(breaks[k]) (equally
// weighted bins). The gradient is exact and closed-form: bin k's Brier
// term depends on S_k, which depends on every h_m for m <= k, so
//   dS_k/dZ_m = -S_k * h_m   (m <= k, 0 otherwise)
// and dLoss/dZ_m accumulates that contribution across every k >= m.
static double brier_loss_and_grad(const mat &Z, const mat &Y, const vec &breaks,
                                  const vec &ghat_grid, mat &grad_Z) {
  uword n = Z.n_rows, K = Z.n_cols;
  const double eps = 1e-8;
  mat H = sigmoid(Z);
  mat S(n, K);
  for (uword i = 0; i < n; ++i) {
    double s = 1.0;
    for (uword k = 0; k < K; ++k) {
      s *= (1.0 - H(i, k));
      S(i, k) = s;
    }
  }

  vec time = Y.col(0), event = Y.col(1), ghat_time = Y.col(2);
  grad_Z.zeros(n, K);
  double total_loss = 0.0;

  for (uword k = 0; k < K; ++k) {
    double bs_k = 0.0;
    vec A(n, fill::zeros); // dBS_k/dS_k per observation
    for (uword i = 0; i < n; ++i) {
      double Ski = S(i, k);
      if (time(i) <= breaks(k) && event(i) == 1.0) {
        double w = 1.0 / std::max(ghat_time(i), eps);
        bs_k += w * Ski * Ski;
        A(i) = 2.0 * w * Ski;
      } else if (time(i) > breaks(k)) {
        double w = 1.0 / std::max(ghat_grid(k), eps);
        bs_k += w * (1.0 - Ski) * (1.0 - Ski);
        A(i) = -2.0 * w * (1.0 - Ski);
      }
    }
    total_loss += bs_k / (double)n;

    for (uword i = 0; i < n; ++i) {
      double contrib = A(i) * S(i, k) / ((double)n * (double)K);
      for (uword m = 0; m <= k; ++m) {
        grad_Z(i, m) += -H(i, m) * contrib;
      }
    }
  }
  return total_loss / (double)K;
}

static double eval_loss(const MLPParams &p, const mat &X, const mat &Y, int task,
                        bool residual, bool gated, bool interaction, bool batch_norm,
                        bool use_proj, int survival_loss,
                        const vec &breaks, const vec &ghat_grid) {
  mat yhat = forward_eval(p, X, task, residual, gated, interaction, batch_norm, use_proj);
  double n = (double)X.n_rows;
  if (task == 0) return arma::accu(arma::square(yhat - Y)) / n;
  if (task == 3) {
    if (survival_loss == 1) {
      mat grad_unused;
      return brier_loss_and_grad(yhat, Y, breaks, ghat_grid, grad_unused);
    }
    vec grad_unused;
    return cox_loss_and_grad(yhat.col(0), Y.col(0), Y.col(1), grad_unused);
  }
  mat clipped = arma::clamp(yhat, 1e-9, 1.0 - 1e-9);
  return -arma::accu(Y % arma::log(clipped)) / n;
}

// Backpropagates dZ_out = (yhat - Y)/n through the whole network, applying
// Adam updates to every parameter tensor (W, B, BN gamma/beta for hidden
// layers when BN is on, the residual/gate/interaction params, and the
// linear input projection) in place.
struct AdamState {
  std::vector<mat> mW, vW;
  std::vector<vec> mB, vB, mGamma, vGamma, mBeta, vBeta;
  std::vector<mat> mResidualW, vResidualW;
  std::vector<vec> mResidualB, vResidualB;
  std::vector<mat> mGateW, vGateW;
  std::vector<vec> mGateB, vGateB;
  vec mInteractionW, vInteractionW, mInteractionB, vInteractionB;
  mat mProjW, vProjW;
  vec mProjB, vProjB;
  int t = 0;
};

static AdamState init_adam(const MLPParams &p) {
  AdamState s;
  for (size_t l = 0; l < p.W.size(); ++l) {
    s.mW.push_back(arma::zeros(size(p.W[l])));
    s.vW.push_back(arma::zeros(size(p.W[l])));
    s.mB.push_back(arma::zeros(size(p.B[l])));
    s.vB.push_back(arma::zeros(size(p.B[l])));
  }
  for (size_t l = 0; l < p.gamma.size(); ++l) {
    s.mGamma.push_back(arma::zeros(size(p.gamma[l])));
    s.vGamma.push_back(arma::zeros(size(p.gamma[l])));
    s.mBeta.push_back(arma::zeros(size(p.beta[l])));
    s.vBeta.push_back(arma::zeros(size(p.beta[l])));
    s.mResidualW.push_back(arma::zeros(size(p.residual_W[l])));
    s.vResidualW.push_back(arma::zeros(size(p.residual_W[l])));
    s.mResidualB.push_back(arma::zeros(size(p.residual_B[l])));
    s.vResidualB.push_back(arma::zeros(size(p.residual_B[l])));
    s.mGateW.push_back(arma::zeros(size(p.gate_W[l])));
    s.vGateW.push_back(arma::zeros(size(p.gate_W[l])));
    s.mGateB.push_back(arma::zeros(size(p.gate_B[l])));
    s.vGateB.push_back(arma::zeros(size(p.gate_B[l])));
  }
  s.mInteractionW = arma::zeros(size(p.interaction_W));
  s.vInteractionW = arma::zeros(size(p.interaction_W));
  s.mInteractionB = arma::zeros(size(p.interaction_B));
  s.vInteractionB = arma::zeros(size(p.interaction_B));
  s.mProjW = arma::zeros(size(p.proj_W));
  s.vProjW = arma::zeros(size(p.proj_W));
  s.mProjB = arma::zeros(size(p.proj_B));
  s.vProjB = arma::zeros(size(p.proj_B));
  return s;
}

static void adam_step(mat &param, mat &m, mat &v, const mat &grad, double lr, double bc1, double bc2) {
  const double beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
  m = beta1 * m + (1 - beta1) * grad;
  v = beta2 * v + (1 - beta2) * arma::square(grad);
  mat mhat = m / bc1, vhat = v / bc2;
  param -= lr * mhat / (arma::sqrt(vhat) + eps);
}
static void adam_step(vec &param, vec &m, vec &v, const vec &grad, double lr, double bc1, double bc2) {
  const double beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
  m = beta1 * m + (1 - beta1) * grad;
  v = beta2 * v + (1 - beta2) * arma::square(grad);
  vec mhat = m / bc1, vhat = v / bc2;
  param -= lr * mhat / (arma::sqrt(vhat) + eps);
}

static void backward_and_update(MLPParams &p, AdamState &s, const ForwardCache &c,
                                 const mat &Y, double lr, int task, bool residual, bool gated,
                                 const vec &dropout, bool interaction, bool batch_norm,
                                 bool use_proj, int survival_loss,
                                 const vec &breaks, const vec &ghat_grid) {
  size_t n_hidden = p.gamma.size();
  double nb = (double)c.A[0].n_rows;

  s.t++;
  double bc1 = 1.0 - std::pow(0.9, s.t);
  double bc2 = 1.0 - std::pow(0.999, s.t);

  mat dZ;
  if (task == 3 && survival_loss == 1) {
    brier_loss_and_grad(c.A.back(), Y, breaks, ghat_grid, dZ);
  } else if (task == 3) {
    vec grad_eta;
    cox_loss_and_grad(c.A.back().col(0), Y.col(0), Y.col(1), grad_eta);
    dZ = grad_eta;
  } else {
    dZ = (c.A.back() - Y) / nb; // output layer gradient (shared simplification)
  }

  // Output layer (index n_hidden in W/B, no BN).
  {
    size_t l = n_hidden;
    mat dW = c.A[l].t() * dZ;
    vec dB = arma::sum(dZ, 0).t();
    mat dA_prev = dZ * p.W[l].t();
    adam_step(p.W[l], s.mW[l], s.vW[l], dW, lr, bc1, bc2);
    adam_step(p.B[l], s.mB[l], s.vB[l], dB, lr, bc1, bc2);
    dZ = dA_prev; // becomes dA for the last hidden layer's output (pre-ReLU-grad)
  }

  for (int li = (int)n_hidden - 1; li >= 0; --li) {
    size_t l = (size_t)li;
    mat dA = dZ; // gradient wrt the block output
    mat dMain = dA;
    if (dropout[l] > 0.0) dMain %= c.DropMask[l];

    mat dGateW;
    vec dGateB;
    mat dH;
    if (gated) {
      mat dGateZ = dMain % c.H[l] % c.Gate[l] % (1.0 - c.Gate[l]);
      dGateW = c.H[l].t() * dGateZ;
      dGateB = arma::sum(dGateZ, 0).t();
      dH = dMain % c.Gate[l] + dGateZ * p.gate_W[l].t();
    } else {
      dH = dMain;
    }
    mat dZbn = dH % relu_grad(c.Zbn[l]);

    vec dgamma, dbeta;
    mat dZlin;
    if (batch_norm) {
      dgamma = arma::sum(dZbn % c.Xhat[l], 0).t();
      dbeta = arma::sum(dZbn, 0).t();

      mat dxhat = dZbn.each_row() % p.gamma[l].t();
      rowvec std_l = arma::sqrt(c.batch_var[l] + BN_EPS);
      mat centered = c.Z[l].each_row() - c.batch_mean[l];

      rowvec inv_var_pow = arma::pow(c.batch_var[l] + BN_EPS, -1.5);
      rowvec dvar = (arma::sum(dxhat % centered, 0) * (-0.5)) % inv_var_pow;
      rowvec mean_neg2centered = arma::mean(centered, 0) * (-2.0);
      rowvec dmean = arma::sum(dxhat.each_row() / (-std_l), 0) + (dvar % mean_neg2centered);

      dZlin = dxhat.each_row() / std_l;
      dZlin += centered.each_row() % (2.0 / nb * dvar);
      dZlin.each_row() += dmean / nb;
    } else {
      dZlin = dZbn;
    }

    mat dW = c.A[l].t() * dZlin;
    vec dB = arma::sum(dZlin, 0).t();
    mat dA_prev = dZlin * p.W[l].t();

    mat dResidualW;
    vec dResidualB;
    if (residual) {
      if (p.residual_W[l].is_empty()) {
        dA_prev += dA;
      } else {
        dResidualW = c.A[l].t() * dA;
        dResidualB = arma::sum(dA, 0).t();
        dA_prev += dA * p.residual_W[l].t();
      }
    }

    adam_step(p.W[l], s.mW[l], s.vW[l], dW, lr, bc1, bc2);
    adam_step(p.B[l], s.mB[l], s.vB[l], dB, lr, bc1, bc2);
    if (batch_norm) {
      adam_step(p.gamma[l], s.mGamma[l], s.vGamma[l], dgamma, lr, bc1, bc2);
      adam_step(p.beta[l], s.mBeta[l], s.vBeta[l], dbeta, lr, bc1, bc2);
    }
    if (residual && !p.residual_W[l].is_empty()) {
      adam_step(p.residual_W[l], s.mResidualW[l], s.vResidualW[l], dResidualW, lr, bc1, bc2);
      adam_step(p.residual_B[l], s.mResidualB[l], s.vResidualB[l], dResidualB, lr, bc1, bc2);
    }
    if (gated) {
      adam_step(p.gate_W[l], s.mGateW[l], s.vGateW[l], dGateW, lr, bc1, bc2);
      adam_step(p.gate_B[l], s.mGateB[l], s.vGateB[l], dGateB, lr, bc1, bc2);
    }

    dZ = dA_prev;
  }

  if (interaction) {
    vec interaction_sum = arma::sum(dZ % c.raw_input, 1);
    vec dInteractionW = c.raw_input.t() * interaction_sum;
    vec dInteractionB = arma::sum(dZ, 0).t();
    adam_step(p.interaction_W, s.mInteractionW, s.vInteractionW,
              dInteractionW, lr, bc1, bc2);
    adam_step(p.interaction_B, s.mInteractionB, s.vInteractionB,
              dInteractionB, lr, bc1, bc2);
  }

  if (use_proj) {
    // `interaction` is disabled whenever `use_proj` is on (enforced in R),
    // so after the hidden loop dZ is exactly the gradient wrt the projected
    // input Xin = proj_input * proj_W + proj_B.
    mat dProjW = c.proj_input.t() * dZ;
    vec dProjB = arma::sum(dZ, 0).t();
    adam_step(p.proj_W, s.mProjW, s.vProjW, dProjW, lr, bc1, bc2);
    adam_step(p.proj_B, s.mProjB, s.vProjB, dProjB, lr, bc1, bc2);
  }
}

static void ema_update_mat(mat &target, const mat &source, double decay) {
  if (!target.is_empty()) target = decay * target + (1.0 - decay) * source;
}

static void ema_update_vec(vec &target, const vec &source, double decay) {
  if (!target.is_empty()) target = decay * target + (1.0 - decay) * source;
}

static void update_ema(MLPParams &ema, const MLPParams &p, double decay) {
  for (size_t l = 0; l < p.W.size(); ++l) {
    ema_update_mat(ema.W[l], p.W[l], decay);
    ema_update_vec(ema.B[l], p.B[l], decay);
  }
  for (size_t l = 0; l < p.gamma.size(); ++l) {
    ema_update_vec(ema.gamma[l], p.gamma[l], decay);
    ema_update_vec(ema.beta[l], p.beta[l], decay);
    ema_update_vec(ema.running_mean[l], p.running_mean[l], decay);
    ema_update_vec(ema.running_var[l], p.running_var[l], decay);
    ema_update_mat(ema.residual_W[l], p.residual_W[l], decay);
    ema_update_vec(ema.residual_B[l], p.residual_B[l], decay);
    ema_update_mat(ema.gate_W[l], p.gate_W[l], decay);
    ema_update_vec(ema.gate_B[l], p.gate_B[l], decay);
  }
  ema_update_vec(ema.interaction_W, p.interaction_W, decay);
  ema_update_vec(ema.interaction_B, p.interaction_B, decay);
  ema_update_mat(ema.proj_W, p.proj_W, decay);
  ema_update_vec(ema.proj_B, p.proj_B, decay);
}

static List params_to_list(const MLPParams &p) {
  List weights(p.W.size()), biases(p.B.size()), gamma(p.gamma.size()), beta(p.beta.size()),
       running_mean(p.running_mean.size()), running_var(p.running_var.size()),
       residual_weights(p.residual_W.size()), residual_biases(p.residual_B.size()),
       gate_weights(p.gate_W.size()), gate_biases(p.gate_B.size());
  for (size_t l = 0; l < p.W.size(); ++l) { weights[l] = p.W[l]; biases[l] = p.B[l]; }
  for (size_t l = 0; l < p.gamma.size(); ++l) {
    gamma[l] = p.gamma[l]; beta[l] = p.beta[l];
    running_mean[l] = p.running_mean[l]; running_var[l] = p.running_var[l];
    residual_weights[l] = p.residual_W[l]; residual_biases[l] = p.residual_B[l];
    gate_weights[l] = p.gate_W[l]; gate_biases[l] = p.gate_B[l];
  }
  return List::create(
    Named("weights") = weights, Named("biases") = biases,
    Named("gamma") = gamma, Named("beta") = beta,
    Named("running_mean") = running_mean, Named("running_var") = running_var,
    Named("residual_weights") = residual_weights, Named("residual_biases") = residual_biases,
    Named("gate_weights") = gate_weights, Named("gate_biases") = gate_biases,
    Named("interaction_weights") = p.interaction_W,
    Named("interaction_biases") = p.interaction_B,
    Named("proj_weights") = p.proj_W,
    Named("proj_biases") = p.proj_B
  );
}

static double epoch_learning_rate(double lr, int epoch, int epochs,
                                  const std::string &schedule) {
  if (schedule == "cosine") {
    return lr * 0.5 * (1.0 + std::cos(arma::datum::pi * epoch / std::max(2, epochs)));
  }
  if (schedule == "step") {
    int step_size = std::max(5, epochs / 3);
    return lr * std::pow(0.5, epoch / step_size);
  }
  return lr;
}

// [[Rcpp::export]]
List densemlp_train_cpp(const arma::mat &X, const arma::mat &Y,
                        Rcpp::IntegerVector hidden_units, int task, int output_dim,
                        int epochs, int batch_size, double lr,
                        double validation, bool early_stopping,
                        int patience, double min_delta, int min_epochs,
                        bool residual, bool gated, Rcpp::NumericVector dropout,
                        bool interaction, bool batch_norm, int input_projection,
                        double ema_decay, std::string lr_schedule,
                        unsigned int seed, bool verbose,
                        int survival_loss, Rcpp::NumericVector breaks_in, Rcpp::NumericVector ghat_grid_in) {
  vec breaks = as<vec>(breaks_in);
  vec ghat_grid = as<vec>(ghat_grid_in);
  bool use_proj = input_projection > 0;
  uword n = X.n_rows, p_in = X.n_cols, p_out = (uword)output_dim;
  uword net_in = use_proj ? (uword)input_projection : p_in;
  std::vector<uword> sizes;
  sizes.push_back(net_in);
  for (int i = 0; i < hidden_units.size(); ++i) sizes.push_back((uword)hidden_units[i]);
  sizes.push_back(p_out);

  vec dropout_vec = as<vec>(dropout);
  MLPParams params = init_params(sizes, seed, residual, gated, interaction,
                                 use_proj ? p_in : (uword)0);
  MLPParams ema_params = params;
  AdamState adam = init_adam(params);

  arma::arma_rng::set_seed(seed + 1u);

  uword valid_n = 0;
  uvec train_idx, valid_idx;
  bool has_validation = validation > 0.0;
  if (has_validation) {
    valid_n = std::max((uword)1, (uword)std::floor((double)n * validation));
    if (valid_n >= n) valid_n = std::max((uword)1, n - 1);
    uvec perm0 = arma::conv_to<uvec>::from(arma::shuffle(arma::regspace(0, n - 1)));
    valid_idx = perm0.subvec(0, valid_n - 1);
    train_idx = perm0.subvec(valid_n, n - 1);
  } else {
    train_idx = arma::conv_to<uvec>::from(arma::regspace(0, n - 1));
  }
  mat X_train = X.rows(train_idx), Y_train = Y.rows(train_idx);
  uword n_train = X_train.n_rows;

  NumericVector train_history(epochs), valid_history(epochs), lr_history(epochs);
  double best_loss = arma::datum::inf;
  int best_epoch = 0, wait = 0;
  MLPParams best_params = params;
  bool have_best = false;
  int epochs_run = 0;

  for (int epoch = 0; epoch < epochs; ++epoch) {
    epochs_run = epoch + 1;
    double current_lr = epoch_learning_rate(lr, epoch, epochs, lr_schedule);
    lr_history[epoch] = current_lr;
    uvec perm = arma::conv_to<uvec>::from(arma::shuffle(arma::regspace(0, n_train - 1)));
    double epoch_loss = 0.0;
    uword n_batches = 0;

    for (uword start = 0; start < n_train; start += batch_size) {
      uword end = std::min(start + (uword)batch_size, n_train) - 1;
      uvec idx = perm.subvec(start, end);
      mat Xb = X_train.rows(idx);
      mat Yb = Y_train.rows(idx);

      ForwardCache cache;
      forward_train(params, Xb, task, residual, gated, dropout_vec, interaction,
                    batch_norm, use_proj, cache);
      mat yhat = cache.A.back();
      double nb = (double)Xb.n_rows;
      if (task == 0) {
        epoch_loss += arma::accu(arma::square(yhat - Yb)) / nb;
      } else if (task == 3 && survival_loss == 1) {
        mat grad_unused;
        epoch_loss += brier_loss_and_grad(yhat, Yb, breaks, ghat_grid, grad_unused);
      } else if (task == 3) {
        vec grad_unused;
        epoch_loss += cox_loss_and_grad(yhat.col(0), Yb.col(0), Yb.col(1), grad_unused);
      } else {
        mat clipped = arma::clamp(yhat, 1e-9, 1.0 - 1e-9);
        epoch_loss += -arma::accu(Yb % arma::log(clipped)) / nb;
      }
      n_batches++;
      backward_and_update(params, adam, cache, Yb, current_lr, task, residual, gated,
                          dropout_vec, interaction, batch_norm, use_proj,
                          survival_loss, breaks, ghat_grid);
      if (ema_decay > 0.0) update_ema(ema_params, params, ema_decay);
    }

    train_history[epoch] = epoch_loss / (double)n_batches;

    if (has_validation) {
      const MLPParams &eval_params = ema_decay > 0.0 ? ema_params : params;
      double vloss = eval_loss(eval_params, X.rows(valid_idx), Y.rows(valid_idx), task,
                               residual, gated, interaction, batch_norm, use_proj,
                               survival_loss, breaks, ghat_grid);
      valid_history[epoch] = vloss;
      if (vloss < best_loss - min_delta) {
        best_loss = vloss; best_epoch = epoch + 1; wait = 0;
        deep_copy(best_params, eval_params);
        have_best = true;
      } else {
        wait++;
      }
      if (verbose && (epoch % 10 == 0)) {
        Rcpp::Rcout << "epoch " << epoch << " train " << train_history[epoch]
                    << " valid " << vloss << std::endl;
      }
      if (early_stopping && (epoch + 1) >= min_epochs && wait >= patience) break;
    } else {
      valid_history[epoch] = NA_REAL;
      if (verbose && (epoch % 10 == 0)) {
        Rcpp::Rcout << "epoch " << epoch << " train " << train_history[epoch] << std::endl;
      }
    }
  }

  MLPParams &final_params = (has_validation && have_best) ? best_params :
    (ema_decay > 0.0 ? ema_params : params);

  List out = params_to_list(final_params);
  out["train_history"] = train_history[Rcpp::Range(0, epochs_run - 1)];
  out["valid_history"] = valid_history[Rcpp::Range(0, epochs_run - 1)];
  out["lr_history"] = lr_history[Rcpp::Range(0, epochs_run - 1)];
  out["best_epoch"] = (has_validation && have_best) ? best_epoch : epochs_run;
  out["epochs_run"] = epochs_run;
  return out;
}

// [[Rcpp::export]]
arma::mat densemlp_predict_cpp(const arma::mat &X, List weights, List biases,
                               List gamma, List beta, List running_mean, List running_var,
                               List residual_weights, List residual_biases,
                               List gate_weights, List gate_biases,
                               arma::vec interaction_weights, arma::vec interaction_biases,
                               arma::mat proj_weights, arma::vec proj_biases,
                               int task, bool residual, bool gated, bool interaction,
                               bool batch_norm, int input_projection) {
  MLPParams p;
  for (int l = 0; l < weights.size(); ++l) {
    p.W.push_back(as<mat>(weights[l]));
    p.B.push_back(as<vec>(biases[l]));
  }
  for (int l = 0; l < gamma.size(); ++l) {
    p.gamma.push_back(as<vec>(gamma[l]));
    p.beta.push_back(as<vec>(beta[l]));
    p.running_mean.push_back(as<vec>(running_mean[l]));
    p.running_var.push_back(as<vec>(running_var[l]));
    p.residual_W.push_back(as<mat>(residual_weights[l]));
    p.residual_B.push_back(as<vec>(residual_biases[l]));
    p.gate_W.push_back(as<mat>(gate_weights[l]));
    p.gate_B.push_back(as<vec>(gate_biases[l]));
  }
  p.interaction_W = interaction_weights;
  p.interaction_B = interaction_biases;
  p.proj_W = proj_weights;
  p.proj_B = proj_biases;
  return forward_eval(p, X, task, residual, gated, interaction, batch_norm,
                      input_projection > 0);
}
