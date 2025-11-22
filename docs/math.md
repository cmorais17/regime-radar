# Mathematical Model

This document describes the statistical and algorithmic model used by `regime-radar` to detect regime shifts in a univariate financial return series.
We focus on daily log-returns of a single asset, but the framework applies to any 1D time series where a piecewise-Gaussian model is reasonable.

## 1. Problem setup

Let $r_0, r_1, \dots, r_{n-1}$ denote the (log-)returns of a single asset over $n$ trading days. We assume that the time axis is partitioned into $K$ regimes:

$$
\{ 0 = \tau_0 < \tau_1 < \dots < \tau_{K-1} < \tau_K = n \},
$$

such that regime $k$ covers indices

$$
\{ \tau_{k-1}, \tau_{k-1} + 1, \dots, \tau_k - 1 \}.
$$

Within each regime $k$, we assume a Gaussian model

$$
r_t \sim \mathcal{N}(\mu_k, \sigma_k^2), \quad \tau_{k-1} \le t < \tau_k,
$$

with regime-specific parameters:

- $\mu_k$: mean return within regime $k$  
- $\sigma_k^2$: variance within regime $k$

The goal is to infer:

1. The number of regimes $K$ (not fixed in advance)  
2. The breakpoints $\tau_1, \dots, \tau_{K-1}$  
3. The regime parameters $(\mu_k, \sigma_k^2)$

subject to a trade-off between goodness of fit and model complexity.

## 2. Gaussian segment likelihood

Consider a single candidate segment $[s, t)$ with length

$$
m = t - s, \quad 0 \le s < t \le n.
$$

The observations in this segment are $r_s, \dots, r_{t-1}$.

### 2.1. Maximum likelihood estimates

Under a Gaussian model

$$
r_i \sim \mathcal{N}(\mu, \sigma^2), \quad s \le i < t,
$$

the log-likelihood of this segment is

$$
\ell(\mu, \sigma^2) = -\frac{m}{2} \log(2\pi) - \frac{m}{2} \log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=s}^{t-1}(r_i - \mu)^2.
$$

The maximum likelihood estimates (MLEs) for this segment are:

$$
\hat{\mu}
= \frac{1}{m}\sum_{i=s}^{t-1} r_i,
\qquad
\hat{\sigma}^2
= \frac{1}{m}\sum_{i=s}^{t-1} (r_i - \hat{\mu})^2.
$$

Define the segment sum of squared errors (SSE)

$$
\text{SSE}(s, t)
= \sum_{i=s}^{t-1} (r_i - \hat{\mu})^2.
$$

Then $\hat{\sigma}^2 = \text{SSE}(s, t) / m$.

### 2.2. Log-likelihood at the optimum

Plugging $(\hat{\mu}, \hat{\sigma}^2)$ back into the log-likelihood,

$$
\ell(\hat{\mu}, \hat{\sigma}^2)
= -\frac{m}{2} \log(2\pi)
  - \frac{m}{2} \log\left( \frac{\text{SSE}(s, t)}{m} \right)
  - \frac{1}{2} m.
$$

Equivalently, up to additive constants (that do not depend on the segmentation),

$$
-2\,\ell(\hat{\mu}, \hat{\sigma}^2)
= m \log\left( \frac{\text{SSE}(s, t)}{m} \right) + \text{const}.
$$

This motivates using

$$
C_{\text{mean-var}}(s, t)
= m \log\left( \frac{\text{SSE}(s, t)}{m} \right)
$$

as a segment cost function for a model where both mean and variance are allowed to change between regimes.

### 2.3. Mean-only model

If we assume a single common variance $\sigma^2$ shared across all regimes and only allow the mean to change, then (ignoring constants) the negative log-likelihood is proportional to the **sum of squared errors**:

$$
C_{\text{mean}}(s, t)
= \text{SSE}(s, t).
$$

This corresponds exactly to the classic **piecewise-constant mean** model.

## 3. Segment cost functions used in `regime-radar`

In `regime-radar`, we implement two model modes:

1. `model = "mean"`:  
   - Piecewise-constant mean  
   - Shared variance (up to scaling)  
   - Segment cost:

$$
C(s, t) = \text{SSE}(s, t).
$$

2. `model = "mean-var"` (default):  
   - Piecewise-constant mean **and** variance  
   - Segment cost:

$$
C(s, t) = m \log\left( \frac{\text{SSE}(s, t)}{m} \right),
\quad m = t - s.
$$

The `"mean-var"` mode aligns naturally with the Gaussian log-likelihood when both $\mu_k$ and $\sigma_k^2$ are re-estimated in each segment. In financial time series, this model is often more realistic, because volatility regimes change more dramatically than the mean return itself.

### 3.1. Efficient computation of SSE

To compute $\text{SSE}(s, t)$ for many candidate segments efficiently, we use prefix sums.
Define:

$$
S_1[t] = \sum_{i=0}^{t-1} r_i,
\qquad
S_2[t] = \sum_{i=0}^{t-1} r_i^2,
\qquad 0 \le t \le n.
$$

Then for a segment $[s, t)$, we have:

$$
\sum_{i=s}^{t-1} r_i = S_1[t] - S_1[s],
\qquad
\sum_{i=s}^{t-1} r_i^2 = S_2[t] - S_2[s].
$$

Let $m = t - s$ and $\bar{r}_{s:t} = \frac{1}{m}\sum_{i=s}^{t-1} r_i$. Then

$$
\text{SSE}(s, t)
= \sum_{i=s}^{t-1} (r_i - \bar{r}_{s:t})^2
= \sum_{i=s}^{t-1} r_i^2 - m \bar{r}_{s:t}^2
= (S_2[t] - S_2[s]) - \frac{(S_1[t] - S_1[s])^2}{m}.
$$

Thus $\text{SSE}(s, t)$ can be computed in $\mathcal{O}(1)$ time per candidate segment, after a single $\mathcal{O}(n)$ pass to build prefix sums.

## 4. Penalized segmentation via dynamic programming

We now describe how the optimal segmentation is computed.

Let a segmentation be specified by breakpoints $0 = \tau_0 < \tau_1 < \dots < \tau_K = n$. The total cost of a segmentation under a chosen segment cost $C(\cdot,\cdot)$ and a penalty $\lambda > 0$ per regime is

$$
\text{TotalCost}
= \sum_{k=1}^K \left[ C(\tau_{k-1}, \tau_k) + \lambda \right].
$$

The goal is to find the $K$ and breakpoints that minimize this total cost. The penalty $\lambda$ controls the trade-off between:

- data fit (through $C$), and  
- model complexity (through the number of regimes $K$).

### 4.1. Dynamic programming recurrence

Define:

- $F(t)$ = minimal cost for segmenting the prefix $[0, t)$  
- With convention $F(0) = 0$.

If the last segment starts at $s$ and ends at $t$, then the recurrence is

$$
F(t) = \min_{0 \le s \le t - m_{\min}}
\left\{ F(s) + C(s, t) + \lambda \right\},
\quad t \ge m_{\min},
$$

where $m_{\min}$ is a minimum allowed segment length (to avoid ultra-short regimes that overfit noise).

We also record, for each $t$, the best starting index $s^\ast(t)$ that attains this minimum:

$$
s^\ast(t)
= \arg\min_{0 \le s \le t - m_{\min}}
\left\{ F(s) + C(s, t) + \lambda \right\}.
$$

The overall optimal cost is $F(n)$, and the corresponding breakpoints are reconstructed by backtracking:

1. Start from $t = n$  
2. Let $s = s^\ast(t)$ be the start of the last segment  
3. Record segment $[s, t)$  
4. Set $t \leftarrow s$ and repeat until $t = 0$

This dynamic programming algorithm runs in $\mathcal{O}(n^2)$ time in the basic implementation used by `regime-radar`. The use of prefix sums ensures that evaluating $C(s, t)$ for any $(s, t)$ pair is $\mathcal{O}(1)$, so the dominant cost is the double loop over $t$ and $s$.

## 5. Penalty choices: AIC, BIC, and custom $\lambda$

The penalty parameter $\lambda$ directly affects the number of detected regimes:

- Larger $\lambda$ $\Rightarrow$ fewer regimes (more conservative)  
- Smaller $\lambda$ $\Rightarrow$ more regimes (more sensitive)

`regime-radar` supports three penalty specifications:

1. `$--penalty bic`  
2. `$--penalty aic`  
3. `$--penalty <float>` (user-specified positive scalar)

### 5.1. BIC-style penalty

For BIC, we set

$$
\lambda_{\text{BIC}} = \log n,
$$

where $n$ is the number of observations. Intuitively, this grows with the sample size and tends to favor simpler models as $n$ increases.

### 5.2. AIC-style penalty

For AIC, we use a constant penalty:

$$
\lambda_{\text{AIC}} = 2.
$$

This is typically less conservative than BIC and often yields more regimes in practice.

### 5.3. Custom penalty

The user can also provide a custom positive float:

$$
\lambda = \text{user value} > 0.
$$

This allows manual tuning of sensitivity to regime changes. In practice, scanning across a grid of $\lambda$ values and comparing the resulting segmentations can be informative.

## 6. Interpretation for financial time series

When applied to daily log-returns $r_t$ of an asset, the `"mean-var"` model detects time intervals where the joint behavior of:

- average return $\mu_k$, and  
- volatility $\sigma_k$

is approximately stable, separated by statistically significant changes.

Typical behavior observed in equity or index returns:

- The mean daily return tends to be small and relatively stable over long periods.  
- Volatility, by contrast, exhibits distinct regimes (e.g., calm vs. crisis periods).

By allowing both mean and variance to change, the `"mean-var"` model can capture volatility regime shifts that a pure mean-shift model would miss. Empirically, this often leads to:

- Few or no segments under `"mean"` with a strong penalty (e.g., BIC)  
- Several interpretable regimes under `"mean-var"` corresponding to periods of high and low volatility (and sometimes shifts in average return)

The resulting regimes can be used to:

- Annotate historical periods (e.g., crises, rallies, low-volatility plateaus)  
- Compute regime-specific statistics (annualized return, volatility, Sharpe-like metrics)  
- Serve as features or labels for downstream models in quantitative research

## 7. Limitations and extensions

The current model makes several simplifying assumptions:

1. **Gaussianity**:  
   Returns are modeled as Gaussian within each regime, which ignores heavy tails and skewness typical of financial returns.

2. **Independence within regimes**:  
   The model assumes i.i.d. observations within each regime, omitting autocorrelation and more complex dynamics (e.g. GARCH effects).

3. **Univariate series**:  
   Only a single asset (or single derived series) is modeled at a time. Multivariate extensions would allow joint regime detection across assets or factors.

4. **$\mathcal{O}(n^2)$ complexity**:  
   While fine for daily data, the naive dynamic programming scaling can become expensive for very long or high-frequency series.