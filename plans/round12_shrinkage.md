# Round-12 pre-registration — covariance shrinkage (denoise, don't discard)

Registered 2026-07-17, before the run. Follow-up to round-11.

## Question

Round-11: composition kernel (clean, field-blind) lost to sim dollar-space
covariance (noisy, field-aware) by ~1.5pp mean_pct. The sim matrix is ~2/3
estimation noise pairwise (split-half r=0.34 @4k sims). Does shrinking the
dollar-space correlation toward the kernel target recover the noise cost?

    C(lam) = lam * kernel_corr + (1-lam) * dollar_corr,  lam in {0.25, 0.5, 0.75}

Endpoints already measured: lam=0 = incumbent (archived portfolios),
lam=1 = round-11 kernel arm.

## Implementation notes (documented approximations)

- dollar_corr rebuilt from cached sims via within-pool payout-proxy ranks
  (pool lineup scores -> per-sim rank within pool -> reference payout curve),
  25k sims (Spearman-Brown reliability ~0.76 vs the incumbent's own basis).
  This is NOT the incumbent's exact matrix (no simulated opponent fields /
  dupes) — the lam=0 proxy arm is therefore also run, so the sweep is
  internally consistent: judge lam in {0(proxy),0.25,0.5,0.75,1} on the
  same construction, with the archived incumbent as external reference.
- Same pools/EV/params as round-11; risk-1 150-blocks; graded on real fields.

## Judgment, pre-committed

- If any intermediate lam beats BOTH endpoints on pooled mean_pct AND cash
  (paired, 33 slates): denoising is real -> candidate for a live selector
  change (would then need implementation against the true in-pipeline
  matrix, not the proxy, before any config change).
- If lam=0(proxy) is best: noise is harmless at selection; close the
  covariance question permanently.
- Secondary: ownership order-gradient per lam (does the chalky-first
  property fade continuously as lam -> 1?).
