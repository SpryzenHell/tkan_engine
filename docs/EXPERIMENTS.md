# Experiments

## Controlled nonlinear inversion

Generator:
0.9 sin(2.6 OFI) + 0.55 OFI cubed + 0.25 dOFI + N(0, 0.08 squared)

### Verified GitHub CI run

3,000 events are converted to 16-step sequences. The split is chronological: 80 percent train, 20 percent test. The run executed from the PR branch on GitHub Actions.

| Model | Parameters | MSE (bps squared) | RMSE (bps) | IC |
|---|---:|---:|---:|---:|
| MLP | 801 | 0.75995 | 0.87175 | 0.75018 |
| T-KAN | 3266 | 0.28884 | 0.53744 | 0.86401 |

Source: results/ci_inversion_3000.json.

The purpose of this experiment is to test nonlinear representation and temporal modelling. It is deliberately not presented as a trading backtest.

## LOB prediction smoke test

The current local 12,000-event smoke run achieved:
- test MSE: 5.2227 bps squared,
- test RMSE: 2.2853 bps,
- test IC: 0.00215.

This is a baseline engineering smoke test on the synthetic generator, not evidence of alpha.

## Performance

Current local CPU model benchmark:
- batch: 8
- sequence length: 16
- features: 18
- steps: 20
- throughput: 4,818.03 samples/s
- batch latency: 1.660 ms
- parameters: 7,715

Current local DDP correctness test:
- ranks: 2
- backend: Gloo
- world size: 2
- one epoch completed successfully.

Arrow and GPU numbers are intentionally absent from the local evidence because those dependencies/hardware were unavailable.