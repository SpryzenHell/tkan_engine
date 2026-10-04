# Architecture

## Event and feature path

The synthetic L2 generator emits ten price levels per event, best bid/ask prices and sizes, trade side and trade size. The pandas feature extractor turns these observations into a compact event representation.

The main derived quantities are:

- spread in basis points
- level-1 queue imbalance
- microprice displacement in basis points
- signed trade size
- level-wise depth imbalance for levels 1–10
- midpoint return

Rolling windows are converted to tensors with shape `[batch, time, feature]`.

## Temporal KAN

The model first projects raw features into a hidden representation. Each TemporalKAN block applies:

```
LayerNorm
   |
causal depthwise temporal convolution
   |
B-spline KAN edge projection
   |
SiLU + linear mixing
   |
residual gated update
```

The B-spline edge implementation follows the efficient flatten-and-project pattern: basis values are evaluated for each input feature and the flattened basis matrix is passed through a single `F.linear`.

Adaptive grid updates use observed sample locations blended with a uniform component. After the knot locations are moved, spline coefficients are refit so that the learned edge function is approximately preserved under the new grid.

## Arrow path

The Arrow benchmark is intentionally batch bounded:

```
Parquet / synthetic generator
        |
    RecordBatch
        |
raw L2 derivation
        |
q-like where / select / within
        |
checksum / metrics
```

The 300M-event run is therefore a streaming stress test. It exercises 300 bounded RecordBatches rather than constructing one 300M-row object.

## DDP

Each process owns a shard of the sequence dataset through `DistributedSampler`. CUDA uses NCCL and CPU smoke tests use Gloo.

Gradient state is handled by DDP. The adaptive spline grid is not gradient state, so the updated grid and spline coefficients are broadcast from rank 0 after an adaptive update.

