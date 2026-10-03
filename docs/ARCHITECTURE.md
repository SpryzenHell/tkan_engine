# Architecture

## Temporal KAN

The KAN edge function is the efficient B-spline formulation from efficient-kan: basis values are evaluated per input feature, flattened and projected with a single F.linear operation rather than materializing a batch x output x input activation tensor.

TemporalKANBlock adds a causal depthwise temporal convolution before the KAN projection and a residual gated output.

The grid updater uses empirical sample locations blended with a uniform component. After moving knots, spline coefficients are re-fitted by least squares so the learned edge function is approximately preserved.

## Microstructure representation

The feature extractor derives:
- bid/ask spread in basis points,
- level-1 queue imbalance,
- microprice displacement,
- signed trade size,
- level-wise depth imbalance,
- midpoint return.

The sequence model consumes rolling [batch, time, feature] windows and predicts the next midpoint return.

## Data path

The production shape is:
feed or dataset reader -> Arrow RecordBatch -> q-like filter/project/group operations -> feature extraction -> rolling windows -> Torch tensor.

The Arrow implementation intentionally keeps the event-count loop outside the memory boundary. A 300M-event run is many bounded batches, not one giant table.

## Distributed training

DDP uses one process per rank and a DistributedSampler. CUDA uses NCCL; CPU correctness tests use Gloo.

The spline grid is non-gradient state. Every adaptive update is performed on the local batch and rank 0 broadcasts the updated grid and spline weights to the other ranks.

A production implementation can replace this rank-0 state update with distributed quantile/statistic aggregation to make grid adaptation itself fully data-parallel.
