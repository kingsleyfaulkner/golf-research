# Sweep: Depth vs Width

Explores the tradeoff between model depth (more layers) and width (larger MLP) for a fixed architecture budget. Uses the baseline encoder-decoder architecture with varying `num_layers` and `mlp_mult`.

## Sweep Properties

| Property | Values |
|:---|:---|
| num_layers | 15, 20 |
| mlp_mult | 2, 3, 4, 5 |

## Filter

Excludes combinations where `mlp_mult > 3` and `num_layers >= 20` to avoid exceeding the compressed size budget.

## Conditional Overrides

When `num_layers >= 20`, `batch_size` is reduced to 32 to avoid OOM on 8xH100.

## Variants

| # | Name | Layers | MLP mult | Est. params |
|:---|:---|---:|---:|---:|
| 1 | mlpmult2-layers15 | 15 | 2 | 28.1M |
| 2 | mlpmult3-layers15 | 15 | 3 | 36.0M |
| 3 | mlpmult4-layers15 | 15 | 4 | 43.8M |
| 4 | mlpmult5-layers15 | 15 | 5 | 51.7M |
| 5 | mlpmult2-layers20 | 20 | 2 | 37.3M |
| 6 | mlpmult3-layers20 | 20 | 3 | 47.8M |

## Hypothesis

Deeper models have more capacity but need more training tokens to converge. Wider MLPs are more compute-efficient per step. On wallclock-limited training, wider may outperform deeper until sufficient GPU throughput is available.
