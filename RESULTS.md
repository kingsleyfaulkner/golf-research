# Results

## 1x GB10

| Experiment | Date | Steps | Loss | Val Loss | Val BPB | nvfp4 BPB | mxfp4 BPB | int6 BPB | int8 BPB | Artifacts |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 008-lower-lr | 2026-03-25 16:08 | 678 | 2.5614 | 2.5768 | 1.5261 | 🟢 1.5623 (9.2M) | 🟢 1.5707 (8.6M) | 🟢 1.5448 (8.3M) | 🟢 1.5278 (12.1M) | [artifacts](experiments/008-lower-lr/artifacts_1x_gb10) |
| baseline | 2026-03-25 13:39 | 673 | 2.6172 | 2.5829 | 1.5297 | 🟢 1.6081 (9.2M) | 🟢 1.6281 (8.6M) | 🟢 1.5433 (9.8M) | 🟢 1.5305 (13.8M) | [artifacts](experiments/baseline/artifacts_1x_gb10) |
| 003-grad-clip | 2026-03-25 14:52 | 675 | 2.5856 | 2.5843 | 1.5306 | 🟢 1.6050 (9.2M) | 🟢 1.6248 (8.6M) | 🟢 1.5466 (9.7M) | 🟢 1.5319 (13.9M) | [artifacts](experiments/003-grad-clip/artifacts_1x_gb10) |
| 002-longer-warmdown | 2026-03-25 14:28 | 677 | 2.5939 | 2.5872 | 1.5323 | 🟢 1.6147 (9.2M) | 🟢 1.6451 (8.6M) | 🟢 1.5486 (10.3M) | 🟢 1.5328 (14.0M) | [artifacts](experiments/002-longer-warmdown/artifacts_1x_gb10) |
| 005-mlp-3x | 2026-03-25 15:43 | 618 | 2.5710 | 2.5913 | 1.5347 | 🟢 1.5915 (11.7M) | 🟢 1.6020 (10.9M) | 🟢 1.5469 (11.9M) | 🔴 1.5355 (17.1M) | [artifacts](experiments/005-mlp-3x/artifacts_1x_gb10) |
| 015-embed-lr-doubled | 2026-03-25 17:01 | 677 | 2.6144 | 2.6135 | 1.5479 | 🟢 1.6702 (9.2M) | 🟢 1.7358 (8.6M) | 🟢 1.5849 (10.1M) | 🟢 1.5501 (13.8M) | [artifacts](experiments/015-embed-lr-doubled/artifacts_1x_gb10) |
| 004-momentum-warmup | 2026-03-25 15:16 | 674 | 2.6224 | 2.6164 | 1.5496 | 🟢 1.6268 (9.2M) | 🟢 1.6686 (8.6M) | 🟢 1.5651 (10.0M) | 🟢 1.5506 (14.4M) | [artifacts](experiments/004-momentum-warmup/artifacts_1x_gb10) |
| 011-layers-11 | 2026-03-25 16:36 | 560 | 2.6489 | 2.6374 | 1.5620 | 🟢 1.6188 (11.2M) | 🟢 1.6457 (10.4M) | 🟢 1.5757 (11.6M) | 🔴 1.5627 (16.3M) | [artifacts](experiments/011-layers-11/artifacts_1x_gb10) |
| 001-weight-decay | 2026-03-25 14:03 | 675 | 6.0388 | 6.0327 | 3.5729 | 🟢 3.5736 (9.2M) | 🟢 3.5951 (8.5M) | 🟢 nan (4.1M) | 🟢 3.5737 (8.3M) | [artifacts](experiments/001-weight-decay/artifacts_1x_gb10) |

## 1x RTX PRO 6000

| Experiment | Date | Steps | Loss | Val Loss | Val BPB | nvfp4 BPB | mxfp4 BPB | int6 BPB | int8 BPB | Artifacts |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| baseline | 2026-03-24 23:00 | 320 | 2.6495 | 2.6324 | 1.5591 | 🟢 1.6033 (9.2M) | 🟢 1.6192 (8.6M) | 🟢 1.5756 (8.2M) | 🟢 1.5596 (12.4M) | [artifacts](experiments/baseline/artifacts_1x_rtx_pro_6000) |

## 2x RTX PRO 6000

| Experiment | Date | Steps | Loss | Val Loss | Val BPB | nvfp4 BPB | mxfp4 BPB | int6 BPB | int8 BPB | Artifacts |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| baseline | 2026-03-24 21:31 | 610 | 2.4345 | 2.4411 | 1.4458 | 🟢 1.4964 (9.2M) | 🟢 1.5158 (8.6M) | 🟢 1.4598 (9.6M) | 🟢 1.4466 (13.7M) | [artifacts](experiments/baseline/artifacts_2x_rtx_pro_6000) |

## 8x H100

| Experiment | Date | Steps | Loss | Val Loss | Val BPB | nvfp4 BPB | mxfp4 BPB | int6 BPB | int8 BPB | Artifacts |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| baseline | 2026-03-24 20:01 | 2375 | 2.1258 | 2.1406 | 1.2678 | 🟢 1.4307 (9.2M) | 🟢 1.5429 (8.6M) | 🟢 1.2992 (11.2M) | 🟢 1.2695 (15.6M) | [artifacts](experiments/baseline/artifacts_8x_h100_2) |

## 8x RTX PRO 6000

| Experiment | Date | Steps | Loss | Val Loss | Val BPB | nvfp4 BPB | mxfp4 BPB | int6 BPB | int8 BPB | Artifacts |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 005-mlp-3x | 2026-03-24 14:04 | 3640 | 2.1994 | 2.1299 | 1.2614 | 🟢 1.4770 (11.7M) | 🟢 1.5457 (11.0M) | 🟢 1.3111 (14.4M) | 🔴 1.2643 (19.9M) | [artifacts](experiments/005-mlp-3x/artifacts_8x_rtx_pro_6000) |
| 004-momentum-warmup | 2026-03-24 15:33 | 4292 | 2.1632 | 2.1481 | 1.2722 | 🟢 1.9462 (9.2M) | 🟢 2.3571 (8.7M) | 🟢 1.3825 (11.5M) | 🟢 1.2794 (15.7M) | [artifacts](experiments/004-momentum-warmup/artifacts_8x_rtx_pro_6000) |
| 003-grad-clip | 2026-03-24 17:03 | 4301 | 2.1749 | 2.1535 | 1.2754 | 🟢 1.5664 (9.2M) | 🟢 1.6938 (8.6M) | 🟢 1.3444 (11.6M) | 🟢 1.2805 (15.7M) | [artifacts](experiments/003-grad-clip/artifacts_8x_rtx_pro_6000) |
| baseline | 2026-03-24 18:32 | 4147 | 2.1428 | 2.1551 | 1.2764 | 🟢 1.6003 (9.2M) | 🟢 1.8141 (8.7M) | 🟢 1.3580 (11.4M) | 🟢 1.2806 (15.7M) | [artifacts](experiments/baseline/artifacts_8x_rtx_pro_6000_4) |

## tmp

| Experiment | Date | Steps | Loss | Val Loss | Val BPB | nvfp4 BPB | mxfp4 BPB | int6 BPB | int8 BPB | Artifacts |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| baseline | 2026-03-25 13:39 | - | - | - | - | - | - | - | - | [artifacts](experiments/baseline/artifacts_tmp) |
