# Results

## 1x GB10

| Experiment | Date | Steps | Tokens | Loss | Val Loss | Val BPB | int4 BPB | int6 BPB | int8 BPB | mxfp4 BPB | nvfp4 BPB | turboquip4c BPB | turboquip4cr BPB | Artifacts |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 008-lower-lr | 2026-03-25 16:08 | 678 | 88.9M | 2.5614 | 2.5768 | 1.5261 | - | 🟢 1.5448 (8.3M) | 🟢 1.5278 (12.1M) | 🟢 1.5707 (8.6M) | 🟢 1.5623 (9.2M) | - | - | [artifacts](008-lower-lr/artifacts_1x_gb10) |
| baseline | 2026-03-27 12:28 | 673 | 88.2M | 2.6172 | 2.5829 | 1.5297 | - | 🟢 1.5433 (9.8M) | 🟢 1.5302 (13.8M) | 🟢 1.6281 (8.6M) | 🟢 1.6081 (9.2M) | - | - | [artifacts](baseline/artifacts_1x_gb10) |
| 003-grad-clip | 2026-03-25 14:52 | 675 | 88.5M | 2.5856 | 2.5843 | 1.5306 | - | 🟢 1.5466 (9.7M) | 🟢 1.5319 (13.9M) | 🟢 1.6248 (8.6M) | 🟢 1.6050 (9.2M) | - | - | [artifacts](003-grad-clip/artifacts_1x_gb10) |
| 002-longer-warmdown | 2026-03-25 14:28 | 677 | 88.7M | 2.5939 | 2.5872 | 1.5323 | - | 🟢 1.5486 (10.3M) | 🟢 1.5328 (14.0M) | 🟢 1.6451 (8.6M) | 🟢 1.6147 (9.2M) | - | - | [artifacts](002-longer-warmdown/artifacts_1x_gb10) |
| 005-mlp-3x | 2026-03-25 15:43 | 618 | 81.0M | 2.5710 | 2.5913 | 1.5347 | - | 🟢 1.5469 (11.9M) | 🔴 1.5355 (17.1M) | 🟢 1.6020 (10.9M) | 🟢 1.5915 (11.7M) | - | - | [artifacts](005-mlp-3x/artifacts_1x_gb10) |
| baseline | 2026-03-28 21:21 | 829 | 108.7M | 2.4895 | 2.5913 | 1.5347 | 🟢 1.9033 (5.5M) | 🟢 1.5494 (9.9M) | 🟢 1.5522 (10.1M) | 🟢 1.6563 (8.5M) | 🟢 1.6697 (8.7M) | 🟢 1.5765 (5.8M) | 🟢 1.5729 (8.2M) | [artifacts](baseline/artifacts_1x_gb10_2) |
| 015-embed-lr-doubled | 2026-03-25 17:01 | 677 | 88.7M | 2.6144 | 2.6135 | 1.5479 | - | 🟢 1.5849 (10.1M) | 🟢 1.5501 (13.8M) | 🟢 1.7358 (8.6M) | 🟢 1.6702 (9.2M) | - | - | [artifacts](015-embed-lr-doubled/artifacts_1x_gb10) |
| 004-momentum-warmup | 2026-03-25 15:16 | 674 | 88.3M | 2.6224 | 2.6164 | 1.5496 | - | 🟢 1.5651 (10.0M) | 🟢 1.5506 (14.4M) | 🟢 1.6686 (8.6M) | 🟢 1.6268 (9.2M) | - | - | [artifacts](004-momentum-warmup/artifacts_1x_gb10) |
| 011-layers-11 | 2026-03-25 16:36 | 560 | 73.4M | 2.6489 | 2.6374 | 1.5620 | - | 🟢 1.5757 (11.6M) | 🔴 1.5627 (16.3M) | 🟢 1.6457 (10.4M) | 🟢 1.6188 (11.2M) | - | - | [artifacts](011-layers-11/artifacts_1x_gb10) |
| sweep-depth-width/6-mlpmult3-layers20 | 2026-03-31 09:45 | 17 | 2.2M | 5.6322 | 5.5179 | 3.2680 | - | - | 🔴 3.2679 (23.0M) | - | - | - | - | [artifacts](sweep-depth-width/6-mlpmult3-layers20/artifacts_1x_gb10) |
| 001-weight-decay | 2026-03-25 14:03 | 675 | 88.5M | 6.0388 | 6.0327 | 3.5729 | - | 🟢 nan (4.1M) | 🟢 3.5737 (8.3M) | 🟢 3.5951 (8.5M) | 🟢 3.5736 (9.2M) | - | - | [artifacts](001-weight-decay/artifacts_1x_gb10) |
| sweep-depth-width/1-mlpmult2-layers15 | 2026-03-31 10:03 | - | - | - | - | - | - | - | - | - | - | - | - | [artifacts](sweep-depth-width/1-mlpmult2-layers15/artifacts_1x_gb10) |
| sweep-depth-width/2-mlpmult3-layers15 | 2026-03-31 10:03 | - | - | - | - | - | - | - | - | - | - | - | - | [artifacts](sweep-depth-width/2-mlpmult3-layers15/artifacts_1x_gb10) |
| sweep-depth-width/3-mlpmult4-layers15 | 2026-03-31 10:03 | - | - | - | - | - | - | - | - | - | - | - | - | [artifacts](sweep-depth-width/3-mlpmult4-layers15/artifacts_1x_gb10) |
| sweep-depth-width/4-mlpmult5-layers15 | 2026-03-31 10:03 | - | - | - | - | - | - | - | - | - | - | - | - | [artifacts](sweep-depth-width/4-mlpmult5-layers15/artifacts_1x_gb10) |
| sweep-depth-width/5-mlpmult2-layers20 | 2026-03-31 10:03 | - | - | - | - | - | - | - | - | - | - | - | - | [artifacts](sweep-depth-width/5-mlpmult2-layers20/artifacts_1x_gb10) |

## 1x RTX PRO 6000

| Experiment | Date | Steps | Tokens | Loss | Val Loss | Val BPB | int4 BPB | int6 BPB | int8 BPB | mxfp4 BPB | nvfp4 BPB | turboquip4c BPB | turboquip4cr BPB | Artifacts |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| baseline | 2026-03-25 22:27 | 831 | 435.7M | 2.3775 | 2.3714 | 1.4045 | - | 🟢 1.4202 (10.7M) | 🟢 1.4216 (10.7M) | 🟢 1.5157 (8.4M) | 🟢 1.5452 (8.7M) | - | - | [artifacts](baseline/artifacts_1x_rtx_pro_6000_2) |

## 2x RTX PRO 6000

| Experiment | Date | Steps | Tokens | Loss | Val Loss | Val BPB | int4 BPB | int6 BPB | int8 BPB | mxfp4 BPB | nvfp4 BPB | turboquip4c BPB | turboquip4cr BPB | Artifacts |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| baseline | 2026-03-25 23:32 | 1635 | 857.2M | 2.2619 | 2.2595 | 1.3382 | - | 🟢 1.3680 (11.1M) | 🟢 1.3397 (15.3M) | 🟢 1.5032 (8.6M) | 🟢 1.4611 (9.2M) | - | - | [artifacts](baseline/artifacts_2x_rtx_pro_6000_2) |

## 8x H100

| Experiment | Date | Steps | Tokens | Loss | Val Loss | Val BPB | int4 BPB | int6 BPB | int8 BPB | mxfp4 BPB | nvfp4 BPB | turboquip4c BPB | turboquip4cr BPB | Artifacts |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| baseline | 2026-03-28 21:53 | 12309 | 6453.5M | 2.0732 | 2.1094 | 1.2493 | 🟢 4.0150 (7.4M) | 🟢 1.3665 (11.5M) | 🟢 1.2561 (15.8M) | 🟢 2.6693 (8.7M) | 🟢 1.8566 (9.2M) | 🟢 1.3237 (7.3M) | 🟢 1.3208 (9.9M) | [artifacts](baseline/artifacts_8x_h100_4) |

## 8x RTX PRO 6000

| Experiment | Date | Steps | Tokens | Loss | Val Loss | Val BPB | int4 BPB | int6 BPB | int8 BPB | mxfp4 BPB | nvfp4 BPB | turboquip4c BPB | turboquip4cr BPB | Artifacts |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 005-mlp-3x | 2026-03-24 14:04 | 3640 | 2385.5M | 2.1994 | 2.1299 | 1.2614 | - | 🟢 1.3111 (14.4M) | 🔴 1.2643 (19.9M) | 🟢 1.5457 (11.0M) | 🟢 1.4770 (11.7M) | - | - | [artifacts](005-mlp-3x/artifacts_8x_rtx_pro_6000) |
| 004-momentum-warmup | 2026-03-24 15:33 | 4292 | 2812.8M | 2.1632 | 2.1481 | 1.2722 | - | 🟢 1.3825 (11.5M) | 🟢 1.2794 (15.7M) | 🟢 2.3571 (8.7M) | 🟢 1.9462 (9.2M) | - | - | [artifacts](004-momentum-warmup/artifacts_8x_rtx_pro_6000) |
| 003-grad-clip | 2026-03-24 17:03 | 4301 | 2818.7M | 2.1749 | 2.1535 | 1.2754 | - | 🟢 1.3444 (11.6M) | 🟢 1.2805 (15.7M) | 🟢 1.6938 (8.6M) | 🟢 1.5664 (9.2M) | - | - | [artifacts](003-grad-clip/artifacts_8x_rtx_pro_6000) |
| baseline | 2026-03-24 18:32 | 4147 | 2717.8M | 2.1428 | 2.1551 | 1.2764 | - | 🟢 1.3580 (11.4M) | 🟢 1.2806 (15.7M) | 🟢 1.8141 (8.7M) | 🟢 1.6003 (9.2M) | - | - | [artifacts](baseline/artifacts_8x_rtx_pro_6000_4) |
