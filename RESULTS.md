# Results

## 1x RTX PRO 6000

| Experiment | Date | Steps | Loss | Val Loss | Val BPB | mxfp4 BPB | nvfp4 BPB | int6 BPB | int8 BPB | Artifacts |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| baseline | 2026-03-23 16:01 | 320 | 2.6495 | 2.6324 | 1.5591 | 🟢 1.6388 (7.8M) | 🟢 1.6603 (8.1M) | 🟢 1.5756 (9.3M) | 🟢 1.5770 (8.1M) | [artifacts](experiments/baseline/artifacts_1x_rtx_pro_6000) |

## 2x RTX PRO 6000

| Experiment | Date | Steps | Loss | Val Loss | Val BPB | mxfp4 BPB | nvfp4 BPB | int6 BPB | int8 BPB | Artifacts |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| baseline | 2026-03-23 16:24 | 610 | 2.4345 | 2.4411 | 1.4458 | 🟢 1.5351 (8.3M) | 🟢 1.5503 (8.6M) | 🟢 1.4598 (11.1M) | 🟢 1.4607 (9.6M) | [artifacts](experiments/baseline/artifacts_2x_rtx_pro_6000) |

## 8x H100

| Experiment | Date | Steps | Loss | Val Loss | Val BPB | mxfp4 BPB | nvfp4 BPB | int6 BPB | int8 BPB | Artifacts |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| baseline | 2026-03-23 17:09 | 2375 | 2.1258 | 2.1406 | 1.2678 | 🟢 1.6109 (8.5M) | 🟢 1.7160 (8.7M) | 🟢 1.2992 (12.3M) | 🟢 1.3004 (11.3M) | [artifacts](experiments/baseline/artifacts_8x_h100_2) |

## 8x RTX PRO 6000

| Experiment | Date | Steps | Loss | Val Loss | Val BPB | mxfp4 BPB | nvfp4 BPB | int6 BPB | int8 BPB | Artifacts |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 005-mlp-3x | 2026-03-24 13:09 | 3640 | 2.1994 | 2.1299 | 1.2614 | 🟢 1.6422 (11.1M) | 🟢 1.8285 (11.3M) | 🟢 1.3111 (14.4M) | 🔴 1.2643 (19.9M) | [artifacts](experiments/005-mlp-3x/artifacts_8x_rtx_pro_6000) |
| 004-momentum-warmup | 2026-03-23 14:42 | 4292 | 2.1632 | 2.1481 | 1.2722 | 🟢 2.7400 (8.8M) | 🟢 3.0434 (9.0M) | 🟢 1.3825 (12.8M) | 🟢 1.3812 (12.3M) | [artifacts](experiments/004-momentum-warmup/artifacts_8x_rtx_pro_6000) |
| 003-grad-clip | 2026-03-24 07:30 | 4301 | 2.1749 | 2.1535 | 1.2754 | 🟢 1.6938 (8.6M) | 🟢 1.6574 (9.4M) | 🟢 1.3444 (12.8M) | 🟢 1.3433 (12.3M) | [artifacts](experiments/003-grad-clip/artifacts_8x_rtx_pro_6000) |
| baseline | 2026-03-23 14:19 | 4147 | 2.1428 | 2.1551 | 1.2764 | 🟢 - (8.5M) | 🟢 1.7303 (9.4M) | 🟢 1.3580 (12.8M) | 🟢 1.3570 (12.3M) | [artifacts](experiments/baseline/artifacts_8x_rtx_pro_6000_4) |
