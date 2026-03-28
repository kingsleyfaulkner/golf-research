# TurboQuIP: Combining QuIP# and TurboQuant for Practical Weight Quantisation

**Kingsley Faulkner**
Daystrom Research

## Abstract

This report describes TurboQuIP, a practical post-training quantisation pipeline that combines two existing techniques:
QuIP#'s randomised Hadamard incoherence processing with GPTQ rounding (Tseng et al., 2024) and TurboQuant's QJL
sign-sketch residual correction (Zandieh et al., 2025). The algorithmic components are drawn almost entirely from
prior work - the contribution here is a clean synthesis into a single configurable pipeline, along with a practical fix
for Hessian numerical stability (eigenvalue-clamped PSD enforcement) that resolved widespread Cholesky failures during
GPTQ rounding. On a 17M-parameter decoder-only language model, calibrated TurboQuIP at 4 bits per weight achieves
1.576 BPB - within 0.042 of the full-precision baseline (1.535 BPB) - while compressing to 5.85 MB with zstandard,
outperforming standard int4 quantisation (1.903 BPB) by 0.33 BPB at comparable size.

## 1. Introduction

Post-training quantisation (PTQ) compresses neural network weights from 16- or 32-bit floating point to lower
precision, reducing storage and memory bandwidth requirements. For size-constrained deployment the quality of the
quantisation scheme directly determines how many parameters - and therefore how many layers - can be allocated within a
fixed budget.

Two recent bodies of work offer complementary insights for weight quantisation:

**TurboQuant** (Zandieh et al., 2025) demonstrates that applying a random orthogonal rotation to vectors before scalar
quantisation achieves near-optimal distortion, within a factor of 2.7 of information-theoretic lower bounds. The
rotation induces a concentrated Beta distribution on coordinates, making simple per-coordinate quantisation effective
without data-dependent codebook training. A companion 1-bit Quantised Johnson-Lindenstrauss (QJL) transform (Zandieh
et al., 2024) provides zero-overhead residual correction with unbiased inner-product estimation.

**QuIP and QuIP#** (Chee et al., 2023; Tseng et al., 2024) show that incoherence processing via randomised Hadamard
transforms, combined with Hessian-guided adaptive rounding (LDLQ/GPTQ), achieves state-of-the-art 2-bit weight
quantisation for large language models. The Hadamard transform serves the same mathematical role as TurboQuant's random
rotation - spreading weight energy uniformly across coordinates - while the Hessian guides rounding decisions toward
directions that minimise output distortion.

These two lines of work are largely complementary: QuIP# provides the calibrated quantisation pipeline while TurboQuant
provides theoretical optimality guarantees and the QJL residual correction technique. TurboQuIP is a straightforward
combination of both, not a novel algorithm. The pipeline is:

1. **Randomised Hadamard Transform (RHT)** for incoherence processing (from QuIP#)
2. **Hessian-guided GPTQ rounding** (from GPTQ, adopted by QuIP#, optional)
3. **QJL sign-sketch residual correction** (from TurboQuant, optional)

The primary practical contributions are: (a) a clean implementation combining these techniques with data-free and
calibrated modes, and (b) an eigenvalue-clamped approach to Hessian regularisation that resolved widespread Cholesky
failures during GPTQ rounding.

## 2. Method

### 2.1 Randomised Hadamard Transform (RHT)

For a weight matrix $W \in \mathbb{R}^{m \times n}$, the forward RHT applies:

$$\widetilde{W} = W \cdot H_n \cdot D$$

where $H_n$ is the normalised Walsh-Hadamard matrix ($H_n H_n^T = I$) and $D = \text{diag}(s)$ with
$s_i \in \{+1, -1\}$ drawn from a fixed seed. The inverse is:

$$W = \widetilde{W} \cdot D \cdot H_n$$

since both $D$ and $H_n$ are self-inverse (symmetric orthogonal). The transform is computed in $O(n \log n)$ via the
Fast Walsh-Hadamard Transform (FWHT), with dimensions padded to the nearest power of two.

**Theoretical justification.** TurboQuant (Zandieh et al., 2025, Theorem 1) proves that after random orthogonal
rotation, independent per-coordinate scalar quantisation achieves MSE distortion:

$$D_{\text{mse}} \leq \frac{\sqrt{3}\pi}{2} \cdot \frac{1}{4^b}$$

for $b$ bits per coordinate, within a constant factor of the information-theoretic lower bound $D^* \geq 1/4^b$. The
optimality gap is approximately 2.7, meaning the simple approach of "rotate then quantise independently" is provably
close to the best possible scheme at any bit width.

The RHT achieves incoherence $\mu = O(\sqrt{\log n})$ (Tseng et al., 2024, Lemma), ensuring no single coordinate
dominates the quantisation error. Storage overhead is negligible: only the random seed (8 bytes) is required to
regenerate the sign vector at dequantisation time.

### 2.2 Per-Row Symmetric Integer Quantisation

After the RHT, each row of $\widetilde{W}$ is quantised using symmetric integer quantisation with percentile clipping:

1. Compute per-row clipping threshold $c_i$ at the 99.99984th percentile of $|\widetilde{W}_{i,:}|$
2. Scale: $s_i = c_i / q_{\max}$ where $q_{\max} = 2^{b-1} - 1$
3. Quantise: $\hat{q}_{ij} = \text{clamp}(\text{round}(\widetilde{W}_{ij} / s_i), -q_{\max}, q_{\max})$

The quantised values are stored as int8 (regardless of target bit width), with per-row float16 scales. Zstandard
compression exploits the reduced entropy of low-bit values to achieve effective bit rates below the nominal $b$ bits
per weight.

### 2.3 Hessian-Guided GPTQ Rounding (Calibrated Mode)

When calibration data is available, round-to-nearest is replaced with GPTQ-style error-propagating rounding (Frantar
et al., 2023). The per-layer Hessian $H = \mathbb{E}[x x^T]$ is estimated from a calibration forward pass, then
transformed to the RHT coordinate system:

$$\widetilde{H} = D \cdot H_n \cdot H \cdot H_n \cdot D$$

This transformation preserves eigenvalues (it is a similarity transform), allowing GPTQ to operate in the rotated
space where coordinates are incoherent.

**Robust PSD enforcement.** Float32 accumulation of outer products can produce Hessians with slightly negative
eigenvalues ($\lambda_{\min} \approx -82$ was observed in some layers). Standard Cholesky-based GPTQ fails on non-PSD
matrices. This is addressed via eigenvalue decomposition:

$$\widetilde{H} = V \Lambda V^T, \quad \Lambda_{\text{clamped}} = \max(\Lambda, \delta I), \quad \delta = 0.01 \cdot \text{mean}(|\Lambda|)$$

The inverse Hessian is then computed robustly as $\widetilde{H}^{-1} = V \Lambda_{\text{clamped}}^{-1} V^T$, and its
Cholesky factor $L$ is used for column-wise error propagation:

For $j = 1, \ldots, n$:
1. Quantise column $j$: $\hat{q}_j = \text{quantise}(W_{:,j})$
2. Compute scaled error: $e_j = (W_{:,j} - \text{dequant}(\hat{q}_j)) / L_{jj}$
3. Propagate: $W_{:,j+1:} \leftarrow W_{:,j+1:} - e_j \cdot L_{j,j+1:}$

This procedure adjusts subsequent columns to compensate for each rounding decision, weighted by the Hessian's
indication of which directions matter most for the layer's output.

### 2.4 QJL Residual Correction (Optional)

After primary quantisation, a 1-bit residual correction based on the Quantised Johnson-Lindenstrauss transform
(Zandieh et al., 2024) can optionally be applied. For each row $i$:

1. Compute residual: $r_i = \widetilde{W}_{i,:} - \text{dequant}(\hat{q}_{i,:})$
2. Store: $\text{sign}(H_n \cdot r_i) \in \{0, 1\}^n$ (1 bit per weight, packed into uint8)
3. Store: $\|r_i\|_2$ (one float16 scalar per row)

At dequantisation, the residual is reconstructed as:

$$\hat{r}_i = \|r_i\|_2 \cdot \sqrt{\frac{\pi}{2n}} \cdot H_n \cdot \text{sign\_bits}_i$$

where sign bits are mapped to $\{-1, +1\}$ before the inverse Hadamard. This estimator is unbiased for inner products
(Zandieh et al., 2024, Lemma 3.2): $\mathbb{E}[\langle y, \hat{r}_i \rangle] = \langle y, r_i \rangle$ for any vector
$y$, ensuring that the residual correction does not introduce systematic bias into the layer's output.

The QJL residual adds exactly 1 bit per weight plus one float16 norm per row, with zero quantisation constants (no
scales or zero-points), making it the most overhead-efficient residual correction available.

## 3. Implementation

### 3.1 Scheme Naming Convention

| Scheme | Description |
|--------|-------------|
| `turboquipN` | Data-free: RHT + round-to-nearest at N bits |
| `turboquipNr` | Data-free + QJL residual correction |
| `turboquipNc` | Calibrated: RHT + GPTQ rounding at N bits |
| `turboquipNcr` | Calibrated + QJL residual correction |

where N is the primary bit width (2-8).

### 3.2 Pipeline

```
Quantise:  [calibration] -> RHT forward -> quantise (RTN or GPTQ) -> [QJL residual] -> zstd compress -> save
Dequantise: load -> zstd decompress -> dequant integers -> [add QJL residual] -> RHT inverse -> cast to dtype
```

All stages operate on CPU tensors. Calibration (Hessian capture) runs on GPU with bfloat16 autocast. The dequantised
output is a standard floating-point state dict loadable by any PyTorch model.

### 3.3 Fast Walsh-Hadamard Transform

The FWHT is implemented as a vectorised butterfly network operating on the last dimension of a tensor:

```python
def _fwht(x: Tensor) -> Tensor:
    n = x.shape[-1]
    h = 1
    while h < n:
        x = x.reshape(*x.shape[:-1], -1, 2, h)
        a = x[..., 0, :] + x[..., 1, :]
        b = x[..., 0, :] - x[..., 1, :]
        x = torch.stack([a, b], dim=-2).reshape(*x.shape[:-3], n)
        h *= 2
    return x * (n ** -0.5)
```

The transform is self-inverse: `_fwht(_fwht(x)) = x`, verified by the orthogonality of the normalised Hadamard
matrix. Input dimensions are padded to the nearest power of two.

## 4. Experiments

### 4.1 Setup

Evaluation uses a 17M-parameter decoder-only transformer with 9 layers (4 encoder + 5 decoder), hidden dimension 512,
grouped-query attention (8 heads, 4 KV heads), and ReLU-squared MLP activation. The model is trained on the FineWeb
10B dataset with a 1024-token SentencePiece vocabulary.

Quality is measured by bits-per-byte (BPB) on the FineWeb validation set (~62M tokens). All quantised checkpoints are
compressed with zstandard at level 22.

For calibrated schemes, Hessians are captured from 1024 training sequences (1M tokens) via forward hooks on all
`torch.nn.Linear` modules.

### 4.2 Results

| Scheme | BPB | Compressed Size | vs FP16 |
|--------|-----|----------------|---------|
| FP16 (baseline) | 1.535 | - | - |
| int4 | 1.903 | 5.49 MB | +0.369 |
| int6 | 1.549 | 9.87 MB | +0.015 |
| mxfp4 | 1.656 | 8.47 MB | +0.122 |
| nvfp4 | 1.670 | 8.72 MB | +0.135 |
| **turboquip4** | 1.971 | 5.55 MB | +0.436 |
| **turboquip4c** | **1.576** | **5.85 MB** | **+0.042** |

### 4.3 Analysis

**Data-free vs calibrated.** The RHT alone (turboquip4, data-free) performs slightly worse than plain int4 (1.971 vs
1.903 BPB). This is expected: the RHT spreads energy uniformly but does not account for which weight directions matter
most for the model's output. Adding Hessian-guided GPTQ rounding (turboquip4c) reduces degradation from +0.436 to
+0.042 BPB - a 10x improvement. This confirms what QuIP# already demonstrated: calibration-guided rounding is the
dominant factor, with the RHT primarily serving to make uniform scalar quantisation effective.

**Comparison with existing schemes.** At 4 bits per weight, turboquip4c (1.576 BPB, 5.85 MB) outperforms:
- int4 (1.903 BPB, 5.49 MB): 0.327 BPB improvement
- mxfp4 (1.656 BPB, 8.47 MB): 0.080 BPB improvement at 69% of the size
- nvfp4 (1.670 BPB, 8.72 MB): 0.094 BPB improvement at 67% of the size

These gains are expected given QuIP#'s published results at similar bit widths on larger models. The contribution here
is confirming these results transfer to small models with a clean implementation.

**Robust Hessian handling.** Before implementing eigenvalue-clamped PSD enforcement, 39 of 55 linear layers failed
Cholesky decomposition due to slightly negative Hessian eigenvalues (up to $\lambda_{\min} = -82$) from float32
accumulation errors. After the fix, all 55 layers successfully use GPTQ rounding, improving BPB from 1.671 to 1.576.
This 0.095 BPB improvement from a numerical fix - larger than the gap between turboquip4c and int6 - underscores that
robust Hessian handling is critical for practical deployment of GPTQ-based methods. Standard implementations typically
use higher damping constants or retry loops; the eigenvalue decomposition approach used here is more principled but
also more expensive ($O(n^3)$ per layer).

### 4.4 Attribution of Results

The results above should be understood as a validation of existing techniques, not evidence of a new method. The
quantisation quality is attributable to QuIP#'s RHT + GPTQ pipeline. The QJL residual from TurboQuant was tested but
provided only marginal improvement (+0.003 BPB) at the cost of ~40% more compressed size, and is omitted from the main
results. The eigenvalue-clamped Hessian fix is the only non-trivial engineering contribution specific to this work.

## 5. Limitations and Future Work

**Quantisation time.** The GPTQ column-wise rounding loop is sequential in $O(n)$ columns, each touching all $m$ rows.
For 512-dimensional layers this completes in seconds, but larger models would benefit from block-wise processing.

**Calibration data sensitivity.** The current evaluation uses 1024 sequences from training data. The sensitivity to
calibration set size and distribution has not been systematically studied.

**Low bit widths.** TurboQuIP at 2-3 bits (data-free) produces unusable results (13.6 and 5.7 BPB respectively).
Calibrated 2-3 bit performance has not been evaluated but is expected to improve significantly based on the 4-bit
results and QuIP#'s published 2-bit results on larger models.

**E8 lattice codebooks.** QuIP# achieves further improvements via E8 lattice-based vector quantisation, which exploits
8-dimensional correlations between adjacent weights. This is the most impactful missing component from the QuIP#
pipeline and the most likely path to improved quality at 2-3 bits per weight.

**Entropy coding.** TurboQuant (Zandieh et al., 2025) notes that entropy coding of quantised indices can reduce storage
by ~5%, exploiting the analytically known post-RHT coordinate distribution. The current implementation relies on
zstandard compression, which provides similar benefits implicitly.

## References

- Chee, J., Cai, Y., Kuleshov, V., & De Sa, C. (2023). QuIP: 2-Bit Quantization of Large Language Models With
  Guarantees. *arXiv:2307.13304*.
- Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D. (2023). GPTQ: Accurate Post-Training Quantization for
  Generative Pre-trained Transformers. *ICLR 2023*.
- Tseng, A., Chee, J., Sun, Q., Kuleshov, V., & De Sa, C. (2024). QuIP#: Even Better LLM Quantization with Hadamard
  Incoherence and Lattice Codebooks. *ICML 2024*. *arXiv:2402.04396*.
- Zandieh, A., Daliri, M., & Han, I. (2024). QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero
  Overhead. *AAAI 2024*. *arXiv:2406.03482*.
- Zandieh, A., Daliri, M., Hadian, M., & Mirrokni, V. (2025). TurboQuant: Online Vector Quantization with
  Near-optimal Distortion Rate. *ICLR 2026*. *arXiv:2504.19874*.
- Zandieh, A., Kacham, P., Han, I., Daliri, M., & Mirrokni, V. (2025). PolarQuant: Quantizing KV Cache of LLMs via
  Polar Coordinate Transformation. *AISTATS 2026*. *arXiv:2502.02617*.

## Appendix A: Self-Contained Reference Implementation

The following is a standalone implementation of the TurboQuIP quantisation and dequantisation pipeline. It depends only
on PyTorch and requires no external libraries beyond the standard library.

```python
"""TurboQuIP: standalone reference implementation.

Quantises a PyTorch state dict using Randomised Hadamard Transform (RHT)
incoherence processing, per-row symmetric integer quantisation, optional
GPTQ-style Hessian-guided rounding, and optional QJL residual correction.

Usage:
    import torch
    from turboquip import quantise_state_dict, dequantise_state_dict

    state_dict = model.state_dict()
    quantised = quantise_state_dict(state_dict, bits=4)
    recovered = dequantise_state_dict(quantised)
    model.load_state_dict(recovered)
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

RHT_SEED = 0x48414D52
CLIP_PERCENTILE = 99.99984 / 100.0
GPTQ_PERCDAMP = 0.01
QUANT_RANGES = {2: 1, 3: 3, 4: 7, 5: 15, 6: 31, 7: 63, 8: 127}


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _fwht(x: Tensor) -> Tensor:
    """Normalised Fast Walsh-Hadamard Transform on the last dimension.

    The last dimension must be a power of 2. The transform is self-inverse:
    _fwht(_fwht(x)) == x.
    """
    n = x.shape[-1]
    h = 1
    while h < n:
        x = x.reshape(*x.shape[:-1], -1, 2, h)
        a = x[..., 0, :] + x[..., 1, :]
        b = x[..., 0, :] - x[..., 1, :]
        x = torch.stack([a, b], dim=-2).reshape(*x.shape[:-3], n)
        h *= 2
    return x * (n**-0.5)


def _generate_signs(seed: int, n: int) -> Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (n,), generator=gen, dtype=torch.float32) * 2 - 1


def _rht_forward(W: Tensor, signs: Tensor) -> Tensor:
    """W_tilde = W @ H @ D."""
    return _fwht(W) * signs


def _rht_inverse(W_tilde: Tensor, signs: Tensor) -> Tensor:
    """W = W_tilde @ D @ H."""
    return _fwht(W_tilde * signs)


def _pack_sign_bits(positive: Tensor) -> Tensor:
    flat = positive.reshape(-1).to(torch.uint8)
    pad = (-len(flat)) % 8
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=torch.uint8)])
    flat = flat.reshape(-1, 8)
    packed = torch.zeros(flat.shape[0], dtype=torch.uint8)
    for b in range(8):
        packed |= flat[:, b] << b
    return packed


def _unpack_sign_bits(packed: Tensor, numel: int) -> Tensor:
    flat = packed.reshape(-1)
    bits = torch.zeros(len(flat) * 8, dtype=torch.float32)
    for b in range(8):
        bits[b::8] = ((flat >> b) & 1).float()
    return bits[:numel] * 2.0 - 1.0


def _quantise_rows(W: Tensor, quant_range: int) -> tuple[Tensor, Tensor]:
    """Per-row symmetric integer quantisation with percentile clipping."""
    t32 = W.float()
    if t32.ndim == 2:
        clip_abs = torch.quantile(t32.abs(), CLIP_PERCENTILE, dim=1)
        clipped = torch.clamp(t32, -clip_abs[:, None], clip_abs[:, None])
        scale = (clip_abs / float(quant_range)).clamp_min(1.0 / float(quant_range))
        q = torch.clamp(
            torch.round(clipped / scale[:, None]), -quant_range, quant_range
        ).to(torch.int8)
        return q.contiguous(), scale.to(torch.float16).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), CLIP_PERCENTILE).item())
    scale = torch.tensor(max(clip_abs / float(quant_range), 1e-8), dtype=torch.float32)
    q = torch.clamp(
        torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale),
        -quant_range, quant_range,
    ).to(torch.int8)
    return q.contiguous(), scale


def _transform_hessian(H: Tensor, signs: Tensor, n_pad: int) -> Tensor:
    """Transform Hessian to RHT coordinate space: D @ Had @ H @ Had @ D."""
    n_orig = H.shape[0]
    H32 = H.float()
    if n_pad > n_orig:
        H32 = torch.nn.functional.pad(H32, (0, n_pad - n_orig, 0, n_pad - n_orig))
    H_rot = _fwht(H32)
    H_rot = _fwht(H_rot.T).T
    H_rot = H_rot * (signs.unsqueeze(1) * signs.unsqueeze(0))
    return H_rot


def _gptq_quantise(
    W: Tensor, H: Tensor, signs: Tensor, n_pad: int, quant_range: int
) -> tuple[Tensor, Tensor]:
    """GPTQ-style Hessian-guided quantisation with robust PSD enforcement."""
    H_rot = _transform_hessian(H, signs, n_pad)
    m, n = W.shape
    W = W.clone().float()

    eigvals, eigvecs = torch.linalg.eigh(H_rot)
    damp = GPTQ_PERCDAMP * eigvals.abs().mean()
    eigvals = eigvals.clamp(min=damp)
    H_inv = eigvecs @ torch.diag(1.0 / eigvals) @ eigvecs.T
    H_inv = (H_inv + H_inv.T) * 0.5

    try:
        L = torch.linalg.cholesky(H_inv, upper=True)
    except torch.linalg.LinAlgError:
        return _quantise_rows(W, quant_range)

    clip_abs = torch.quantile(W.abs(), CLIP_PERCENTILE, dim=1)
    scales = (clip_abs / float(quant_range)).clamp_min(1.0 / float(quant_range))

    Q = torch.zeros(m, n, dtype=torch.int8)
    for j in range(n):
        w_j = W[:, j]
        clipped = torch.clamp(w_j, -clip_abs, clip_abs)
        q_int = torch.clamp(torch.round(clipped / scales), -quant_range, quant_range)
        Q[:, j] = q_int.to(torch.int8)
        q_deq = q_int * scales
        err = (w_j - q_deq) / L[j, j]
        if j + 1 < n:
            W[:, j + 1:] -= err.unsqueeze(1) * L[j:j + 1, j + 1:]

    return Q.contiguous(), scales.to(torch.float16).contiguous()


def quantise_tensor(
    t: Tensor,
    bits: int = 4,
    hessian: Tensor | None = None,
    use_residual: bool = False,
    rht_seed: int = RHT_SEED,
) -> dict[str, Any]:
    """Quantise a single weight tensor using TurboQuIP.

    Args:
        t: Weight tensor (1D or 2D).
        bits: Quantisation bit width (2-8).
        hessian: Optional per-layer Hessian E[x x^T] for GPTQ rounding.
        use_residual: Whether to add 1-bit QJL residual correction.
        rht_seed: Seed for random sign generation.

    Returns:
        Dict containing all data needed for dequantisation.
    """
    t32 = t.float()
    orig_shape = list(t.shape)
    is_1d = t32.ndim == 1
    if is_1d:
        t32 = t32.unsqueeze(0)

    m, n = t32.shape
    n_pad = _next_pow2(n)
    if n_pad > n:
        t32 = torch.nn.functional.pad(t32, (0, n_pad - n))

    signs = _generate_signs(rht_seed, n_pad)
    w_tilde = _rht_forward(t32, signs)

    quant_range = QUANT_RANGES[bits]
    if hessian is not None:
        q, s = _gptq_quantise(w_tilde, hessian, signs, n_pad, quant_range)
    else:
        q, s = _quantise_rows(w_tilde, quant_range)

    result: dict[str, Any] = {
        "quantised": q,
        "scales": s,
        "rht_seed": rht_seed,
        "orig_shape": orig_shape,
        "n_pad": n_pad,
        "bits": bits,
        "use_residual": use_residual,
    }

    if use_residual:
        if s.ndim > 0:
            w_hat = q.float() * s.float().view(-1, 1)
        else:
            w_hat = q.float() * float(s.item())
        residual = w_tilde - w_hat
        norms = residual.norm(dim=1)
        sign_positive = _fwht(residual) >= 0
        result["qjl_sign_bits"] = _pack_sign_bits(sign_positive)
        result["qjl_norms"] = norms.to(torch.float16)

    return result


def dequantise_tensor(data: dict[str, Any], dtype: torch.dtype) -> Tensor:
    """Dequantise a TurboQuIP tensor back to floating point."""
    q = data["quantised"]
    s = data["scales"]
    n_pad = data["n_pad"]
    orig_shape = data["orig_shape"]

    if isinstance(s, Tensor) and s.ndim > 0:
        w_tilde = q.float() * s.float().view(-1, 1)
    else:
        w_tilde = q.float() * (float(s.item()) if isinstance(s, Tensor) else float(s))

    if data.get("use_residual") and "qjl_sign_bits" in data:
        norms = data["qjl_norms"].float()
        m, n = w_tilde.shape
        sign_float = _unpack_sign_bits(data["qjl_sign_bits"], m * n).reshape(m, n)
        correction = _fwht(sign_float) * (norms.unsqueeze(1) * math.sqrt(math.pi / 2) / n)
        w_tilde = w_tilde + correction

    signs = _generate_signs(data["rht_seed"], n_pad)
    w = _rht_inverse(w_tilde, signs)

    n_orig = orig_shape[-1] if len(orig_shape) >= 2 else orig_shape[0]
    w = w[:, :n_orig]
    if len(orig_shape) == 1:
        w = w.squeeze(0)

    return w.reshape(orig_shape).to(dtype).contiguous()


def quantise_state_dict(
    state_dict: dict[str, Tensor],
    bits: int = 4,
    hessians: dict[str, Tensor] | None = None,
    use_residual: bool = False,
    min_numel: int = 65_536,
) -> dict[str, Any]:
    """Quantise an entire model state dict.

    Tensors with fewer than ``min_numel`` elements or non-float dtypes are
    stored as-is in the ``passthrough`` dict.

    Args:
        state_dict: Model state dict.
        bits: Quantisation bit width (2-8).
        hessians: Per-layer Hessians keyed by weight name (e.g. from forward hooks).
        use_residual: Whether to add QJL residual correction.
        min_numel: Tensors smaller than this are passed through unquantised.
    """
    quantised: dict[str, Any] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}

    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        if not t.is_floating_point() or t.numel() <= min_numel:
            passthrough[name] = t
            continue
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        h = hessians.get(name) if hessians else None
        quantised[name] = quantise_tensor(t, bits=bits, hessian=h, use_residual=use_residual)

    return {
        "quantised": quantised,
        "dtypes": dtypes,
        "passthrough": passthrough,
        "bits": bits,
    }


def dequantise_state_dict(obj: dict[str, Any]) -> dict[str, Tensor]:
    """Reconstruct a full-precision state dict from a quantised object."""
    out: dict[str, Tensor] = {}
    for name, data in obj["quantised"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        out[name] = dequantise_tensor(data, dtype)
    for name, t in obj["passthrough"].items():
        out[name] = t.detach().cpu().contiguous()
    return out


def capture_hessians(
    model: torch.nn.Module,
    tokens: Tensor,
    seq_len: int = 1024,
    batch_size: int = 8,
    device: str = "cuda",
) -> dict[str, Tensor]:
    """Capture per-linear-layer Hessians H = E[x x^T] from calibration data.

    Args:
        model: The model to calibrate.
        tokens: 1D tensor of token IDs.
        seq_len: Sequence length for batching.
        batch_size: Number of sequences per forward pass.
        device: Device for forward passes.

    Returns:
        Dict mapping weight state-dict keys to CPU Hessian tensors.
    """
    hessians: dict[str, Tensor] = {}
    nsamples: dict[str, int] = {}
    handles = []

    vocab_size = None
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            vocab_size = module.num_embeddings
            break

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            key = f"{name}.weight"

            def _make_hook(k: str):
                def hook(mod, inp, out):
                    x = inp[0].detach().float()
                    if x.ndim == 3:
                        x = x.reshape(-1, x.shape[-1])
                    h = (x.T @ x).cpu()
                    if k not in hessians:
                        hessians[k] = h
                        nsamples[k] = x.shape[0]
                    else:
                        hessians[k] += h
                        nsamples[k] += x.shape[0]
                return hook

            handles.append(module.register_forward_hook(_make_hook(key)))

    model.eval()
    num_seqs = (tokens.numel() - 1) // seq_len
    autocast = torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16)

    with torch.no_grad(), autocast:
        for i in range(0, num_seqs, batch_size):
            batch_end = min(i + batch_size, num_seqs)
            batch = torch.stack([tokens[j * seq_len:(j + 1) * seq_len] for j in range(i, batch_end)])
            if vocab_size is not None:
                batch = batch.clamp(max=vocab_size - 1)
            model(batch.to(device))

    for h in handles:
        h.remove()
    for key in hessians:
        hessians[key] /= nsamples[key]

    return hessians
```
