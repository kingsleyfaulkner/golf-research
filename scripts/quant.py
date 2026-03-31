"""Post-training quantization with compression.

Loads a Composer checkpoint, applies quantization, and saves compressed
artifacts. Supports multiple quantization schemes:

  - int4..int8: Per-row symmetric integer quantization with bit-packing
  - mxfp4: OCP Microscaling FP4 (E2M1 values, E8M0 block scales, block=32)
  - nvfp4: NVIDIA FP4 (E2M1 values, E4M3 block scales + FP32 tensor scale, block=16)
  - turboquipN: TurboQuIP N-bit (Randomised Hadamard Transform + symmetric integer)
  - turboquipNr: TurboQuIP N-bit with 1-bit QJL residual correction

TurboQuIP combines incoherence processing (RHT from QuIP#) with near-optimal scalar
quantization (TurboQuant) and optional QJL sign-sketch residual correction. The RHT
spreads weight energy uniformly across coordinates, making simple per-coordinate
quantization near-optimal. The QJL residual adds 1 bit per weight with zero overhead
(no quantization constants) for unbiased inner-product correction.

The quantized files are saved inside the checkpoint directory with the
scheme encoded in the name (e.g. ``step_2823_final_model_int6``,
``step_2823_final_model_turboquip4r``).

Usage:
    python quant.py                                    # default int8
    python quant.py --schemes int6,int8,mxfp4          # multiple schemes
    python quant.py --schemes turboquip4,turboquip3r   # TurboQuIP variants
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import re
import shutil
import tarfile
import tempfile
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import zstandard

    _COMPRESSOR = "zstd"
except ImportError:
    _COMPRESSOR = "zlib"

import click
import numpy as np
import torch
from composer.nn.architecture.model import Model
from composer.utils.logging import setup_logging
from torch import Tensor

logger = logging.getLogger(__name__)

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",")
    if pattern
)
KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
KEEP_FLOAT_MAX_NUMEL = 65_536
KEEP_FLOAT_STORE_DTYPE = torch.float16
PER_ROW_SCALE_DTYPE = torch.float16
CLIP_PERCENTILE = 99.99984
CLIP_Q = CLIP_PERCENTILE / 100.0

# All supported scheme names (turboquipN / turboquipNr validated dynamically)
SUPPORTED_SCHEMES = {"int4", "int5", "int6", "int7", "int8", "mxfp4", "nvfp4"}

# intN clamp ranges
QUANT_RANGES = {4: 7, 5: 15, 6: 31, 7: 63, 8: 127}

# TurboQuIP: extended ranges for 2-3 bit, plus existing intN ranges
TURBOQUIP_QUANT_RANGES = {2: 1, 3: 3, **QUANT_RANGES}
TURBOQUIP_RHT_SEED = 0x48414D52
TURBOQUIP_GPTQ_PERCDAMP = 0.01
_TURBOQUIP_RE = re.compile(r"^turboquip(\d+)(c?)(r?)$")

# FP4 E2M1 lookup table (indices 0-15, sign bit is MSB)
FP4_E2M1_LUT = torch.tensor(
    [
        +0.0,
        +0.5,
        +1.0,
        +1.5,
        +2.0,
        +3.0,
        +4.0,
        +6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)
FP4_E2M1_MAX = 6.0
E8M0_BIAS = 127


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t


def compress_bytes(raw: bytes) -> bytes:
    if _COMPRESSOR == "zstd":
        return zstandard.ZstdCompressor(level=22).compress(raw)
    return zlib.compress(raw, level=9)


def decompress_bytes(blob: bytes) -> bytes:
    try:
        import zstandard as _zstd

        return _zstd.ZstdDecompressor().decompress(blob)
    except Exception:
        return zlib.decompress(blob)


def pack_nbits(q: Tensor, bits: int) -> Tensor:
    """Pack signed int8 values into a bit-packed uint8 tensor."""
    if bits == 8:
        return q
    offset = 1 << (bits - 1)
    flat = q.reshape(-1).numpy().astype(np.int16) + offset
    n = len(flat)
    vals = flat.astype(np.uint64)
    bit_matrix = np.zeros((n, bits), dtype=np.uint8)
    for b in range(bits):
        bit_matrix[:, b] = (vals >> b) & 1
    bit_stream = bit_matrix.reshape(-1)
    pad = (-len(bit_stream)) % 8
    if pad:
        bit_stream = np.concatenate([bit_stream, np.zeros(pad, dtype=np.uint8)])
    bit_stream = bit_stream.reshape(-1, 8)
    packed = np.zeros(len(bit_stream), dtype=np.uint8)
    for b in range(8):
        packed |= bit_stream[:, b].astype(np.uint8) << b
    return torch.from_numpy(packed)


def unpack_nbits(packed: Tensor, bits: int, numel: int) -> Tensor:
    """Unpack a bit-packed uint8 tensor back to signed int8 values."""
    if bits == 8:
        return packed
    packed_np = packed.numpy()
    offset = 1 << (bits - 1)
    bit_stream = np.zeros(len(packed_np) * 8, dtype=np.uint8)
    for b in range(8):
        bit_stream[b::8] = (packed_np >> b) & 1
    total_bits = numel * bits
    bit_stream = bit_stream[:total_bits].reshape(numel, bits)
    vals = np.zeros(numel, dtype=np.int32)
    for b in range(bits):
        vals += bit_stream[:, b].astype(np.int32) << b
    result = (vals - offset).astype(np.int8)
    return torch.from_numpy(result)


def quantize_e2m1(x: Tensor) -> Tensor:
    """Quantize float values to FP4 E2M1 (4-bit indices 0-15).

    Uses the three-step threshold approach from fouroversix (MIT HAN Lab):
    - |x| < 2:  round to nearest {0, 0.5, 1, 1.5} (half-integer precision)
    - 2 <= |x| < 4: round to nearest {2, 3} (integer precision)
    - |x| >= 4: round to nearest {4, 6} (even-integer precision)

    Returns uint8 tensor with FP4 indices (0-7 positive, 8-15 negative).
    """
    x32 = x.float()
    ax = x32.abs()

    step1 = torch.round(2.0 * ax) / 2.0  # half-integer rounding
    step2 = torch.round(ax)  # integer rounding
    step3 = 2.0 * torch.round(ax / 2.0)  # even-integer rounding
    step3 = step3.clamp_max(FP4_E2M1_MAX)

    mask1 = ax < 2.0
    mask2 = ax < 4.0
    quantized = step1 * mask1 + step2 * (~mask1 & mask2) + step3 * (~mask1 & ~mask2)

    # Map float values to FP4 indices using the LUT
    # Positive values: 0->0, 0.5->1, 1->2, 1.5->3, 2->4, 3->5, 4->6, 6->7
    # Build reverse lookup: value -> index
    pos_lut = FP4_E2M1_LUT[:8]
    # Find nearest index by broadcasting comparison
    diffs = (quantized.unsqueeze(-1) - pos_lut.to(quantized.device)).abs()
    indices = diffs.argmin(dim=-1).to(torch.uint8)

    # Set sign bit (bit 3) for negative values
    indices = indices | ((x32 < 0).to(torch.uint8) << 3)
    return indices


def dequantize_e2m1(indices: Tensor) -> Tensor:
    """Dequantize FP4 E2M1 indices back to float values."""
    lut = FP4_E2M1_LUT.to(indices.device)
    return lut[indices.to(torch.long)]


def pack_fp4(indices: Tensor) -> Tensor:
    """Pack FP4 indices (uint8, one per element) into nibble pairs (2 per byte).

    Even-position values go in lower nibble, odd-position in upper nibble.
    Input numel must be even.
    """
    flat = indices.reshape(-1)
    if flat.numel() % 2 != 0:
        flat = torch.cat([flat, torch.zeros(1, dtype=torch.uint8, device=flat.device)])
    lo = flat[0::2]
    hi = flat[1::2]
    return (lo | (hi << 4)).contiguous()


def unpack_fp4(packed: Tensor, numel: int) -> Tensor:
    """Unpack nibble-packed FP4 back to uint8 indices."""
    flat = packed.reshape(-1)
    lo = flat & 0x0F
    hi = (flat >> 4) & 0x0F
    # Interleave: [lo0, hi0, lo1, hi1, ...]
    out = torch.empty(flat.numel() * 2, dtype=torch.uint8, device=packed.device)
    out[0::2] = lo
    out[1::2] = hi
    return out[:numel]


def quantize_mxfp4_tensor(t: Tensor, block_size: int = 32) -> dict[str, Tensor]:
    """Quantize a tensor using MXFP4 (OCP Microscaling).

    - Block size: 32 elements share one E8M0 (power-of-2) scale
    - Values quantized to FP4 E2M1, packed as nibble pairs

    Returns dict with 'blocks' (packed uint8) and 'scales' (int32, biased).
    """
    t32 = t.float().reshape(-1)

    # Pad to block boundary
    pad = (-t32.numel()) % block_size
    if pad:
        t32 = torch.cat([t32, torch.zeros(pad, dtype=torch.float32)])

    blocks_2d = t32.reshape(-1, block_size)
    num_blocks = blocks_2d.shape[0]

    # Compute per-block E8M0 scale (power-of-2 from max absolute value)
    block_amax = blocks_2d.abs().amax(dim=1).clamp_min(1e-12)

    # E8M0 shared exponent: floor(log2(amax)) - FP4_MAX_POW2
    # This is the standard OCP Microscaling approach from torchao.
    # The scale is a lower bound — values up to 1.5x the scale's range
    # are still representable by FP4 (since FP4 max = 6 = 1.5 * 2^2).
    amax_int32 = block_amax.view(torch.int32)
    extracted_exp = ((amax_int32 >> 23) & 0xFF).to(torch.int32) - 127
    scale_exp = extracted_exp - 2  # FP4 E2M1 max pow2 = 2

    scale_biased = (scale_exp + E8M0_BIAS).clamp(0, 255).to(torch.int32)

    # Convert biased exponent back to float for division
    scale_float = ((scale_biased << 23).view(torch.float32)).clamp_min(1e-45)

    # Scale and quantize
    scaled = blocks_2d / scale_float.unsqueeze(1)
    fp4_indices = quantize_e2m1(scaled)

    # Pack nibble pairs
    packed = pack_fp4(fp4_indices.reshape(-1)).reshape(num_blocks, block_size // 2)

    return {
        "blocks": packed,
        "scales": scale_biased,
        "num_elements": t.numel(),
        "orig_shape": list(t.shape),
    }


def dequantize_mxfp4_tensor(data: dict[str, Tensor], dtype: torch.dtype) -> Tensor:
    """Dequantize MXFP4 packed data back to a float tensor."""
    packed = data["blocks"]
    scale_biased = data["scales"]
    num_elements = data["num_elements"]
    orig_shape = data["orig_shape"]
    block_size = 32

    num_blocks = packed.shape[0]
    # Unpack nibbles
    indices = unpack_fp4(packed.reshape(-1), num_blocks * block_size)
    values = dequantize_e2m1(indices).reshape(num_blocks, block_size)

    # Apply E8M0 scaling
    scale_exp = (scale_biased.to(torch.int32) - E8M0_BIAS).unsqueeze(1)
    values = torch.ldexp(values, scale_exp.expand_as(values))

    return values.reshape(-1)[:num_elements].reshape(orig_shape).to(dtype)


def quantize_nvfp4_tensor(t: Tensor, block_size: int = 16) -> dict[str, Tensor]:
    """Quantize a tensor using NVFP4 (NVIDIA FP4).

    - Block size: 16 elements share one E4M3 scale
    - Per-tensor FP32 scale normalizes so block scales fit E4M3
    - Two-level scaling for better precision than MXFP4

    Returns dict with 'blocks' (packed uint8), 'block_scales' (float8_e4m3fn),
    and 'tensor_scale' (float32).
    """
    t32 = t.float().reshape(-1)

    # Pad to block boundary
    pad = (-t32.numel()) % block_size
    if pad:
        t32 = torch.cat([t32, torch.zeros(pad, dtype=torch.float32)])

    blocks_2d = t32.reshape(-1, block_size)
    num_blocks = blocks_2d.shape[0]

    # Level 1: per-tensor scale
    # sg = global_amax / (FP4_MAX * E4M3_MAX)
    E4M3_MAX = 448.0  # max representable E4M3 value
    global_amax = t32.abs().max().clamp_min(1e-12)
    tensor_scale = (global_amax / (FP4_E2M1_MAX * E4M3_MAX)).to(torch.float32)

    # Level 2: per-block E4M3 scale
    # sb = block_amax / (tensor_scale * FP4_MAX)
    block_amax = blocks_2d.abs().amax(dim=1).clamp_min(1e-12)
    block_scales_fp32 = block_amax / (tensor_scale * FP4_E2M1_MAX)
    block_scales_e4m3 = block_scales_fp32.to(torch.float8_e4m3fn)

    # Quantize: x / (sg * sb), then round to E2M1
    combined_scale = tensor_scale * block_scales_e4m3.float()
    scaled = blocks_2d / combined_scale.unsqueeze(1).clamp_min(1e-45)
    fp4_indices = quantize_e2m1(scaled)

    # Pack nibble pairs
    packed = pack_fp4(fp4_indices.reshape(-1)).reshape(num_blocks, block_size // 2)

    return {
        "blocks": packed,
        "block_scales": block_scales_e4m3,
        "tensor_scale": tensor_scale,
        "num_elements": t.numel(),
        "orig_shape": list(t.shape),
    }


def dequantize_nvfp4_tensor(data: dict[str, Tensor], dtype: torch.dtype) -> Tensor:
    """Dequantize NVFP4 packed data back to a float tensor."""
    packed = data["blocks"]
    block_scales = data["block_scales"]
    tensor_scale = data["tensor_scale"]
    num_elements = data["num_elements"]
    orig_shape = data["orig_shape"]
    block_size = 16

    num_blocks = packed.shape[0]
    indices = unpack_fp4(packed.reshape(-1), num_blocks * block_size)
    values = dequantize_e2m1(indices).reshape(num_blocks, block_size)

    # Apply two-level scaling: x_hat = sg * sb * fp4_value
    combined_scale = tensor_scale.float() * block_scales.float()
    values = values * combined_scale.unsqueeze(1)

    return values.reshape(-1)[:num_elements].reshape(orig_shape).to(dtype)


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _fwht(x: Tensor) -> Tensor:
    """Normalised Fast Walsh-Hadamard Transform on the last dimension.

    The last dimension must be a power of 2. The transform is self-inverse.
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


def _generate_rht_signs(seed: int, n: int) -> Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (n,), generator=gen, dtype=torch.float32) * 2 - 1


def _rht_forward(W: Tensor, signs: Tensor) -> Tensor:
    """Forward Randomised Hadamard Transform: W @ H @ diag(signs)."""
    return _fwht(W) * signs


def _rht_inverse(W_tilde: Tensor, signs: Tensor) -> Tensor:
    """Inverse RHT: W_tilde @ diag(signs) @ H."""
    return _fwht(W_tilde * signs)


def _pack_sign_bits(positive: Tensor) -> Tensor:
    """Pack a boolean tensor into uint8 (LSB-first bit packing)."""
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
    """Unpack uint8 back to float +1/-1 values."""
    flat = packed.reshape(-1)
    bits = torch.zeros(len(flat) * 8, dtype=torch.float32)
    for b in range(8):
        bits[b::8] = ((flat >> b) & 1).float()
    return bits[:numel] * 2.0 - 1.0


def _parse_turboquip_scheme(scheme: str) -> tuple[int, bool, bool] | None:
    m = _TURBOQUIP_RE.match(scheme)
    if not m:
        return None
    return int(m.group(1)), bool(m.group(2)), bool(m.group(3))


def _is_valid_single_scheme(s: str) -> bool:
    if s in SUPPORTED_SCHEMES:
        return True
    tq = _parse_turboquip_scheme(s)
    return tq is not None and 2 <= tq[0] <= 8


def _parse_compound_scheme(scheme: str) -> tuple[str, str]:
    """Parse an ``A-B`` compound scheme into (linear_scheme, embed_scheme).

    A single scheme (no dash) applies to both linear and embedding tensors.
    """
    parts = scheme.split("-", 1)
    if len(parts) == 1:
        return scheme, scheme
    return parts[0], parts[1]


def _is_embedding_tensor(name: str) -> bool:
    return "emb" in name.lower()


def quantize_turboquip_tensor(
    t: Tensor,
    bits: int,
    rht_seed: int,
    use_residual: bool,
    hessian: Tensor | None = None,
) -> dict[str, Any]:
    """Quantize a tensor using TurboQuIP (RHT + symmetric integer + optional QJL residual).

    Applies a Randomised Hadamard Transform to spread weight energy uniformly across
    coordinates (incoherence processing), then quantizes with per-row symmetric integers.
    Optionally adds a 1-bit QJL sign-sketch of the quantization residual for unbiased
    inner-product correction.
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

    signs = _generate_rht_signs(rht_seed, n_pad)
    w_tilde = _rht_forward(t32, signs)

    quant_range = TURBOQUIP_QUANT_RANGES[bits]
    if hessian is not None:
        q, s = _gptq_quantize(w_tilde, hessian, signs, n_pad, quant_range)
    else:
        q, s = _quantize_intN_tensor(w_tilde, quant_range)

    result: dict[str, Any] = {
        "quantized": q,
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


def dequantize_turboquip_tensor(data: dict[str, Any], dtype: torch.dtype) -> Tensor:
    """Dequantize a TurboQuIP tensor back to the original representation."""
    q = data["quantized"]
    s = data["scales"]
    rht_seed = data["rht_seed"]
    orig_shape = data["orig_shape"]
    n_pad = data["n_pad"]
    use_residual = data["use_residual"]

    if isinstance(s, Tensor) and s.ndim > 0:
        w_tilde = q.float() * s.float().view(-1, 1)
    else:
        s_val = float(s.item()) if isinstance(s, Tensor) else float(s)
        w_tilde = q.float() * s_val

    if use_residual and "qjl_sign_bits" in data:
        norms = data["qjl_norms"].float()
        m, n = w_tilde.shape
        sign_float = _unpack_sign_bits(data["qjl_sign_bits"], m * n).reshape(m, n)
        correction = _fwht(sign_float) * (norms.unsqueeze(1) * math.sqrt(math.pi / 2) / n)
        w_tilde = w_tilde + correction

    signs = _generate_rht_signs(rht_seed, n_pad)
    w = _rht_inverse(w_tilde, signs)

    n_orig = orig_shape[-1] if len(orig_shape) >= 2 else orig_shape[0]
    w = w[:, :n_orig]

    if len(orig_shape) == 1:
        w = w.squeeze(0)

    return w.reshape(orig_shape).to(dtype).contiguous()


def _transform_hessian_to_rht(H: Tensor, signs: Tensor, n_pad: int) -> Tensor:
    """Transform a Hessian to the RHT-rotated coordinate space.

    Given H = E[x x^T] in the original space, computes
    H_tilde = D @ Had @ H @ Had @ D in the rotated space where D = diag(signs).
    """
    n_orig = H.shape[0]
    H32 = H.float()
    if n_pad > n_orig:
        H32 = torch.nn.functional.pad(H32, (0, n_pad - n_orig, 0, n_pad - n_orig))
    # Had @ H @ Had (apply FWHT to both rows and columns)
    H_rot = _fwht(H32)  # rows: H @ Had
    H_rot = _fwht(H_rot.T).T  # cols: Had @ (H @ Had)
    # D @ ... @ D: multiply by outer product of signs
    H_rot = H_rot * (signs.unsqueeze(1) * signs.unsqueeze(0))
    return H_rot


def _gptq_quantize(
    W: Tensor, H_orig: Tensor, rht_signs: Tensor, n_pad: int, quant_range: int
) -> tuple[Tensor, Tensor]:
    """GPTQ-style Hessian-guided quantization in the RHT-rotated space.

    Processes columns sequentially, propagating rounding error to subsequent
    columns weighted by the inverse Hessian. Uses eigenvalue decomposition for
    robust PSD enforcement - float32 Hessian accumulation can produce slightly
    negative eigenvalues that would otherwise cause Cholesky failure.
    """
    H_rot = _transform_hessian_to_rht(H_orig, rht_signs, n_pad)

    m, n = W.shape
    W = W.clone().float()

    # Robust PSD enforcement via eigenvalue clamping
    eigvals, eigvecs = torch.linalg.eigh(H_rot)
    damp = TURBOQUIP_GPTQ_PERCDAMP * eigvals.abs().mean()
    eigvals = eigvals.clamp(min=damp)

    # Compute H_inv from clamped eigenvalues (always PD)
    H_inv = eigvecs @ torch.diag(1.0 / eigvals) @ eigvecs.T
    # Enforce exact symmetry (numerical noise from matmul)
    H_inv = (H_inv + H_inv.T) * 0.5

    try:
        L = torch.linalg.cholesky(H_inv, upper=True)
    except torch.linalg.LinAlgError:
        logger.warning("Cholesky failed after eigenvalue clamping, falling back")
        return _quantize_intN_tensor(W, quant_range)

    clip_abs = (
        torch.quantile(W.abs(), CLIP_Q, dim=1)
        if W.numel()
        else torch.empty((m,), dtype=torch.float32)
    )
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
            W[:, j + 1 :] -= err.unsqueeze(1) * L[j : j + 1, j + 1 :]

    return Q.contiguous(), scales.to(PER_ROW_SCALE_DTYPE).contiguous()


def _load_calibration_tokens(path_pattern: str, max_tokens: int = 0) -> Tensor:
    """Load tokenized data from binary files (uint16) for calibration."""
    import glob as globmod

    files = sorted(globmod.glob(path_pattern))
    if not files:
        raise ValueError(f"No calibration data files matching: {path_pattern}")

    chunks = []
    total = 0
    for f in files:
        data = np.memmap(f, dtype=np.uint16, mode="r")
        chunks.append(torch.from_numpy(data.astype(np.int64)))
        total += len(data)
        if 0 < max_tokens <= total:
            break

    tokens = torch.cat(chunks)
    if max_tokens > 0:
        tokens = tokens[:max_tokens]
    return tokens


def _capture_layer_hessians(
    model: torch.nn.Module,
    calibration_tokens: Tensor,
    seq_len: int,
    batch_size: int = 8,
    device: torch.device | str = "cpu",
) -> dict[str, Tensor]:
    """Run calibration data through the model and capture per-layer Hessians.

    Registers forward hooks on all ``torch.nn.Linear`` modules to accumulate
    H = E[x x^T] where x is the flattened layer input.  Returns a dict mapping
    weight state-dict keys (e.g. ``"encoder.0.attn.qkv.weight"``) to CPU Hessian
    tensors.
    """
    hessians: dict[str, Tensor] = {}
    nsamples: dict[str, int] = {}
    handles = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            weight_key = f"{name}.weight"

            def _make_hook(key: str):
                def hook(mod, inp, out):
                    x = inp[0].detach().float()
                    if x.ndim == 3:
                        x = x.reshape(-1, x.shape[-1])
                    h = (x.T @ x).cpu()
                    if key not in hessians:
                        hessians[key] = h
                        nsamples[key] = x.shape[0]
                    else:
                        hessians[key] += h
                        nsamples[key] += x.shape[0]

                return hook

            handles.append(module.register_forward_hook(_make_hook(weight_key)))

    model.eval()
    total = calibration_tokens.numel()
    num_seqs = (total - 1) // seq_len

    # Infer vocab_size from embedding layer to clamp out-of-range tokens
    vocab_size = None
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            vocab_size = module.num_embeddings
            break

    logger.info(
        f"[bold]Capturing Hessians:[/] [cyan]sequences[/]={num_seqs}" f" [cyan]seq_len[/]={seq_len}"
    )

    autocast = torch.autocast(device_type=str(device).split(":")[0], dtype=torch.bfloat16)
    with torch.no_grad(), autocast:
        for i in range(0, num_seqs, batch_size):
            batch_end = min(i + batch_size, num_seqs)
            batch = torch.stack(
                [calibration_tokens[j * seq_len : (j + 1) * seq_len] for j in range(i, batch_end)]
            )
            if vocab_size is not None:
                batch = batch.clamp(max=vocab_size - 1)
            model(batch.to(device))

    for h in handles:
        h.remove()

    for key in hessians:
        hessians[key] /= nsamples[key]

    logger.info(f"[bold green]Captured Hessians for {len(hessians)} layers[/]")
    return hessians


def _quantize_tensor_with_scheme(
    name: str,
    t: Tensor,
    eff_scheme: str,
    hessians: dict[str, Tensor] | None,
) -> tuple[Any, Tensor | None, dict[str, object] | None, int]:
    """Quantize a single tensor with the given scheme.

    Returns (quantized_data, scale_or_None, qmeta_or_None, payload_bytes).
    For intN: quantized_data is a Tensor, scale is set.
    For dict-based schemes (turboquip/mxfp4/nvfp4): quantized_data is a dict, scale is None.
    """
    eff_intN = eff_scheme.startswith("int")
    eff_tq = _parse_turboquip_scheme(eff_scheme)

    if eff_intN:
        bits = int(eff_scheme[3:])
        quant_range = QUANT_RANGES[bits]
        q, s = _quantize_intN_tensor(t, quant_range)
        meta: dict[str, object] = {}
        if s.ndim > 0:
            meta["scheme"] = "per_row"
            meta["axis"] = 0
        if bits < 8:
            meta["orig_shape"] = list(q.shape)
            meta["orig_numel"] = int(q.numel())
        return q, s, meta or None, tensor_nbytes(q) + tensor_nbytes(s)

    if eff_tq is not None:
        tq_bits, tq_calibrated, tq_residual = eff_tq
        layer_hessian = hessians.get(name) if hessians else None
        if tq_calibrated and layer_hessian is None:
            logger.warning(f"No Hessian for [cyan]{name}[/], using round-to-nearest")
        data = quantize_turboquip_tensor(
            t, tq_bits, TURBOQUIP_RHT_SEED, tq_residual, hessian=layer_hessian
        )
        payload = tensor_nbytes(data["quantized"]) + tensor_nbytes(data["scales"])
        if tq_residual:
            payload += tensor_nbytes(data["qjl_sign_bits"]) + tensor_nbytes(data["qjl_norms"])
        return data, None, None, payload

    if eff_scheme == "mxfp4":
        data = quantize_mxfp4_tensor(t)
        payload = tensor_nbytes(data["blocks"]) + tensor_nbytes(data["scales"])
        return data, None, None, payload

    if eff_scheme == "nvfp4":
        data = quantize_nvfp4_tensor(t)
        payload = (
            tensor_nbytes(data["blocks"])
            + tensor_nbytes(data["block_scales"])
            + tensor_nbytes(data["tensor_scale"])
        )
        return data, None, None, payload

    raise ValueError(f"Unknown scheme: {eff_scheme}")


def quantize_state_dict(
    state_dict: dict[str, Tensor],
    scheme: str = "int8",
    hessians: dict[str, Tensor] | None = None,
) -> tuple[dict[str, object], dict[str, int]]:
    """Quantize a state dict using the given scheme.

    Supports single schemes (e.g. ``int8``, ``turboquip4c``) and compound
    ``A-B`` schemes where A is applied to linear layers and B to embedding
    tensors (e.g. ``turboquip4c-int6``).
    """
    linear_scheme, embed_scheme = _parse_compound_scheme(scheme)
    is_compound = linear_scheme != embed_scheme

    quantized: dict[str, Any] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    tensor_schemes: dict[str, str] = {}
    stats = dict.fromkeys(
        (
            "param_count",
            "num_tensors",
            "num_float_tensors",
            "num_nonfloat_tensors",
            "baseline_tensor_bytes",
            "quantized_payload_bytes",
        ),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["quantized_payload_bytes"] += tensor_nbytes(t)
            continue

        if t.numel() <= KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["quantized_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        # Cast float32 master weights to bfloat16 before quantising.
        # Mixed-precision training computes forward passes in bfloat16 so the
        # extra float32 mantissa bits are never exploited during training.
        if t.dtype == torch.float32:
            t = t.to(torch.bfloat16)
        orig_dtype = str(t.dtype).removeprefix("torch.")

        eff_scheme = embed_scheme if _is_embedding_tensor(name) else linear_scheme
        q_data, s, meta, payload = _quantize_tensor_with_scheme(name, t, eff_scheme, hessians)

        quantized[name] = q_data
        dtypes[name] = orig_dtype
        if s is not None:
            scales[name] = s
        if meta is not None:
            qmeta[name] = meta
        if is_compound:
            tensor_schemes[name] = eff_scheme
        stats["quantized_payload_bytes"] += payload

    obj: dict[str, object] = {
        "__quant_format__": scheme,
        "__quant_scheme__": scheme,
        "quantized": quantized,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if scales:
        obj["scales"] = scales
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    if tensor_schemes:
        obj["tensor_schemes"] = tensor_schemes
    return obj, stats


def _quantize_intN_tensor(t: Tensor, quant_range: int) -> tuple[Tensor, Tensor]:
    """Per-row symmetric integer quantization with percentile clipping."""
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / float(quant_range)).clamp_min(1.0 / float(quant_range))
        q = (
            torch.clamp(torch.round(clipped / scale[:, None]), -quant_range, quant_range)
            .to(torch.int8)
            .contiguous()
        )
        return q, scale.to(dtype=PER_ROW_SCALE_DTYPE).contiguous()

    clip_abs = float(torch.quantile(t32.abs().flatten(), CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(
        clip_abs / float(quant_range) if clip_abs > 0 else 1.0, dtype=torch.float32
    )
    q = (
        torch.clamp(
            torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -quant_range, quant_range
        )
        .to(torch.int8)
        .contiguous()
    )
    return q, scale


def _dequantize_single_tensor(
    name: str,
    q_data: Any,
    dtype: torch.dtype,
    eff_scheme: str,
    obj: dict[str, object],
) -> Tensor:
    """Dequantize a single tensor given its effective scheme."""
    qmeta = obj.get("qmeta", {})

    if eff_scheme.startswith("int"):
        q = q_data
        s = obj["scales"][name]
        meta = qmeta.get(name, {})
        if "orig_numel" in meta:
            q = q.reshape(meta["orig_shape"])
        if meta.get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            return (
                (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
            )
        return (q.float() * float(s.item())).to(dtype=dtype).contiguous()

    if eff_scheme.startswith("turboquip"):
        return dequantize_turboquip_tensor(q_data, dtype)

    if eff_scheme == "mxfp4":
        return dequantize_mxfp4_tensor(q_data, dtype)

    if eff_scheme == "nvfp4":
        return dequantize_nvfp4_tensor(q_data, dtype)

    raise ValueError(f"Unknown scheme for dequantization: {eff_scheme}")


def dequantize_state_dict(obj: dict[str, object]) -> dict[str, Tensor]:
    """Reconstruct a full-precision state dict from a quantized object.

    Dispatches to the appropriate dequantization based on the scheme.
    Supports compound ``A-B`` schemes via per-tensor scheme metadata.
    """
    scheme = obj.get("__quant_scheme__") or obj.get("__quant_format__", "int8")
    linear_scheme, embed_scheme = _parse_compound_scheme(scheme)
    tensor_schemes = obj.get("tensor_schemes", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})

    out: dict[str, Tensor] = {}

    for name, q_data in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        eff_scheme = tensor_schemes.get(name)
        if eff_scheme is None:
            eff_scheme = embed_scheme if _is_embedding_tensor(name) else linear_scheme
        out[name] = _dequantize_single_tensor(name, q_data, dtype, eff_scheme, obj)

    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


def quantize_one(
    model: torch.nn.Module,
    state_dict: dict[str, Tensor],
    composer_config: dict | None,
    scheme: str,
    output_path: Path,
    hessians: dict[str, Tensor] | None = None,
) -> dict[str, Any]:
    """Quantize at a single scheme. Returns report dict."""
    logger.info(f"[bold]Quantizing to {scheme}[/]")
    quant_obj, quant_stats = quantize_state_dict(state_dict, scheme=scheme, hessians=hessians)

    if composer_config is not None:
        quant_obj["__composer_config__"] = composer_config

    # Serialize + compress
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = compress_bytes(quant_raw)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(quant_blob)

    file_bytes = output_path.stat().st_size

    logger.info(
        f"[bold green]{scheme} ({_COMPRESSOR}):[/]"
        f" [cyan]compressed[/]={file_bytes:,} bytes"
        f" [cyan]serialized[/]={len(quant_raw):,} bytes"
        f" [cyan]payload[/]={quant_stats['quantized_payload_bytes']:,} bytes"
    )

    # Verify roundtrip
    with open(output_path, "rb") as f:
        verify_blob = f.read()
    verify_state = torch.load(
        io.BytesIO(decompress_bytes(verify_blob)),
        map_location="cpu",
        weights_only=False,
    )
    dequantized = dequantize_state_dict(verify_state)
    model.load_state_dict(dequantized, strict=True)
    logger.info(f"[bold green]{scheme} roundtrip verified[/]")

    return {
        "scheme": scheme,
        "compressor": _COMPRESSOR,
        "compressed_bytes": file_bytes,
        "serialized_bytes": len(quant_raw),
        "param_count": quant_stats["param_count"],
    }


@click.command()
@click.option(
    "--checkpoint",
    default=None,
    type=click.Path(),
    help="Path to checkpoint directory or .tar.gz (default: artifacts/checkpoint).",
)
@click.option(
    "--step",
    default=None,
    type=int,
    help="Specific checkpoint step to load (latest if omitted).",
)
@click.option(
    "--schemes",
    default="int8",
    type=str,
    help="Quantization scheme(s), comma-separated (default: int8). "
    "Supported: int4-int8, mxfp4, nvfp4, turboquipN[c][r] (e.g. turboquip4, turboquip3cr).",
)
@click.option(
    "--output-dir",
    "output_dir",
    default=None,
    type=click.Path(),
    help="Directory to write quantized files (default: same as checkpoint).",
)
@click.option(
    "--calibration-data",
    "calibration_data",
    default=None,
    type=str,
    help="Glob pattern for calibration data (uint16 .bin files). "
    "Required for turboquipNc schemes.",
)
@click.option(
    "--calibration-seqs",
    "calibration_seqs",
    default=128,
    type=int,
    help="Number of calibration sequences for Hessian capture (default: 128).",
)
def main(
    checkpoint: str | None,
    step: int | None,
    schemes: str,
    output_dir: str | None,
    calibration_data: str | None,
    calibration_seqs: int,
):
    """Quantize a Composer checkpoint."""
    setup_logging()

    # Parse schemes
    scheme_list = [s.strip() for s in schemes.split(",")]
    needs_calibration = False
    for s in scheme_list:
        linear_s, embed_s = _parse_compound_scheme(s)
        for part in (linear_s, embed_s):
            if not _is_valid_single_scheme(part):
                raise click.BadParameter(
                    f"Unsupported scheme component: {part!r} (from {s!r}). "
                    f"Valid: {sorted(SUPPORTED_SCHEMES)} or turboquipN[c][r]"
                )
            tq = _parse_turboquip_scheme(part)
            if tq is not None and tq[1]:
                needs_calibration = True

    if needs_calibration and calibration_data is None:
        raise click.BadParameter(
            "Calibrated TurboQuIP schemes (turboquipNc) require --calibration-data"
        )

    # Resolve checkpoint
    tmp_dir = None
    tar_source_dir = None
    if checkpoint is None:
        checkpoint_path = Path("artifacts/checkpoint")
        if not checkpoint_path.exists():
            for candidate in [Path("artifacts/checkpoint.tar.gz"), Path("checkpoint.tar.gz")]:
                if candidate.exists():
                    checkpoint_path = candidate
                    break
            else:
                raise click.BadParameter(
                    "No checkpoint found. Looked for artifacts/checkpoint/"
                    " and artifacts/checkpoint.tar.gz"
                )
    else:
        checkpoint_path = Path(checkpoint)

    if not checkpoint_path.exists():
        raise click.BadParameter(f"Checkpoint not found: {checkpoint_path}")

    # If directory has only quant files, fall back to tar.gz
    if checkpoint_path.is_dir():
        full_candidates = [
            p
            for p in checkpoint_path.glob("step_*")
            if not re.search(r"_(int\d+|mxfp\d+|nvfp\d+|turboquip\d+\w*)$", p.name)
        ]
        if not full_candidates:
            tar_fallback = checkpoint_path.parent / "checkpoint.tar.gz"
            if tar_fallback.exists():
                checkpoint_path = tar_fallback
            else:
                raise click.BadParameter(
                    f"No full-precision checkpoint in {checkpoint_path}"
                    " and no checkpoint.tar.gz found"
                )

    # Extract tar.gz if needed
    if checkpoint_path.suffix == ".gz" and checkpoint_path.name.endswith(".tar.gz"):
        logger.info(f"[bold]Extracting compressed checkpoint:[/] {checkpoint_path}")
        tar_source_dir = checkpoint_path.parent
        tmp_dir = tempfile.mkdtemp(prefix="quant_checkpoint_")
        with tarfile.open(checkpoint_path, "r:gz") as tar:
            tar.extractall(tmp_dir)
        extracted = Path(tmp_dir) / "checkpoint"
        if not extracted.exists():
            extracted = Path(tmp_dir)
        checkpoint_path = extracted

    # Resolve the actual checkpoint file
    resolved_checkpoint = checkpoint_path
    if checkpoint_path.is_dir():
        candidates = [
            p
            for p in sorted(checkpoint_path.glob("step_*"))
            if not re.search(r"_(int\d+|mxfp\d+|nvfp\d+|turboquip\d+\w*)$", p.name)
        ]
        if not candidates:
            raise click.BadParameter(f"No full-precision checkpoint found in: {checkpoint_path}")
        resolved_checkpoint = max(candidates, key=lambda p: int(p.name.split("_")[1]))

    # Output directory
    if output_dir is not None:
        out_dir = Path(output_dir)
    elif tar_source_dir is not None:
        out_dir = tar_source_dir / "checkpoint"
    else:
        out_dir = resolved_checkpoint.parent

    # Load model once
    logger.info(f"[bold]Loading checkpoint:[/] {resolved_checkpoint}")
    model = Model.from_checkpoint(checkpoint_path, device="cpu", max_step=step)
    state_dict = model.state_dict()
    composer_config = state_dict.pop("__composer_config__", None)

    # Capture Hessians for calibrated schemes
    hessians = None
    if needs_calibration:
        cal_device = "cuda" if torch.cuda.is_available() else "cpu"
        model_cal = model.to(cal_device)
        cal_tokens = _load_calibration_tokens(
            calibration_data, max_tokens=calibration_seqs * 1024 + 1
        )
        seq_len = 1024
        if composer_config and "context_length" in str(composer_config):
            # Try to extract context_length from config
            try:
                arch_cfg = list(composer_config.values())[0]
                if isinstance(arch_cfg, dict):
                    seq_len = arch_cfg.get("context_length", seq_len)
            except Exception:
                pass
        hessians = _capture_layer_hessians(
            model_cal, cal_tokens, seq_len=seq_len, device=cal_device
        )
        model_cal.cpu()
        if cal_device == "cuda":
            torch.cuda.empty_cache()

    # Quantize each scheme
    quant = {}
    for scheme in scheme_list:
        out_path = out_dir / f"{resolved_checkpoint.name}_{scheme}"
        lin_s, emb_s = _parse_compound_scheme(scheme)
        tq_lin = _parse_turboquip_scheme(lin_s)
        tq_emb = _parse_turboquip_scheme(emb_s)
        needs_h = (tq_lin is not None and tq_lin[1]) or (tq_emb is not None and tq_emb[1])
        scheme_hessians = hessians if needs_h else None
        result = quantize_one(
            model, state_dict, composer_config, scheme, out_path, hessians=scheme_hessians
        )
        quant[scheme] = result

    # Merge into existing quant report (preserve results from previous runs)
    report_dir = out_dir.parent if out_dir.name == "checkpoint" else out_dir
    report_path = report_dir / "quant_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
        report.setdefault("quant", {}).update(quant)
    else:
        report = {"quant": quant}
    report["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved to {report_path}")

    # Clean up
    if tmp_dir is not None:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
