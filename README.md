# FlipZip

**Involution-aware compression for regime-switching time series**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

FlipZip is a compression algorithm designed for time series that exhibit regime-switching behavior—signals where the underlying statistical structure changes at discrete transition points. Examples include:

- Financial returns (volatility regimes)
- Wind turbine telemetry (operational modes)
- ECG signals (normal vs. arrhythmic states)

The core insight is that regime transitions manifest as sparsity discontinuities in Walsh-Hadamard Transform (WHT) coefficients. FlipZip detects these transitions and adapts its encoding accordingly.

## Important Notice

**This is a preliminary study with honest, reproducible benchmarks.** After identifying inflated claims in an earlier draft, this project was rebuilt from scratch with rigorous methodology:

- **Real results**: ~5-7% compression improvement on regime-switching signals (not 15-22%)
- **Real detection**: 57% average F1 with adaptive method (not 88%)
- **Fair comparisons**: All methods use the same quantization level
- **Clear limitations**: Documented what works and what doesn't

FlipZip is a domain-specific tool that shows promise for certain signal types, not a universal compression improvement.

## Key Features

- **Walsh-Hadamard Transform** basis for efficient frequency-like decomposition
- **Parameter-free seam detection** using robust MAD-based threshold
- **Involution-aware encoding** that tracks basis switches with minimal overhead
- **O(N log N) complexity** via fast dyadic WHT

## Installation

```bash
git clone https://github.com/macmayo/flipzip.git
cd flipzip
pip install -r requirements.txt
```

## Quick Start

### Basic Seam Detection

```python
import numpy as np
from flipzip import detect_seams, FlipZipCompressor

# Generate a signal with regime transition
signal = np.concatenate([
    np.sin(np.linspace(0, 4*np.pi, 256)) + 0.05 * np.random.randn(256),
    np.cos(np.linspace(0, 4*np.pi, 256)) + 0.05 * np.random.randn(256)
])

# Detect seams
positions, tau_values, seams = detect_seams(signal, window_size=64)
print(f"Detected seams at: {seams}")
```

### Full Compression with Entropy Coding

```python
from flipzip import FlipZipCompressor

# Create compressor
compressor = FlipZipCompressor(window_size=256, quantization_bits=10)

# Compress to actual bitstream
compressed_data = compressor.compress_to_bytes(signal)
print(f"Compressed size: {len(compressed_data)} bytes")

# Decompress (round-trip)
reconstructed = compressor.decompress_from_bytes(compressed_data)

# Calculate actual compression metrics
ratio = compressor.compression_ratio(signal)
bps = compressor.bits_per_sample(signal, use_actual=True)
print(f"Compression ratio: {ratio:.2f}x")
print(f"Bits per sample: {bps:.2f}")
```

### Save/Load Compressed Files

```python
# Write compressed data to disk
with open('signal.flpz', 'wb') as f:
    f.write(compressed_data)

# Load and decompress
with open('signal.flpz', 'rb') as f:
    loaded_data = f.read()
reconstructed = compressor.decompress_from_bytes(loaded_data)
```

### Enhanced Detection Methods

```python
from flipzip import FlipZipEnhanced, detect_seams_adaptive

# Auto-select best detection method
enhanced = FlipZipEnhanced(detection_method='auto')
seams, method_used = enhanced.detect_seams(signal)
print(f"Method: {method_used}, Seams: {seams}")

# Or use adaptive consensus across multiple methods
results, consensus_seams = detect_seams_adaptive(signal)
print(f"Consensus seams: {consensus_seams}")
```

## Running Benchmarks

### Synthetic Experiments

```bash
cd benchmarks
python synthetic_benchmark.py      # Basic seam detection tests
python fair_comparison.py          # Compression vs LZMA/GZIP/zstd
python enhanced_benchmark.py       # Compare tau vs period vs wavelet
python ablation_study.py           # PELT vs wavelet analysis
```

Results are saved to JSON files for reproducibility.

### Detection Method Comparison (January 2026)

| Signal Type | Tau (WHT) | Period | Wavelet | Adaptive |
|-------------|-----------|--------|---------|----------|
| ECG-like (rate change) | 0.20 | 0.00 | **0.36** | 0.36 |
| Frequency modulation | 0.33 | **0.67** | 0.31 | 0.40 |
| Amplitude change | 0.33 | 0.67 | 0.31 | **1.00** |
| Piecewise constant | 0.33 | 0.00 | **1.00** | 0.50 |
| **Average F1** | 0.30 | 0.33 | 0.49 | **0.57** |

**Key findings:**
- **Adaptive method** (combining tau + wavelet) achieves best overall performance
- **Wavelet detail** is best for abrupt changes (piecewise constant)
- **Period tracking** is best for pure frequency modulation
- **Tau (WHT sparsity)** works but has lower precision than specialized methods

### Compression Benchmark (Fair comparison at 10-bit quantization)

| Signal Type | FlipZip vs LZMA |
|-------------|-----------------|
| Regime-switching (freq/amp changes) | **+5-7%** (better) |
| White noise | **+17%** (better) |
| Pure sine | -10% (worse) |
| Random walk | -18% (worse) |

**Honest assessment:** FlipZip shows modest but real improvements on specific signal types. This is a domain-specific tool, not a universal compression improvement over LZMA.

**Important limitations:**
- WHT sparsity (tau) alone fails on quasi-periodic signals (e.g., ECG) where rate changes don't alter sparsity patterns
- Performance is worse than LZMA on structured/quasi-periodic signals
- Compression works best for sparse and regime-switching signals

## Algorithm Overview

### Sparsity Statistic τ

FlipZip uses a parameter-free sparsity statistic:

```
τ(x) = (# coefficients above θ) / N
```

where the threshold θ is computed adaptively:

```
θ = median(|X|) + 3 × MAD(|X|)
```

This is scale-invariant and robust to outliers.

### Seam Detection

Multiple detection methods are available to address different signal types:

1. **Tau (WHT sparsity)**: Original method - detects changes in WHT coefficient sparsity. Works well for structural changes.

2. **Period tracking**: Uses autocorrelation to detect rate/frequency changes. Best for oscillatory signals where period changes (e.g., heart rate variability).

3. **Wavelet detail**: Detects sharp transitions via spikes in wavelet coefficients. Excellent for abrupt changes (e.g., piecewise constant signals).

4. **Adaptive method**: Combines multiple detectors with consensus voting. Best overall performance (57% avg F1).

The enhanced methods were developed to address the limitation that WHT sparsity fails on quasi-periodic signals.

### Compression

FlipZip encodes signals window-by-window with full entropy coding:

1. **Transform**: Apply WHT to each window
2. **Quantize**: Uniform quantization of coefficients
3. **Detect seams**: Track regime transitions (tau changes)
4. **Entropy code**: Compress quantized coefficients with zlib
5. **Serialize**: Pack into binary bitstream with metadata

**Bitstream Format:**
- Header: Magic bytes (FLPZ), version, window size, quantization bits
- Per-window: Min/max values (float32), seam flags (bit-packed), entropy-coded coefficients
- Compression gain: 2-10x over theoretical estimates (signal-dependent)

**Decompression:**
- Parse header and metadata
- Decode entropy-coded coefficients
- Dequantize and apply inverse WHT
- Concatenate windows and truncate to original length

**Actual Compression Results (v0.3.0):**
- Clean sine wave: 12.5x compression, 5.1 bits/sample
- Regime-switching signal: 10.1x compression, 6.4 bits/sample
- Sparse signal: 43.6x compression, 1.5 bits/sample
- Round-trip verified with MSE < 0.001

## Project Status

**Current version: 0.3.0 (beta)**

- ✅ Core WHT algorithm implemented
- ✅ Multiple detection methods (tau, period, wavelet, adaptive)
- ✅ Fair baseline comparisons (same quantization)
- ✅ Comprehensive benchmarks with honest results
- ✅ Adaptive method achieves 0.57 avg F1 on detection
- ✅ **Full entropy coding implemented** (v0.3.0)
  - Actual compressed bitstreams (not estimates)
  - Round-trip compression/decompression verified
  - File I/O support (.flpz format)
  - 10-40x compression on suitable signals
- 🚧 Real data validation with MIT-BIH (pending network access)

## Citation

If you use FlipZip in your research, please cite:

```bibtex
@misc{mayo2026flipzip,
  author = {Mayo, Mac},
  title = {FlipZip: Exploiting Walsh-Hadamard Involution Structure for Regime-Switching Compression},
  year = {2026},
  howpublished = {\url{https://github.com/macmayo/flipzip}}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Author

Mac Mayo  
Battleboro, North Carolina  
Independent Researcher

## Acknowledgments

This work builds on concepts from:
- Walsh-Hadamard Transform theory (Beauchamp, 1984)
- Arithmetic coding (Witten, Moffat & Bell, 1999)
- The M³ framework for topological signal analysis
