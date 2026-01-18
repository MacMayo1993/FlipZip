#!/usr/bin/env python3
"""
Quick round-trip test for FlipZip entropy coding.
Tests compression and decompression without pytest dependency.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/user/FlipZip')

from flipzip import FlipZipCompressor


def test_basic_roundtrip():
    """Test basic round-trip compression."""
    print("Testing basic round-trip compression...")

    # Create a simple sine wave
    t = np.linspace(0, 4 * np.pi, 2048)
    signal = np.sin(t)

    print(f"  Original signal: {len(signal)} samples, {signal.nbytes} bytes")

    # Create compressor
    compressor = FlipZipCompressor(window_size=256, quantization_bits=10)

    # Compress
    compressed = compressor.compress_to_bytes(signal)
    print(f"  Compressed size: {len(compressed)} bytes")

    # Decompress
    reconstructed = compressor.decompress_from_bytes(compressed)
    print(f"  Reconstructed: {len(reconstructed)} samples")

    # Check
    assert len(reconstructed) == len(signal), "Length mismatch!"

    # Calculate error
    mse = np.mean((signal - reconstructed) ** 2)
    max_error = np.max(np.abs(signal - reconstructed))

    print(f"  MSE: {mse:.6f}")
    print(f"  Max error: {max_error:.6f}")

    # Calculate metrics
    bps = compressor.bits_per_sample(signal, use_actual=True)
    ratio = compressor.compression_ratio(signal)

    print(f"  Bits per sample: {bps:.2f} (original: 64.0)")
    print(f"  Compression ratio: {ratio:.2f}x")

    # Verify reasonable reconstruction
    assert mse < 0.01, f"MSE too high: {mse}"
    assert max_error < 0.2, f"Max error too high: {max_error}"
    assert ratio > 1.0, f"No compression achieved: {ratio}"

    print("  ✓ PASSED")


def test_regime_switching():
    """Test on regime-switching signal."""
    print("\nTesting regime-switching signal...")

    # Create signal with two regimes
    t = np.linspace(0, 10, 4096)
    signal = np.concatenate([
        np.sin(2 * np.pi * 1 * t[:2048]),  # Slow oscillation
        np.sin(2 * np.pi * 10 * t[2048:])  # Fast oscillation
    ])

    print(f"  Original signal: {len(signal)} samples")

    compressor = FlipZipCompressor(window_size=256, quantization_bits=10)

    compressed = compressor.compress_to_bytes(signal)
    reconstructed = compressor.decompress_from_bytes(compressed)

    print(f"  Compressed size: {len(compressed)} bytes")

    mse = np.mean((signal - reconstructed) ** 2)
    print(f"  MSE: {mse:.6f}")

    assert len(reconstructed) == len(signal)
    assert mse < 0.01

    print("  ✓ PASSED")


def test_sparse_signal():
    """Test on sparse signal (should compress very well)."""
    print("\nTesting sparse signal compression...")

    # Mostly zeros
    signal = np.zeros(4096)
    signal[::100] = 1.0

    print(f"  Original signal: {len(signal)} samples (sparse)")

    compressor = FlipZipCompressor(window_size=256, quantization_bits=8)

    compressed = compressor.compress_to_bytes(signal)
    reconstructed = compressor.decompress_from_bytes(compressed)

    ratio = compressor.compression_ratio(signal)

    print(f"  Compressed size: {len(compressed)} bytes")
    print(f"  Compression ratio: {ratio:.2f}x")

    assert len(reconstructed) == len(signal)
    # Sparse signal should compress very well
    assert ratio > 5.0, f"Expected high compression for sparse signal, got {ratio}x"

    print("  ✓ PASSED")


def test_various_lengths():
    """Test with various signal lengths."""
    print("\nTesting various signal lengths...")

    for length in [100, 500, 1000, 2000, 5000]:
        signal = np.sin(np.linspace(0, 2 * np.pi, length))

        compressor = FlipZipCompressor(window_size=256, quantization_bits=10)

        compressed = compressor.compress_to_bytes(signal)
        reconstructed = compressor.decompress_from_bytes(compressed)

        assert len(reconstructed) == length, f"Length mismatch for {length} samples"

        mse = np.mean((signal - reconstructed) ** 2)
        assert mse < 0.01, f"High error for {length} samples: MSE={mse}"

        print(f"  Length {length}: {len(compressed)} bytes, MSE={mse:.6f} ✓")

    print("  ✓ ALL PASSED")


def main():
    """Run all tests."""
    print("=" * 60)
    print("FlipZip Entropy Coding - Round-trip Verification Tests")
    print("=" * 60)

    try:
        test_basic_roundtrip()
        test_regime_switching()
        test_sparse_signal()
        test_various_lengths()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print("\nEntropy coding implementation is working correctly:")
        print("  • Bitstream serialization ✓")
        print("  • Round-trip compression/decompression ✓")
        print("  • Actual compression ratios measured ✓")
        print("  • Lossless reconstruction (within quantization) ✓")

        return 0

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
