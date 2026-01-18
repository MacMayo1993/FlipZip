"""
Tests for FlipZip entropy coding and round-trip compression.

Verifies that:
1. Bitstream serialization works correctly
2. Round-trip compression preserves data (within quantization error)
3. Actual compression produces valid bitstreams
4. Compression ratios are measurable and realistic
"""

import pytest
import numpy as np
from flipzip import FlipZipCompressor
from flipzip.entropy import (
    encode_flipzip,
    decode_flipzip,
    BitstreamWriter,
    BitstreamReader,
    encode_array_simple,
    decode_array_simple,
)


class TestBitstreamIO:
    """Test basic bitstream read/write operations."""

    def test_write_read_bits(self):
        """Test writing and reading bits."""
        writer = BitstreamWriter()
        writer.write_bits(0b1010, 4)
        writer.write_bits(0b11, 2)
        writer.write_bits(0b101010, 6)

        data = writer.getvalue()
        reader = BitstreamReader(data)

        assert reader.read_bits(4) == 0b1010
        assert reader.read_bits(2) == 0b11
        assert reader.read_bits(6) == 0b101010

    def test_write_read_bytes(self):
        """Test writing and reading raw bytes."""
        writer = BitstreamWriter()
        test_bytes = b"Hello, FlipZip!"
        writer.write_bytes(test_bytes)

        data = writer.getvalue()
        reader = BitstreamReader(data)

        assert reader.read_bytes(len(test_bytes)) == test_bytes

    def test_mixed_bits_and_bytes(self):
        """Test mixing bit-level and byte-level operations."""
        writer = BitstreamWriter()
        writer.write_bits(0b1010, 4)
        writer.write_bytes(b"ABC")
        writer.write_bits(0b11, 2)

        data = writer.getvalue()
        reader = BitstreamReader(data)

        assert reader.read_bits(4) == 0b1010
        assert reader.read_bytes(3) == b"ABC"
        assert reader.read_bits(2) == 0b11


class TestArrayEncoding:
    """Test array compression with zlib."""

    def test_encode_decode_small_array(self):
        """Test encoding/decoding a small array."""
        arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        encoded = encode_array_simple(arr)
        decoded = decode_array_simple(encoded, len(arr))

        np.testing.assert_array_equal(decoded, arr)

    def test_encode_decode_sparse_array(self):
        """Test encoding/decoding a sparse array (should compress well)."""
        arr = np.zeros(1000, dtype=np.int32)
        arr[10] = 100
        arr[500] = 200
        arr[999] = 300

        encoded = encode_array_simple(arr)
        decoded = decode_array_simple(encoded, len(arr))

        np.testing.assert_array_equal(decoded, arr)

        # Sparse array should compress significantly
        original_bytes = arr.nbytes
        compressed_bytes = len(encoded)
        assert compressed_bytes < original_bytes * 0.1  # At least 10x compression

    def test_encode_decode_large_values(self):
        """Test encoding/decoding arrays with large values."""
        arr = np.array([0, 255, 65535, 100000], dtype=np.int32)
        encoded = encode_array_simple(arr)
        decoded = decode_array_simple(encoded, len(arr))

        np.testing.assert_array_equal(decoded, arr)


class TestFlipZipRoundTrip:
    """Test full round-trip compression and decompression."""

    def test_constant_signal(self):
        """Test round-trip on a constant signal."""
        signal = np.ones(1024) * 5.0
        compressor = FlipZipCompressor(window_size=256, quantization_bits=8)

        # Full round-trip
        compressed = compressor.compress_to_bytes(signal)
        reconstructed = compressor.decompress_from_bytes(compressed)

        # Should be close (within quantization error)
        assert len(reconstructed) == len(signal)
        np.testing.assert_allclose(reconstructed, signal, rtol=0.01, atol=0.1)

    def test_sine_wave(self):
        """Test round-trip on a sine wave."""
        t = np.linspace(0, 4 * np.pi, 2048)
        signal = np.sin(t)

        compressor = FlipZipCompressor(window_size=256, quantization_bits=10)

        compressed = compressor.compress_to_bytes(signal)
        reconstructed = compressor.decompress_from_bytes(compressed)

        assert len(reconstructed) == len(signal)
        np.testing.assert_allclose(reconstructed, signal, rtol=0.05, atol=0.02)

    def test_regime_switching_signal(self):
        """Test round-trip on a signal with regime switches."""
        # Low frequency + high frequency regime
        t = np.linspace(0, 10, 4096)
        signal = np.concatenate([
            np.sin(2 * np.pi * 1 * t[:2048]),  # Slow
            np.sin(2 * np.pi * 10 * t[2048:])  # Fast
        ])

        compressor = FlipZipCompressor(window_size=256, quantization_bits=10)

        compressed = compressor.compress_to_bytes(signal)
        reconstructed = compressor.decompress_from_bytes(compressed)

        assert len(reconstructed) == len(signal)
        np.testing.assert_allclose(reconstructed, signal, rtol=0.05, atol=0.02)

    def test_random_signal(self):
        """Test round-trip on random noise."""
        np.random.seed(42)
        signal = np.random.randn(2048)

        compressor = FlipZipCompressor(window_size=256, quantization_bits=10)

        compressed = compressor.compress_to_bytes(signal)
        reconstructed = compressor.decompress_from_bytes(compressed)

        assert len(reconstructed) == len(signal)
        # Random noise has high quantization error, so be lenient
        np.testing.assert_allclose(reconstructed, signal, rtol=0.1, atol=0.2)

    def test_different_window_sizes(self):
        """Test round-trip with different window sizes."""
        signal = np.sin(np.linspace(0, 2 * np.pi, 1024))

        for window_size in [64, 128, 256, 512]:
            compressor = FlipZipCompressor(window_size=window_size, quantization_bits=8)

            compressed = compressor.compress_to_bytes(signal)
            reconstructed = compressor.decompress_from_bytes(compressed)

            assert len(reconstructed) == len(signal)
            np.testing.assert_allclose(reconstructed, signal, rtol=0.05, atol=0.02)

    def test_different_quantization_bits(self):
        """Test round-trip with different quantization levels."""
        signal = np.sin(np.linspace(0, 2 * np.pi, 1024))

        for quant_bits in [6, 8, 10, 12]:
            compressor = FlipZipCompressor(window_size=256, quantization_bits=quant_bits)

            compressed = compressor.compress_to_bytes(signal)
            reconstructed = compressor.decompress_from_bytes(compressed)

            assert len(reconstructed) == len(signal)
            # Higher quantization should have lower error
            tolerance = 0.1 / (2 ** (quant_bits - 6))
            np.testing.assert_allclose(reconstructed, signal, rtol=tolerance, atol=tolerance)

    def test_non_power_of_two_length(self):
        """Test round-trip with signal length not a power of 2."""
        signal = np.sin(np.linspace(0, 2 * np.pi, 1000))  # Not power of 2

        compressor = FlipZipCompressor(window_size=256, quantization_bits=10)

        compressed = compressor.compress_to_bytes(signal)
        reconstructed = compressor.decompress_from_bytes(compressed)

        # Should preserve exact original length
        assert len(reconstructed) == 1000
        np.testing.assert_allclose(reconstructed, signal, rtol=0.05, atol=0.02)

    def test_short_signal(self):
        """Test round-trip on very short signals."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        compressor = FlipZipCompressor(window_size=64, quantization_bits=8)

        compressed = compressor.compress_to_bytes(signal)
        reconstructed = compressor.decompress_from_bytes(compressed)

        assert len(reconstructed) == len(signal)
        np.testing.assert_allclose(reconstructed, signal, rtol=0.05, atol=0.1)


class TestCompressionMetrics:
    """Test compression ratio and bits-per-sample calculations."""

    def test_bits_per_sample_actual(self):
        """Test actual bits-per-sample calculation."""
        signal = np.sin(np.linspace(0, 2 * np.pi, 1024))
        compressor = FlipZipCompressor(window_size=256, quantization_bits=10)

        bps_actual = compressor.bits_per_sample(signal, use_actual=True)
        bps_estimate = compressor.bits_per_sample(signal, use_actual=False)

        # Actual should be less than estimate (due to entropy coding)
        assert bps_actual < bps_estimate
        assert bps_actual > 0
        assert bps_actual < 64  # Less than original float64

    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        signal = np.sin(np.linspace(0, 2 * np.pi, 1024))
        compressor = FlipZipCompressor(window_size=256, quantization_bits=10)

        ratio = compressor.compression_ratio(signal)

        # Should achieve some compression (ratio > 1)
        assert ratio > 1.0
        # But not impossible compression
        assert ratio < 100.0

    def test_sparse_signal_compresses_better(self):
        """Test that sparse signals achieve better compression."""
        # Sparse signal (mostly zeros)
        sparse = np.zeros(2048)
        sparse[::100] = 1.0

        # Dense signal (random)
        np.random.seed(42)
        dense = np.random.randn(2048)

        compressor = FlipZipCompressor(window_size=256, quantization_bits=10)

        bps_sparse = compressor.bits_per_sample(sparse, use_actual=True)
        bps_dense = compressor.bits_per_sample(dense, use_actual=True)

        # Sparse should compress better
        assert bps_sparse < bps_dense


class TestEncodedDataFormat:
    """Test the encoded data format and structure."""

    def test_encoded_dict_structure(self):
        """Test that encode() returns correct dictionary structure."""
        signal = np.sin(np.linspace(0, 2 * np.pi, 512))
        compressor = FlipZipCompressor(window_size=256, quantization_bits=8)

        encoded = compressor.encode(signal)

        # Check structure
        assert 'windows' in encoded
        assert 'seam_flags' in encoded
        assert 'original_length' in encoded
        assert 'window_size' in encoded

        # Check values
        assert encoded['original_length'] == 512
        assert encoded['window_size'] == 256
        assert len(encoded['windows']) == 2  # 512 / 256 = 2 windows
        assert len(encoded['seam_flags']) == 2

        # Check window structure
        for window in encoded['windows']:
            assert 'quantized' in window
            assert 'min' in window
            assert 'max' in window
            assert len(window['quantized']) == 256

    def test_bitstream_has_magic_header(self):
        """Test that compressed bitstream has magic bytes."""
        signal = np.sin(np.linspace(0, 2 * np.pi, 256))
        compressor = FlipZipCompressor(window_size=256, quantization_bits=8)

        compressed = compressor.compress_to_bytes(signal)

        # Should start with magic bytes "FLPZ"
        assert compressed[:4] == b'FLPZ'

    def test_decode_invalid_magic(self):
        """Test that decoding fails with invalid magic bytes."""
        bad_data = b'BADM' + b'\x00' * 100

        compressor = FlipZipCompressor(window_size=256, quantization_bits=8)

        with pytest.raises(ValueError, match="Invalid magic bytes"):
            compressor.decompress_from_bytes(bad_data)

    def test_compressed_size_reasonable(self):
        """Test that compressed size is reasonable."""
        signal = np.sin(np.linspace(0, 2 * np.pi, 1024))
        compressor = FlipZipCompressor(window_size=256, quantization_bits=10)

        compressed = compressor.compress_to_bytes(signal)
        original_bytes = signal.nbytes  # 1024 * 8 = 8192 bytes (float64)

        # Compressed should be significantly smaller
        assert len(compressed) < original_bytes
        # But not impossibly small
        assert len(compressed) > 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
