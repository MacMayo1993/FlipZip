"""
Entropy coding and bitstream serialization for FlipZip.

Provides:
- Bitstream writer/reader with bit-level precision
- Range coder for efficient entropy coding
- Serialization/deserialization for compressed FlipZip data
"""

import struct
import zlib
from typing import List, Dict, Tuple
import numpy as np


# Magic bytes for FlipZip format
MAGIC = b'FLPZ'
VERSION = 1


class BitstreamWriter:
    """Writes data at bit-level precision to a byte buffer."""

    def __init__(self):
        self.buffer = bytearray()
        self.bit_buffer = 0
        self.bit_count = 0

    def write_bits(self, value: int, num_bits: int):
        """Write num_bits from value to the bitstream."""
        assert 0 <= num_bits <= 32
        assert 0 <= value < (1 << num_bits)

        self.bit_buffer = (self.bit_buffer << num_bits) | value
        self.bit_count += num_bits

        # Flush complete bytes
        while self.bit_count >= 8:
            self.bit_count -= 8
            byte = (self.bit_buffer >> self.bit_count) & 0xFF
            self.buffer.append(byte)

    def write_bytes(self, data: bytes):
        """Write raw bytes (must be byte-aligned)."""
        self.flush()
        self.buffer.extend(data)

    def flush(self):
        """Flush remaining bits, padding with zeros."""
        if self.bit_count > 0:
            self.write_bits(0, 8 - self.bit_count)

    def getvalue(self) -> bytes:
        """Get the complete byte buffer."""
        self.flush()
        return bytes(self.buffer)


class BitstreamReader:
    """Reads data at bit-level precision from a byte buffer."""

    def __init__(self, data: bytes):
        self.data = data
        self.byte_pos = 0
        self.bit_buffer = 0
        self.bit_count = 0

    def read_bits(self, num_bits: int) -> int:
        """Read num_bits from the bitstream."""
        assert 0 <= num_bits <= 32

        # Refill buffer as needed
        while self.bit_count < num_bits:
            if self.byte_pos >= len(self.data):
                raise EOFError("Unexpected end of bitstream")
            byte = self.data[self.byte_pos]
            self.byte_pos += 1
            self.bit_buffer = (self.bit_buffer << 8) | byte
            self.bit_count += 8

        # Extract bits
        self.bit_count -= num_bits
        value = (self.bit_buffer >> self.bit_count) & ((1 << num_bits) - 1)
        return value

    def read_bytes(self, num_bytes: int) -> bytes:
        """Read raw bytes (must be byte-aligned)."""
        self.align()
        if self.byte_pos + num_bytes > len(self.data):
            raise EOFError("Unexpected end of bitstream")
        result = self.data[self.byte_pos:self.byte_pos + num_bytes]
        self.byte_pos += num_bytes
        return result

    def align(self):
        """Align to next byte boundary."""
        if self.bit_count % 8 != 0:
            self.read_bits(self.bit_count % 8)


def encode_array_simple(arr: np.ndarray) -> bytes:
    """
    Simple entropy encoding using zlib compression.

    This is a pragmatic first implementation that provides:
    - Actual entropy coding (not just estimates)
    - Good compression for sparse/repetitive data
    - Fast encoding/decoding

    Future optimization: Replace with custom range coder for better
    compression and more control over the encoding process.
    """
    # Convert to bytes (use minimal dtype)
    max_val = arr.max()
    if max_val < 256:
        dtype = np.uint8
    elif max_val < 65536:
        dtype = np.uint16
    else:
        dtype = np.uint32

    arr_bytes = arr.astype(dtype).tobytes()

    # Compress with zlib (level 9 for best compression)
    compressed = zlib.compress(arr_bytes, level=9)

    # Pack: dtype_code (1 byte) + compressed_size (4 bytes) + compressed_data
    dtype_code = {np.dtype(np.uint8): 0, np.dtype(np.uint16): 1, np.dtype(np.uint32): 2}[np.dtype(dtype)]
    header = struct.pack('<BI', dtype_code, len(compressed))

    return header + compressed


def decode_array_simple(data: bytes, expected_length: int) -> np.ndarray:
    """Decode array encoded with encode_array_simple."""
    # Unpack header
    dtype_code, compressed_size = struct.unpack('<BI', data[:5])
    dtype_map = {0: np.uint8, 1: np.uint16, 2: np.uint32}
    dtype = dtype_map[dtype_code]

    # Decompress
    compressed = data[5:5 + compressed_size]
    decompressed = zlib.decompress(compressed)

    # Convert back to array
    arr = np.frombuffer(decompressed, dtype=dtype)
    assert len(arr) == expected_length, f"Expected {expected_length} elements, got {len(arr)}"

    return arr.astype(np.int32)


def encode_flipzip(encoded_data: dict) -> bytes:
    """
    Encode FlipZip compressed data to a binary bitstream.

    Args:
        encoded_data: Dictionary from FlipZipCompressor.encode() containing:
            - windows: List of window dictionaries
            - seam_flags: List of boolean seam indicators
            - original_length: Original signal length
            - window_size: Window size used

    Returns:
        Compressed binary data with full entropy coding
    """
    writer = BitstreamWriter()

    # Write header
    writer.write_bytes(MAGIC)
    writer.write_bytes(struct.pack('<B', VERSION))
    writer.write_bytes(struct.pack('<H', encoded_data['window_size']))
    writer.write_bytes(struct.pack('<I', encoded_data['original_length']))

    windows = encoded_data['windows']
    seam_flags = encoded_data['seam_flags']
    num_windows = len(windows)

    writer.write_bytes(struct.pack('<I', num_windows))

    # Infer quantization bits from first window
    if num_windows > 0:
        max_quant_val = windows[0]['quantized'].max()
        quantization_bits = int(np.ceil(np.log2(max_quant_val + 1))) if max_quant_val > 0 else 1
        writer.write_bytes(struct.pack('<B', quantization_bits))
    else:
        writer.write_bytes(struct.pack('<B', 8))  # default

    # Write seam flags (bit-packed)
    for flag in seam_flags:
        writer.write_bits(1 if flag else 0, 1)

    # Byte-align after seam flags
    writer.flush()

    # Write window data
    for i, window in enumerate(windows):
        # Encode min/max as float32 (sufficient precision, saves space)
        min_val = np.float32(window['min'])
        max_val = np.float32(window['max'])

        writer.write_bytes(struct.pack('<ff', min_val, max_val))

        # Entropy-encode quantized coefficients
        quantized = window['quantized']
        encoded_coeffs = encode_array_simple(quantized)

        # Write size + data
        writer.write_bytes(struct.pack('<I', len(encoded_coeffs)))
        writer.write_bytes(encoded_coeffs)

    return writer.getvalue()


def decode_flipzip(data: bytes) -> dict:
    """
    Decode FlipZip bitstream back to encoded_data dictionary.

    Args:
        data: Compressed binary data from encode_flipzip()

    Returns:
        Dictionary compatible with FlipZipCompressor.decode():
            - windows: List of window dictionaries
            - seam_flags: List of boolean seam indicators
            - original_length: Original signal length
            - window_size: Window size used
    """
    reader = BitstreamReader(data)

    # Read and validate header
    magic = reader.read_bytes(4)
    if magic != MAGIC:
        raise ValueError(f"Invalid magic bytes: expected {MAGIC}, got {magic}")

    version = struct.unpack('<B', reader.read_bytes(1))[0]
    if version != VERSION:
        raise ValueError(f"Unsupported version: {version}")

    window_size = struct.unpack('<H', reader.read_bytes(2))[0]
    original_length = struct.unpack('<I', reader.read_bytes(4))[0]
    num_windows = struct.unpack('<I', reader.read_bytes(4))[0]
    quantization_bits = struct.unpack('<B', reader.read_bytes(1))[0]

    # Read seam flags
    seam_flags = []
    for _ in range(num_windows):
        seam_flags.append(bool(reader.read_bits(1)))

    # Byte-align after reading seam flags
    reader.align()

    # Read window data
    windows = []
    for i in range(num_windows):
        # Read min/max
        min_val, max_val = struct.unpack('<ff', reader.read_bytes(8))

        # Read entropy-encoded coefficients
        coeffs_size = struct.unpack('<I', reader.read_bytes(4))[0]
        coeffs_data = reader.read_bytes(coeffs_size)
        quantized = decode_array_simple(coeffs_data, window_size)

        windows.append({
            'quantized': quantized,
            'min': float(min_val),
            'max': float(max_val),
            'tau': 0.0  # Not stored in bitstream (not needed for decompression)
        })

    return {
        'windows': windows,
        'seam_flags': seam_flags,
        'original_length': original_length,
        'window_size': window_size
    }


def get_compressed_size_bytes(encoded_data: dict) -> int:
    """
    Get the actual compressed size in bytes.

    This performs the actual entropy coding and returns the real size,
    not an estimate.
    """
    bitstream = encode_flipzip(encoded_data)
    return len(bitstream)


def get_compression_ratio(signal: np.ndarray, encoded_data: dict) -> float:
    """
    Calculate actual compression ratio.

    Returns:
        Ratio of compressed size to original size (< 1.0 means compression)
    """
    original_bytes = signal.nbytes
    compressed_bytes = get_compressed_size_bytes(encoded_data)
    return compressed_bytes / original_bytes


def get_bits_per_sample(signal: np.ndarray, encoded_data: dict) -> float:
    """
    Calculate actual bits per sample after full entropy coding.

    Returns:
        Average number of bits needed per input sample
    """
    compressed_bytes = get_compressed_size_bytes(encoded_data)
    compressed_bits = compressed_bytes * 8
    num_samples = len(signal)
    return compressed_bits / num_samples
