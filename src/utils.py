import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List, Tuple

def db_to_linear(db_value: float) -> float:
    """Convert dB value to linear scale."""
    return 10 ** (db_value / 10)

def linear_to_db(linear_value: float) -> float:
    """Convert linear value to dB scale."""
    return 10 * np.log10(linear_value) if linear_value > 0 else -np.inf

def normalize_power(signal: np.ndarray, power_dbm: float = 0) -> np.ndarray:
    """Normalize signal to specific power in dBm."""
    current_power = np.mean(np.abs(signal) ** 2)
    target_power = db_to_linear(power_dbm - 30)  # Convert dBm to Watts
    scaling_factor = np.sqrt(target_power / current_power)
    return signal * scaling_factor

def calculate_ber(received_bits: np.ndarray, transmitted_bits: np.ndarray) -> float:
    """Calculate Bit Error Rate."""
    if len(received_bits) != len(transmitted_bits):
        raise ValueError("Bit arrays must have the same length")
    
    errors = np.sum(received_bits != transmitted_bits)
    return errors / len(transmitted_bits)

def calculate_evm(received_symbols: np.ndarray, reference_symbols: np.ndarray) -> float:
    """Calculate Error Vector Magnitude."""
    if len(received_symbols) != len(reference_symbols):
        raise ValueError("Symbol arrays must have the same length")
    
    error_vector = received_symbols - reference_symbols
    evm_rms = np.sqrt(np.mean(np.abs(error_vector) ** 2))
    reference_power = np.sqrt(np.mean(np.abs(reference_symbols) ** 2))
    return evm_rms / reference_power

def plot_constellation(symbols: np.ndarray, title: str = "Constellation Diagram") -> None:
    """Plot constellation diagram."""
    plt.figure(figsize=(8, 8))
    plt.scatter(np.real(symbols), np.imag(symbols), alpha=0.6)
    plt.axhline(0, color='black', linestyle='--')
    plt.axvline(0, color='black', linestyle='--')
    plt.grid(True)
    plt.title(title)
    plt.xlabel("In-Phase")
    plt.ylabel("Quadrature")
    plt.axis('equal')
    plt.show()

def plot_spectrum(signal: np.ndarray, sample_rate: float, title: str = "Power Spectrum") -> None:
    """Plot power spectrum of signal."""
    n = len(signal)
    fft_result = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(n, 1/sample_rate)
    fft_power = np.abs(fft_result) ** 2 / n
    
    # Shift zero frequency to center
    fft_freq_shifted = np.fft.fftshift(fft_freq)
    fft_power_shifted = np.fft.fftshift(fft_power)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fft_freq_shifted / 1e6, 10 * np.log10(fft_power_shifted))
    plt.title(title)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Power (dB)")
    plt.grid(True)
    plt.show()