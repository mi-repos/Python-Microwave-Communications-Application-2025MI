import numpy as np
from typing import Union, List, Tuple

class Demodulator:
    def __init__(self, sample_rate: float = 1e6):
        self.sample_rate = sample_rate
        
    def bpsk_demodulate(self, signal: np.ndarray, carrier_freq: float, 
                       symbol_rate: float) -> np.ndarray:
        """BPSK demodulation."""
        samples_per_symbol = int(self.sample_rate / symbol_rate)
        t = np.arange(0, len(signal)) / self.sample_rate
        
        # Downconvert
        carrier = np.exp(-1j * 2 * np.pi * carrier_freq * t)
        baseband = signal * carrier
        
        # Low-pass filter (simple moving average)
        filtered = np.convolve(baseband, np.ones(samples_per_symbol)/samples_per_symbol, mode='same')
        
        # Sample at symbol times
        sampled = filtered[samples_per_symbol//2::samples_per_symbol]
        
        # Decision
        bits = (np.real(sampled) < 0).astype(int)
        
        return bits
    
    def qam_demodulate(self, signal: np.ndarray, carrier_freq: float, 
                      symbol_rate: float, modulation_order: int = 4) -> np.ndarray:
        """QAM demodulation."""
        samples_per_symbol = int(self.sample_rate / symbol_rate)
        t = np.arange(0, len(signal)) / self.sample_rate
        
        # Downconvert
        carrier = np.exp(-1j * 2 * np.pi * carrier_freq * t)
        baseband = signal * carrier
        
        # Low-pass filter
        filtered = np.convolve(baseband, np.ones(samples_per_symbol)/samples_per_symbol, mode='same')
        
        # Sample at symbol times
        sampled = filtered[samples_per_symbol//2::samples_per_symbol]
        
        # For simplicity, we'll just return the symbols
        # In a real implementation, you'd map to bits
        return sampled
    
    def ofdm_demodulate(self, signal: np.ndarray, carrier_freq: float, 
                       num_subcarriers: int, cp_ratio: float = 0.25) -> np.ndarray:
        """OFDM demodulation."""
        t = np.arange(0, len(signal)) / self.sample_rate
        
        # Downconvert
        carrier = np.exp(-1j * 2 * np.pi * carrier_freq * t)
        baseband = signal * carrier
        
        # This is a simplified version - in real OFDM, you'd remove CP and do FFT
        # For now, we'll just return the baseband signal
        return baseband