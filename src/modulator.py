import numpy as np
from typing import Union
from .utils import normalize_power

class Modulator:
    def __init__(self, sample_rate: float = 1e6):
        self.sample_rate = sample_rate
        
    def bpsk_modulate(self, bits: np.ndarray, carrier_freq: float, 
                     symbol_rate: float) -> np.ndarray:
        """BPSK modulation."""
        samples_per_symbol = int(self.sample_rate / symbol_rate)
        t = np.arange(0, len(bits) * samples_per_symbol) / self.sample_rate
        
        # Map bits to symbols: 0 -> 1, 1 -> -1
        symbols = 1 - 2 * bits
        
        # Upsample symbols
        upsampled = np.zeros(len(bits) * samples_per_symbol, dtype=complex)
        upsampled[::samples_per_symbol] = symbols
        
        # Apply pulse shaping (rectangular for simplicity)
        # In real systems, you'd use a raised cosine or similar filter
        pulse = np.ones(samples_per_symbol)
        shaped = np.convolve(upsampled, pulse, mode='same')
        
        # Generate carrier
        carrier = np.exp(1j * 2 * np.pi * carrier_freq * t)
        
        # Modulate
        modulated = shaped * carrier
        
        return normalize_power(modulated)
    
    def qam_modulate(self, symbols: np.ndarray, carrier_freq: float, 
                    symbol_rate: float) -> np.ndarray:
        """QAM modulation."""
        samples_per_symbol = int(self.sample_rate / symbol_rate)
        t = np.arange(0, len(symbols) * samples_per_symbol) / self.sample_rate
        
        # Upsample symbols
        upsampled = np.zeros(len(symbols) * samples_per_symbol, dtype=complex)
        upsampled[::samples_per_symbol] = symbols
        
        # Apply pulse shaping
        pulse = np.ones(samples_per_symbol)
        shaped = np.convolve(upsampled, pulse, mode='same')
        
        # Generate carrier
        carrier = np.exp(1j * 2 * np.pi * carrier_freq * t)
        
        # Modulate
        modulated = shaped * carrier
        
        return normalize_power(modulated)
    
    def ofdm_modulate(self, ofdm_symbols: np.ndarray, carrier_freq: float) -> np.ndarray:
        """OFDM modulation."""
        # This is a simplified version - in real OFDM, the IFFT is already done
        # and we're just upconverting to the carrier frequency
        t = np.arange(0, len(ofdm_symbols)) / self.sample_rate
        carrier = np.exp(1j * 2 * np.pi * carrier_freq * t)
        modulated = ofdm_symbols * carrier
        
        return normalize_power(modulated)