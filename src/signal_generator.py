import numpy as np
from typing import Union, List
from .utils import normalize_power

class SignalGenerator:
    def __init__(self, sample_rate: float = 1e6):
        self.sample_rate = sample_rate
        
    def generate_carrier(self, frequency: float, duration: float, 
                        amplitude: float = 1.0, phase: float = 0.0) -> np.ndarray:
        """Generate a carrier wave."""
        t = np.arange(0, duration, 1/self.sample_rate)
        return amplitude * np.exp(1j * (2 * np.pi * frequency * t + phase))
    
    def generate_bits(self, num_bits: int) -> np.ndarray:
        """Generate random bits."""
        return np.random.randint(0, 2, num_bits)
    
    def generate_ofdm_signal(self, num_subcarriers: int, num_symbols: int, 
                            bandwidth: float, cp_ratio: float = 0.25) -> np.ndarray:
        """Generate OFDM signal."""
        # Calculate subcarrier spacing
        subcarrier_spacing = bandwidth / num_subcarriers
        
        # Generate random QAM symbols for each subcarrier
        symbols = (np.random.randint(0, 4, (num_symbols, num_subcarriers)) * 2 - 3) / np.sqrt(2)
        symbols = symbols.astype(complex)
        
        ofdm_symbols = []
        for i in range(num_symbols):
            # Perform IFFT
            time_domain = np.fft.ifft(symbols[i, :]) * num_subcarriers
            
            # Add cyclic prefix
            cp_length = int(cp_ratio * num_subcarriers)
            cp = time_domain[-cp_length:]
            ofdm_symbol = np.concatenate([cp, time_domain])
            
            ofdm_symbols.append(ofdm_symbol)
        
        return np.concatenate(ofdm_symbols)
    
    def generate_qam_signal(self, bits: np.ndarray, modulation_order: int = 4) -> np.ndarray:
        """Generate QAM modulated signal from bits."""
        if modulation_order not in [4, 16, 64, 256]:
            raise ValueError("Modulation order must be 4, 16, 64, or 256")
        
        # Map bits to symbols
        bits_per_symbol = int(np.log2(modulation_order))
        if len(bits) % bits_per_symbol != 0:
            raise ValueError(f"Number of bits must be divisible by {bits_per_symbol}")
        
        # Reshape bits into symbols
        symbol_bits = bits.reshape(-1, bits_per_symbol)
        
        # Map to constellation points
        if modulation_order == 4:
            # QPSK
            mapping = {
                (0, 0): (1 + 1j) / np.sqrt(2),
                (0, 1): (1 - 1j) / np.sqrt(2),
                (1, 0): (-1 + 1j) / np.sqrt(2),
                (1, 1): (-1 - 1j) / np.sqrt(2)
            }
        elif modulation_order == 16:
            # 16-QAM
            mapping = {
                (0, 0, 0, 0): (-3 + 3j) / np.sqrt(10),
                (0, 0, 0, 1): (-3 + 1j) / np.sqrt(10),
                (0, 0, 1, 0): (-3 - 3j) / np.sqrt(10),
                (0, 0, 1, 1): (-3 - 1j) / np.sqrt(10),
                (0, 1, 0, 0): (-1 + 3j) / np.sqrt(10),
                (0, 1, 0, 1): (-1 + 1j) / np.sqrt(10),
                (0, 1, 1, 0): (-1 - 3j) / np.sqrt(10),
                (0, 1, 1, 1): (-1 - 1j) / np.sqrt(10),
                (1, 0, 0, 0): (3 + 3j) / np.sqrt(10),
                (1, 0, 0, 1): (3 + 1j) / np.sqrt(10),
                (1, 0, 1, 0): (3 - 3j) / np.sqrt(10),
                (1, 0, 1, 1): (3 - 1j) / np.sqrt(10),
                (1, 1, 0, 0): (1 + 3j) / np.sqrt(10),
                (1, 1, 0, 1): (1 + 1j) / np.sqrt(10),
                (1, 1, 1, 0): (1 - 3j) / np.sqrt(10),
                (1, 1, 1, 1): (1 - 1j) / np.sqrt(10)
            }
        else:
            # For higher orders, use a simple approach
            # In a real implementation, you'd want a proper mapping
            symbols = np.array([(2*(i % int(np.sqrt(modulation_order))) - int(np.sqrt(modulation_order)) + 1 + 
                              1j*(2*(i // int(np.sqrt(modulation_order))) - int(np.sqrt(modulation_order)) + 1))
                              for i in range(modulation_order)])
            symbols = symbols / np.sqrt(np.mean(np.abs(symbols)**2))
            
            # Create mapping
            mapping = {}
            for i in range(modulation_order):
                bits_tuple = tuple([int(b) for b in format(i, f'0{bits_per_symbol}b')])
                mapping[bits_tuple] = symbols[i]
        
        # Map bits to symbols
        symbols = np.array([mapping[tuple(bits)] for bits in symbol_bits])
        
        return symbols