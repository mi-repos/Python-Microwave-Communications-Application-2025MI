import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List, Tuple
from .utils import calculate_ber, calculate_evm, plot_constellation, plot_spectrum

class Analyzer:
    def __init__(self, sample_rate: float = 1e6):
        self.sample_rate = sample_rate
        
    def calculate_spectral_efficiency(self, data_rate: float, bandwidth: float) -> float:
        """Calculate spectral efficiency in bits/sec/Hz."""
        return data_rate / bandwidth
    
    def calculate_capacity(self, snr_db: float, bandwidth: float) -> float:
        """Calculate channel capacity using Shannon-Hartley theorem."""
        snr_linear = 10 ** (snr_db / 10)
        return bandwidth * np.log2(1 + snr_linear)
    
    def analyze_performance(self, transmitted_bits: np.ndarray, 
                           received_bits: np.ndarray, 
                           transmitted_symbols: np.ndarray = None,
                           received_symbols: np.ndarray = None) -> dict:
        """Analyze communication system performance."""
        results = {}
        
        # Calculate BER
        results['ber'] = calculate_ber(received_bits, transmitted_bits)
        
        # Calculate EVM if symbols are provided
        if transmitted_symbols is not None and received_symbols is not None:
            results['evm'] = calculate_evm(received_symbols, transmitted_symbols)
        
        return results
    
    def plot_ber_vs_snr(self, snr_range: np.ndarray, ber_values: np.ndarray) -> None:
        """Plot BER vs SNR curve."""
        plt.figure(figsize=(10, 6))
        plt.semilogy(snr_range, ber_values)
        plt.xlabel("SNR (dB)")
        plt.ylabel("Bit Error Rate (BER)")
        plt.title("BER vs SNR")
        plt.grid(True)
        plt.show()
    
    def plot_spectral_efficiency(self, modulation_orders: List[int], 
                                efficiencies: List[float]) -> None:
        """Plot spectral efficiency for different modulation orders."""
        plt.figure(figsize=(10, 6))
        plt.plot(modulation_orders, efficiencies, 'o-')
        plt.xlabel("Modulation Order")
        plt.ylabel("Spectral Efficiency (bits/sec/Hz)")
        plt.title("Spectral Efficiency vs Modulation Order")
        plt.grid(True)
        plt.show()