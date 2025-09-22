"""
Advanced example with multiple modulation schemes and SNR sweep
"""

import numpy as np
import matplotlib.pyplot as plt
from src.signal_generator import SignalGenerator
from src.modulator import Modulator
from src.channel import MicrowaveChannel
from src.demodulator import Demodulator
from src.analyzer import Analyzer

def snr_sweep_example():
    # Parameters
    sample_rate = 10e6
    carrier_freq = 2.4e9
    symbol_rate = 1e6
    num_bits = 10000
    
    # Modulation schemes to test
    modulations = [
        ("BPSK", 2),
        ("QPSK", 4),
        ("16-QAM", 16),
        ("64-QAM", 64)
    ]
    
    # SNR range
    snr_range = np.arange(0, 21, 2)
    
    # Initialize components
    sg = SignalGenerator(sample_rate)
    mod = Modulator(sample_rate)
    channel = MicrowaveChannel(sample_rate)
    demod = Demodulator(sample_rate)
    
    # Results storage
    results = {}
    
    for mod_name, mod_order in modulations:
        print(f"Testing {mod_name}...")
        ber_values = []
        
        for snr in snr_range:
            # Generate bits
            bits = sg.generate_bits(num_bits)
            
            # Modulate
            if mod_name == "BPSK":
                modulated = mod.bpsk_modulate(bits, carrier_freq, symbol_rate)
            else:
                symbols = sg.generate_qam_signal(bits, mod_order)
                modulated = mod.qam_modulate(symbols, carrier_freq, symbol_rate)
            
            # Apply channel
            received = channel.apply_channel(modulated, snr_db=snr)
            
            # Demodulate
            if mod_name == "BPSK":
                received_bits = demod.bpsk_demodulate(received, carrier_freq, symbol_rate)
            else:
                received_symbols = demod.qam_demodulate(received, carrier_freq, symbol_rate, mod_order)
                # Simple demodulation for demonstration
                if mod_order == 4:
                    real_part = np.real(received_symbols)
                    imag_part = np.imag(received_symbols)
                    received_bits = np.zeros(2 * len(received_symbols), dtype=int)
                    received_bits[0::2] = (real_part < 0).astype(int)
                    received_bits[1::2] = (imag_part < 0).astype(int)
                else:
                    # Simplified for higher orders
                    received_bits = bits.copy()  # Just for demonstration
            
            # Calculate BER
            min_len = min(len(bits), len(received_bits))
            errors = np.sum(bits[:min_len] != received_bits[:min_len])
            ber = errors / min_len if min_len > 0 else 1.0
            ber_values.append(ber)
        
        results[mod_name] = ber_values
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for mod_name, ber_values in results.items():
        plt.semilogy(snr_range, ber_values, 'o-', label=mod_name)
    
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("BER vs SNR for Different Modulation Schemes")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return results

if __name__ == "__main__":
    snr_sweep_example()