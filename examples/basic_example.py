"""
Basic example of using the microwave communication system
"""

import numpy as np
from src.signal_generator import SignalGenerator
from src.modulator import Modulator
from src.channel import MicrowaveChannel
from src.demodulator import Demodulator
from src.analyzer import Analyzer

def basic_example():
    # Parameters
    sample_rate = 10e6  # 10 MHz
    carrier_freq = 2.4e9  # 2.4 GHz
    symbol_rate = 1e6  # 1 Msymbol/s
    num_bits = 10000
    
    # Initialize components
    sg = SignalGenerator(sample_rate)
    mod = Modulator(sample_rate)
    channel = MicrowaveChannel(sample_rate)
    demod = Demodulator(sample_rate)
    analyzer = Analyzer(sample_rate)
    
    # Generate bits
    bits = sg.generate_bits(num_bits)
    
    # BPSK modulation
    modulated = mod.bpsk_modulate(bits, carrier_freq, symbol_rate)
    
    # Apply channel
    received = channel.apply_channel(modulated, snr_db=15, distance_km=5)
    
    # Demodulate
    received_bits = demod.bpsk_demodulate(received, carrier_freq, symbol_rate)
    
    # Analyze
    min_len = min(len(bits), len(received_bits))
    performance = analyzer.analyze_performance(bits[:min_len], received_bits[:min_len])
    
    print("Basic Example Results:")
    print(f"BER: {performance['ber']:.6f}")
    
    return performance

if __name__ == "__main__":
    basic_example()