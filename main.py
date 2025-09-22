#!/usr/bin/env python3
"""
Microwave Communication Application
A comprehensive simulation of microwave communication systems
"""

import numpy as np
import argparse
from src.signal_generator import SignalGenerator
from src.modulator import Modulator
from src.channel import MicrowaveChannel
from src.demodulator import Demodulator
from src.analyzer import Analyzer
from src.utils import plot_constellation, plot_spectrum

def main():
    parser = argparse.ArgumentParser(description="Microwave Communication Simulator")
    parser.add_argument("--modulation", type=str, default="BPSK", help="Modulation type (BPSK, QPSK, QAM16, QAM64)")
    parser.add_argument("--snr", type=float, default=20.0, help="SNR in dB")
    parser.add_argument("--bits", type=int, default=1000, help="Number of bits to transmit")
    parser.add_argument("--carrier", type=float, default=2.4e9, help="Carrier frequency in Hz")
    parser.add_argument("--symbol-rate", type=float, default=1e6, help="Symbol rate in symbols/sec")
    parser.add_argument("--sample-rate", type=float, default=10e6, help="Sample rate in samples/sec")
    parser.add_argument("--distance", type=float, default=1.0, help="Distance in km")
    parser.add_argument("--doppler", type=float, default=0.0, help="Maximum Doppler spread in Hz")
    
    args = parser.parse_args()
    
    # Initialize components
    sg = SignalGenerator(sample_rate=args.sample_rate)
    mod = Modulator(sample_rate=args.sample_rate)
    channel = MicrowaveChannel(sample_rate=args.sample_rate)
    demod = Demodulator(sample_rate=args.sample_rate)
    analyzer = Analyzer(sample_rate=args.sample_rate)
    
    # Generate bits
    bits = sg.generate_bits(args.bits)
    print(f"Generated {len(bits)} bits")
    
    # Modulate based on modulation type
    if args.modulation.upper() == "BPSK":
        modulated = mod.bpsk_modulate(bits, args.carrier, args.symbol_rate)
        modulation_order = 2
    elif args.modulation.upper() == "QPSK":
        symbols = sg.generate_qam_signal(bits, 4)
        modulated = mod.qam_modulate(symbols, args.carrier, args.symbol_rate)
        modulation_order = 4
    elif args.modulation.upper() == "QAM16":
        symbols = sg.generate_qam_signal(bits, 16)
        modulated = mod.qam_modulate(symbols, args.carrier, args.symbol_rate)
        modulation_order = 16
    elif args.modulation.upper() == "QAM64":
        symbols = sg.generate_qam_signal(bits, 64)
        modulated = mod.qam_modulate(symbols, args.carrier, args.symbol_rate)
        modulation_order = 64
    else:
        raise ValueError(f"Unsupported modulation type: {args.modulation}")
    
    print(f"Modulated using {args.modulation}")
    
    # Apply channel effects
    frequency_ghz = args.carrier / 1e9
    received = channel.apply_channel(
        modulated, 
        snr_db=args.snr,
        max_doppler=args.doppler,
        distance_km=args.distance,
        frequency_ghz=frequency_ghz
    )
    
    print("Applied channel effects")
    
    # Demodulate
    if args.modulation.upper() == "BPSK":
        received_bits = demod.bpsk_demodulate(received, args.carrier, args.symbol_rate)
        received_symbols = None
    else:
        received_symbols = demod.qam_demodulate(received, args.carrier, args.symbol_rate, modulation_order)
        # Simple threshold detection for demonstration
        # In a real system, you'd use proper QAM demodulation
        real_part = np.real(received_symbols)
        imag_part = np.imag(received_symbols)
        
        if modulation_order == 4:
            # QPSK
            received_bits = np.zeros(2 * len(received_symbols), dtype=int)
            received_bits[0::2] = (real_part < 0).astype(int)
            received_bits[1::2] = (imag_part < 0).astype(int)
        else:
            # For higher order QAM, we'll just do a simple mapping
            # This is not accurate but serves for demonstration
            received_bits = np.zeros(int(np.log2(modulation_order)) * len(received_symbols), dtype=int)
            # Simple threshold detection for each bit
            for i in range(int(np.log2(modulation_order))):
                threshold = 0  # Simple threshold
                if i % 2 == 0:  # Real part bits
                    received_bits[i::int(np.log2(modulation_order))] = (real_part < threshold).astype(int)
                else:  # Imaginary part bits
                    received_bits[i::int(np.log2(modulation_order))] = (imag_part < threshold).astype(int)
    
    print("Demodulated signal")
    
    # Ensure we have the same number of bits
    min_len = min(len(bits), len(received_bits))
    bits = bits[:min_len]
    received_bits = received_bits[:min_len]
    
    # Analyze performance
    performance = analyzer.analyze_performance(bits, received_bits)
    
    print("\n=== Performance Results ===")
    print(f"Modulation: {args.modulation}")
    print(f"SNR: {args.snr} dB")
    print(f"Distance: {args.distance} km")
    print(f"Doppler: {args.doppler} Hz")
    print(f"Bit Error Rate: {performance['ber']:.6f}")
    
    if 'evm' in performance:
        print(f"Error Vector Magnitude: {performance['evm']:.4f}")
    
    # Calculate spectral efficiency
    spectral_eff = analyzer.calculate_spectral_efficiency(
        args.symbol_rate * np.log2(modulation_order),
        args.symbol_rate
    )
    print(f"Spectral Efficiency: {spectral_eff:.2f} bits/sec/Hz")
    
    # Calculate channel capacity
    capacity = analyzer.calculate_capacity(args.snr, args.symbol_rate)
    print(f"Channel Capacity: {capacity/1e6:.2f} Mbps")
    
    # Plot constellation if QAM
    if args.modulation.upper() != "BPSK" and received_symbols is not None:
        plot_constellation(received_symbols[:1000], f"Received {args.modulation} Constellation")
    
    # Plot spectrum
    plot_spectrum(modulated, args.sample_rate, "Transmitted Signal Spectrum")

if __name__ == "__main__":
    main()