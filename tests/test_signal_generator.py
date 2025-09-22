import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.signal_generator import SignalGenerator

class TestSignalGenerator:
    def test_generate_carrier(self):
        sg = SignalGenerator(sample_rate=1e6)
        carrier = sg.generate_carrier(1e6, 1e-3)  # 1 MHz carrier, 1 ms duration
        
        assert len(carrier) == 1000  # 1e6 * 1e-3 = 1000 samples
        assert np.isclose(np.mean(np.abs(carrier)), 1.0)  # Normalized amplitude
    
    def test_generate_bits(self):
        sg = SignalGenerator()
        bits = sg.generate_bits(1000)
        
        assert len(bits) == 1000
        assert set(bits).issubset({0, 1})  # All values are 0 or 1
    
    def test_generate_qam_signal(self):
        sg = SignalGenerator()
        bits = np.array([0, 0, 0, 1, 1, 0, 1, 1])  # 8 bits for 2 QPSK symbols
        symbols = sg.generate_qam_signal(bits, 4)  # QPSK
        
        assert len(symbols) == 4  # 8 bits / 2 bits per symbol = 4 symbols
        assert all(np.isclose(np.abs(symbol), 1.0) for symbol in symbols)  # Unit magnitude