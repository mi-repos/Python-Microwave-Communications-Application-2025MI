import numpy as np
from typing import Union, List, Tuple
from scipy import signal
from .utils import db_to_linear

class MicrowaveChannel:
    def __init__(self, sample_rate: float = 1e6):
        self.sample_rate = sample_rate
        
    def add_awgn(self, signal: np.ndarray, snr_db: float) -> np.ndarray:
        """Add Additive White Gaussian Noise."""
        signal_power = np.mean(np.abs(signal) ** 2)
        noise_power = signal_power / db_to_linear(snr_db)
        noise = np.sqrt(noise_power / 2) * (np.random.randn(len(signal)) + 
                                          1j * np.random.randn(len(signal)))
        return signal + noise
    
    def add_fading(self, signal: np.ndarray, max_doppler: float = 10.0, 
                  num_taps: int = 4) -> np.ndarray:
        """Add multipath fading."""
        # Generate Rayleigh fading taps
        taps = (np.random.randn(num_taps) + 1j * np.random.randn(num_taps)) / np.sqrt(2)
        
        # Apply Doppler spread
        t = np.arange(0, len(signal)) / self.sample_rate
        doppler_phase = 2 * np.pi * max_doppler * t
        
        # Create time-varying channel
        channel = np.zeros_like(signal, dtype=complex)
        for i, tap in enumerate(taps):
            delay = i / self.sample_rate
            channel += tap * np.exp(1j * doppler_phase) * np.sinc(max_doppler * (t - delay))
        
        # Apply channel
        return signal * channel
    
    def add_path_loss(self, signal: np.ndarray, distance_km: float, 
                     frequency_ghz: float) -> np.ndarray:
        """Add free space path loss."""
        # Free space path loss formula: FSPL = (4πdƒ/c)²
        # d = distance, ƒ = frequency, c = speed of light
        c = 3e8  # speed of light in m/s
        wavelength = c / (frequency_ghz * 1e9)
        path_loss = (4 * np.pi * distance_km * 1000 / wavelength) ** 2
        
        # Convert to linear scale and apply
        path_loss_linear = path_loss
        return signal / np.sqrt(path_loss_linear)
    
    def add_phase_noise(self, signal: np.ndarray, phase_noise_db: float = -80) -> np.ndarray:
        """Add phase noise."""
        phase_noise_power = db_to_linear(phase_noise_db)
        phase_noise = np.sqrt(phase_noise_power) * np.random.randn(len(signal))
        phase_shift = np.exp(1j * np.cumsum(phase_noise) / self.sample_rate)
        return signal * phase_shift
    
    def apply_channel(self, signal: np.ndarray, snr_db: float = 20, 
                     max_doppler: float = 0, distance_km: float = 1, 
                     frequency_ghz: float = 2.4, phase_noise_db: float = -100) -> np.ndarray:
        """Apply all channel effects."""
        # Apply path loss
        signal = self.add_path_loss(signal, distance_km, frequency_ghz)
        
        # Apply fading if Doppler spread is specified
        if max_doppler > 0:
            signal = self.add_fading(signal, max_doppler)
        
        # Add phase noise
        signal = self.add_phase_noise(signal, phase_noise_db)
        
        # Add AWGN
        signal = self.add_awgn(signal, snr_db)
        
        return signal