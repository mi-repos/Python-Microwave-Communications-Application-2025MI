# Microwave Communication Application

A comprehensive Python application for simulating microwave communication systems with various modulation schemes, channel effects, and performance analysis.

## Features

- **Modulation Schemes**: BPSK, QPSK, 16-QAM, 64-QAM, OFDM
- **Channel Effects**: AWGN, multipath fading, path loss, phase noise
- **Performance Analysis**: BER, EVM, spectral efficiency, channel capacity
- **Visualization**: Constellation diagrams, spectrum plots, BER vs SNR curves

## Installation

1. Clone the repository:
```bash
git clone <pasteRepository-Url-HERE>
cd microwave_communication_app

#2 Create and activate virtual environment:
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows

#3 Install dependencies:
pip install -r requirements.txt


Usage

# Basic BPSK transmission
$ python main.py --modulation BPSK --snr 10 --bits 5000

# QAM16 with higher SNR
$ python main.py --modulation QAM16 --snr 20 --bits 10000 --distance 2.5

# With Doppler effect
$ python main.py --modulation QPSK --snr 15 --doppler 50 --bits 5000

# Long distance transmission
$ python main.py --modulation BPSK --snr 5 --distance 10 --bits 10000


------------------------------
# Programmed By Mohammed Issa
------------------------------

