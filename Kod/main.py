import numpy as np
import matplotlib.pyplot as plt
import time

# ==========================================
# HAMMING (7,4)
# ==========================================
G = np.array([
    [1, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1]
])

H = np.array([
    [1, 1, 0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0, 1, 0],
    [0, 1, 1, 1, 0, 0, 1]
])

def encode_hamming(bits):
    bits = bits.reshape(-1, 4)
    return (bits @ G) % 2

def decode_hamming(codeword):
    syndrome = (H @ codeword.T) % 2
    syndrome = syndrome.T

    corrected = []
    for i, s in enumerate(syndrome):
        error_pos = None
        for j in range(7):
            if np.array_equal(H[:, j], s):
                error_pos = j
                break

        cw = codeword[i].copy()
        if error_pos is not None:
            cw[error_pos] ^= 1
        corrected.append(cw[:4])

    return np.array(corrected).reshape(-1)

# ==========================================
# REPETITION CODE (3,1)
# ==========================================
def encode_repetition(bits):
    return np.repeat(bits, 3)

def decode_repetition(bits):
    bits = bits.reshape(-1, 3)
    return (np.sum(bits, axis=1) >= 2).astype(int)

# ==========================================
# BPSK
# ==========================================
def bpsk_mod(bits):
    return 2 * bits - 1

def bpsk_demod(symbols):
    return (symbols > 0).astype(int)

# ==========================================
# AWGN
# ==========================================
def awgn(signal, snr_db):
    snr = 10 ** (snr_db / 10)
    power = np.mean(signal ** 2)
    noise_power = power / snr
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))
    return signal + noise

# ==========================================
# BER TEST
# ==========================================
def simulate(snr_db_range, n_bits=4000):
    ber_no = []
    ber_hamming = []
    ber_rep = []

    throughput_no = []
    throughput_hamming = []
    throughput_rep = []

    for snr_db in snr_db_range:
        bits = np.random.randint(0, 2, n_bits)

        # ==========================
        # NO ECC
        # ==========================
        tx = bpsk_mod(bits)
        rx = awgn(tx, snr_db)
        rx_bits = bpsk_demod(rx)

        ber = np.mean(bits != rx_bits)
        ber_no.append(ber)
        throughput_no.append(1 * (1 - ber))

        # ==========================
        # HAMMING
        # ==========================
        bits_h = bits[:(len(bits)//4)*4]

        coded = encode_hamming(bits_h)
        tx_h = bpsk_mod(coded.flatten())
        rx_h = awgn(tx_h, snr_db)
        rx_bits_h = bpsk_demod(rx_h)

        decoded = decode_hamming(rx_bits_h.reshape(-1, 7))
        ber = np.mean(bits_h != decoded[:len(bits_h)])
        ber_hamming.append(ber)
        throughput_hamming.append((4/7)*(1-ber))

        # ==========================
        # REPETITION
        # ==========================
        rep = encode_repetition(bits)
        tx_r = bpsk_mod(rep)
        rx_r = awgn(tx_r, snr_db)
        rx_bits_r = bpsk_demod(rx_r)

        decoded_r = decode_repetition(rx_bits_r)
        ber = np.mean(bits != decoded_r)
        ber_rep.append(ber)
        throughput_rep.append((1/3)*(1-ber))

    return (ber_no, ber_hamming, ber_rep,
            throughput_no, throughput_hamming, throughput_rep)

# ==========================================
# BLOCK SIZE TEST
# ==========================================
def block_size_test():
    sizes = [4000, 8000, 16000, 32000]

    times_no = []
    times_hamming = []
    times_rep = []

    for size in sizes:
        bits = np.random.randint(0, 2, size)

        # ==========================
        # NO ECC
        # ==========================
        start = time.time()
        tx = bpsk_mod(bits)
        rx = awgn(tx, 5)
        rx_bits = bpsk_demod(rx)
        elapsed = time.time() - start
        times_no.append(elapsed)

        # ==========================
        # HAMMING
        # ==========================
        bits_h = bits[:(len(bits)//4)*4]

        start = time.time()
        coded = encode_hamming(bits_h)
        tx = bpsk_mod(coded.flatten())
        rx = awgn(tx, 5)
        rx_bits = bpsk_demod(rx)
        decode_hamming(rx_bits.reshape(-1, 7))
        elapsed = time.time() - start
        times_hamming.append(elapsed)

        # ==========================
        # REPETITION
        # ==========================
        start = time.time()
        rep = encode_repetition(bits)
        tx = bpsk_mod(rep)
        rx = awgn(tx, 5)
        rx_bits = bpsk_demod(rx)
        decode_repetition(rx_bits)
        elapsed = time.time() - start
        times_rep.append(elapsed)

    return sizes, times_no, times_hamming, times_rep
# ==========================================
# RUN
# ==========================================
snr_db_range = np.arange(0, 11, 1)

(ber_no, ber_hamming, ber_rep,
 throughput_no, throughput_hamming, throughput_rep) = simulate(snr_db_range)

sizes, times_no, times_hamming, times_rep = block_size_test()

# ==========================================
# GRAF 1 - BER
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(snr_db_range, ber_no, 'o-', label="Bez ECC")
plt.plot(snr_db_range, ber_hamming, 's-', label="Hamming")
plt.plot(snr_db_range, ber_rep, '^-', label="Repetition")
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.title("BER analiza")
plt.grid()
plt.legend()
plt.show()

# ==========================================
# GRAF 2 - Throughput
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(snr_db_range, throughput_no, 'o-', label="Bez ECC")
plt.plot(snr_db_range, throughput_hamming, 's-', label="Hamming")
plt.plot(snr_db_range, throughput_rep, '^-', label="Repetition")
plt.xlabel("SNR (dB)")
plt.ylabel("Throughput")
plt.title("Throughput analiza")
plt.grid()
plt.legend()
plt.show()

# ==========================================
# GRAF 3 - Block Size
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(sizes, times_no, 'o-', label="Bez ECC")
plt.plot(sizes, times_hamming, 's-', label="Hamming")
plt.plot(sizes, times_rep, '^-', label="Repetition")

plt.xlabel("Broj bitova")
plt.ylabel("Vreme izvršavanja (s)")
plt.title("Uticaj veličine bloka na vreme izvršavanja (SNR = 5 dB)")
plt.grid()
plt.legend()
plt.show()

# ==========================================
# GRAF 4 - Coding Rate
# ==========================================
methods = ["Bez ECC", "Hamming", "Repetition"]
rates = [1, 4/7, 1/3]

plt.figure(figsize=(8, 5))
plt.bar(methods, rates)
plt.ylabel("Coding Rate")
plt.title("Poređenje Coding Rate")
plt.show()