import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
# ------------------------
# Simulation parameters
# ------------------------
n0_list = np.arange(1, 11)        # photon numbers per symbol
N_trials = 2000               # Monte Carlo trials per symbol
M = 2**6                           # Number of symbols
#n0_list = [0.001,0.005,0.01,0.05,0.1,0.2,0.5,1]


# ------------------------
# Function to build RIS-NPM symbols
# ------------------------
import numpy as np

# def build_RIS_NPM_symbols_64(NI_list):
#     """
#     Build 64-ary RIS modulation constellation:
#     - 4 amplitude rings
#     - i-th ring has n_i = 4*(2*i - 1) evenly spaced phase points.
#
#     NI_list : list/array of amplitude power levels for each ring (length = 4)
#               Typically increasing (e.g. np.linspace(1, 4, 4))
#
#     Returns:
#         complex numpy array of size 64
#     """
#     assert len(NI_list) == 4, "NI_list must have 4 amplitude rings for 64-RIS."
#     alphas = []
#
#     for i in range(1, 5):  # 1..4 rings
#         n_i = 4 * (2 * i - 1)
#         psi0 = np.pi / (2 * n_i)     # phase offset for symmetry
#         psi = psi0 + 2 * np.pi * np.arange(n_i) / n_i
#         ring = np.sqrt(NI_list[i - 1]) * np.exp(1j * psi)
#         alphas.append(ring)
#
#     return np.concatenate(alphas)

import numpy as np

def build_RIS_64_adaptive_rings_fixedNI(n0_single,
                                        R_min=4,
                                        R_max=8,
                                        total_symbols=64):
    """
    Build a 64-symbol RIS constellation:
      - Number of rings increases with n0 (4→8)
      - NI_base = np.linspace(10, 40, R), fixed and not scaled by n0
      - Total symbol count fixed at 64

    Args:
        n0_single : float
            average photon number per RIS element (1–10)
        R_min, R_max : int
            minimum and maximum number of rings (default 4–8)
        total_symbols : int
            total number of constellation points (default 64)

    Returns:
        alphas : complex ndarray of 64 elements
        n_i : list of ints, symbol count per ring
        NI_list : ndarray of power per ring
        R : int, number of rings
    """

    # --- map n0 → ring count (linear) ---
    n0_clamped = float(np.clip(n0_single, 1.0, 10.0))
    R = int(round(R_min + (R_max - R_min) * (n0_clamped - 1.0) / (10.0 - 1.0)))
    R = max(R_min, min(R_max, R))

    # --- compute symbol count per ring proportional to (2i−1) ---
    weights = np.array([2 * i - 1 for i in range(1, R + 1)], dtype=float)
    raw = weights / weights.sum() * total_symbols
    n_i = np.floor(raw).astype(int)

    # adjust rounding to keep total = 64
    deficit = total_symbols - n_i.sum()
    if deficit > 0:
        frac = raw - n_i
        for idx in np.argsort(-frac)[:deficit]:
            n_i[idx] += 1
    elif deficit < 0:
        frac = raw - n_i
        for idx in np.argsort(frac)[:(-deficit)]:
            if n_i[idx] > 1:
                n_i[idx] -= 1

    # --- define ring power levels (fixed NI_base) ---
    NI_list = np.linspace(10, 40, R)

    # --- build rings ---
    alphas_parts = []
    for i in range(1, R + 1):
        ni = n_i[i - 1]
        psi0 = np.pi / (2 * ni)
        psi = psi0 + 2 * np.pi * np.arange(ni) / ni
        ring = np.sqrt(NI_list[i - 1]) * np.exp(1j * psi)
        alphas_parts.append(ring)

    alphas = np.concatenate(alphas_parts)
    if len(alphas) > total_symbols:
        alphas = alphas[:total_symbols]

    return alphas#, n_i.tolist(), NI_list, R

# ------------------------
# Monte Carlo classical discrimination
# ------------------------
def monte_carlo_SQL(alpha_symbols, n0_list, N_trials):
    M = len(alpha_symbols)
    P_err_list = []
    for n0 in n0_list:
        alpha_scaled = np.sqrt(n0) * alpha_symbols
        errors = 0
        for j in range(M):
            # Heterodyne-like measurement
            z = alpha_scaled[j] + (np.random.randn(N_trials) + 1j*np.random.randn(N_trials))/np.sqrt(2)
            distances = np.abs(z[:, np.newaxis] - alpha_scaled[np.newaxis, :])**2
            decisions = np.argmin(distances, axis=1)
            errors += np.sum(decisions != j)
        P_err = errors / (M * N_trials)
        P_err_list.append(P_err)
    return np.array(P_err_list)

def monte_carlo_SQL_RIS( n0_list, N_trials):
    M = 64
    P_err_list = []
    for n0 in n0_list:
        alphas_64 = build_RIS_64_adaptive_rings_fixedNI(n0)
        print(len(alphas_64))  # 64

        alpha_symbols = np.copy(alphas_64)
        alpha_scaled = np.sqrt(n0) * alpha_symbols
        errors = 0
        for j in range(M):
            # Heterodyne-like measurement
            z = alpha_scaled[j] + (np.random.randn(N_trials) + 1j*np.random.randn(N_trials))/np.sqrt(2)
            distances = np.abs(z[:, np.newaxis] - alpha_scaled[np.newaxis, :])**2
            decisions = np.argmin(distances, axis=1)
            errors += np.sum(decisions != j)
        P_err = errors / (M * N_trials)
        P_err_list.append(P_err)
    return np.array(P_err_list)

# ------------------------
# Apply RIS enhancement (linear)
# ------------------------
NI_list = np.linspace(10, 40, 4)  # amplitude levels


# ------------------------
# 64-ary PSK symbols
# ------------------------
theta_PSK = np.pi/M  # phase step for PSK
alpha_PSK = np.exp(1j * np.arange(M) * theta_PSK)




# ------------------------
# Run Monte Carlo
# ------------------------
P_err_RIS = monte_carlo_SQL_RIS( n0_list, N_trials)
P_err_PSK = monte_carlo_SQL(alpha_PSK, n0_list, N_trials)



Delta_omega = 2*np.pi*7e-1#3              # pulse duration (normalized)
Delta_omega_T = np.pi # constraint Δω T = π
T = Delta_omega_T / Delta_omega
Delta_theta = np.pi / M   # equal phase separation
trials = N_trials         # Monte Carlo trials per photon number
n0_vals = n0_list  # avg photon numbers

# Precompute correlation coefficients gamma_ml
def gamma(m, l):
    k = m - l
    if k == 0:
        return 1.0
    return (np.cos(k * (Delta_omega_T/2 + Delta_theta)) *
            np.sinc(k * Delta_omega_T / (2*np.pi)))  # sinc(x) = sin(pi x)/(pi x)

gamma_mat = np.zeros((M, M), dtype=float)
for m in range(M):
    for l in range(M):
        gamma_mat[m, l] = gamma(m, l)

# Gram-Schmidt orthogonalization of signal set {s_m}
# Start with correlation matrix G = gamma_mat, Cholesky factorization gives coefficients
G = gamma_mat
Q, R = np.linalg.qr(G)  # orthonormal basis approx
smi = Q.T @ G           # coefficients smi

P_err_CFSK = []

# Monte Carlo simulation
for n0 in n0_vals:
    errors = 0
    for _ in range(trials):
        j = np.random.randint(M)  # transmitted symbol
        # Build correlations C(rj, sm) for all m
        C_vals = np.zeros(M, dtype=float)
        for m in range(M):
            signal_part = np.sqrt(2*n0) * gamma_mat[j, m]
            noise_part = np.sum(smi[m, :] * np.random.randn(M))
            C_vals[m] = signal_part + noise_part
        m_hat = np.argmax(C_vals)
        if m_hat != j:
            errors += 1
    P_err_CFSK.append(errors / trials)

save_dict = {
    'n0': n0_list,
    'P_err_RIS': P_err_RIS,
    'P_err_PSK': P_err_PSK,
    'P_err_CFSK': P_err_CFSK
}

filename = f"d1020_TNRM_results_SQL_ring.mat"
savemat(filename, save_dict)
print("Saved to:", filename)


# plt.figure(figsize=(5,5))
# plt.plot(alphas_64.real, alphas_64.imag, 'o')
# plt.axis('equal')
# plt.title('64-ary RIS Constellation (n_i = 4*(2i-1))')
# plt.xlabel('Re')
# plt.ylabel('Im')
# plt.show()

# ------------------------
# Plot results
# ------------------------
plt.figure(figsize=(8,5))
plt.plot(n0_list, P_err_RIS, 'o-', label='RIS-NPM (64 symbols, double-ring)')
plt.plot(n0_list, P_err_PSK, 's-', label='64-ary PSK')
plt.plot(n0_list, P_err_CFSK, '^-', label='64-ary CFSK')
plt.xlabel('Average photon number per symbol $n_0$')
plt.ylabel('Classical error probability (SQL)')
plt.title('RIS-NPM vs 64-ary CFSK at shot-noise limit')
plt.grid(True)
plt.legend()
plt.yscale('log')  # optional: log-scale for better visualization
plt.savefig("SQL--CFSK_vs_RISring.png", dpi=300, bbox_inches="tight")
plt.show()

# Print numerical values
print("n0:", n0_list)
print("P_err RIS-NPM:", P_err_RIS)
print("P_err CFSK:", P_err_CFSK)
print("P_err PSK:", P_err_PSK)
