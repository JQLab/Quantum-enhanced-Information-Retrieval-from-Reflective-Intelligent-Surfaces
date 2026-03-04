import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
# ------------------------
# Simulation parameters
# ------------------------
n0_list = np.arange(1, 11)        # photon numbers per symbol
N_trials = 100000                 # Monte Carlo trials per symbol
M = 16                            # Number of symbols
# ------------------------
# RIS-NPM double-ring parameters
# ------------------------
n1 = 4                             # inner ring points
n2 = 12                            # outer ring points


# ------------------------
# Function to build RIS-NPM symbols
# ------------------------
def build_RIS_NPM_symbols(n1, n2, NI, NO):
    # Inner ring phases
    psi_inner = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    alpha_inner = np.sqrt(NI) * np.exp(1j * psi_inner)
    # Outer ring phases
    psi0 = np.pi/4 - np.pi/n2
    psi_outer = psi0 + 2*np.pi*np.arange(n2)/n2
    alpha_outer = np.sqrt(NO) * np.exp(1j * psi_outer)
    # Combine symbols
    return np.concatenate([alpha_inner, alpha_outer])


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

# ------------------------
# Apply RIS enhancement (linear)
# ------------------------
alpha_RIS = 10*build_RIS_NPM_symbols(n1, n2, 20, 80)
alpha_RIS_enhanced = np.copy(alpha_RIS)
# ------------------------
# 16-ary PSK symbols
# ------------------------
theta_PSK = np.pi/M  # phase step for PSK
alpha_PSK = np.exp(1j * np.arange(M) * theta_PSK)




# ------------------------
# Run Monte Carlo
# ------------------------
P_err_RIS = monte_carlo_SQL(alpha_RIS_enhanced, n0_list, N_trials)
P_err_PSK = monte_carlo_SQL(alpha_PSK, n0_list, N_trials)



Delta_omega = 2*np.pi*7e-1#3              # pulse duration (normalized)
Delta_omega_T = np.pi # constraint Δω T = π
T = Delta_omega_T / Delta_omega
Delta_theta = np.pi / M   # equal phase separation
trials = 5000         # Monte Carlo trials per photon number
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

filename = (f"d1020_TNRM_results_SQL_bit4_0001.mat")
savemat(filename, save_dict)
print("Saved to:", filename)

# ------------------------
# Plot results
# ------------------------
plt.figure(figsize=(8,5))
plt.plot(n0_list, P_err_RIS, 'o-', label='RIS-NPM (16 symbols, double-ring)')
plt.plot(n0_list, P_err_PSK, 's-', label='16-ary PSK')
plt.plot(n0_list, P_err_CFSK, '^-', label='16-ary CFSK')
plt.xlabel('Average photon number per symbol $n_0$')
plt.ylabel('Classical error probability (SQL)')
plt.title('RIS-NPM vs 16-ary CFSK at shot-noise limit')
plt.grid(True)
plt.legend()
plt.yscale('log')  # optional: log-scale for better visualization
plt.savefig("SQL--CFSK_vs_RIS.png", dpi=300, bbox_inches="tight")
plt.show()

# Print numerical values
print("n0:", n0_list)
print("P_err RIS-NPM:", P_err_RIS)
print("P_err CFSK:", P_err_CFSK)
print("P_err PSK:", P_err_PSK)
