import numpy as np
from math import factorial
from functools import partial
import matplotlib.pyplot as plt
from numpy.array_api import argmax
from scipy.io import savemat
rng = np.random.default_rng(12345)
eps = 1e-300

# ---------------------------
# Build RIS symbol set (user provided)
# ---------------------------
def build_RIS_NPM_symbols_256(NI_list):

    assert len(NI_list) == 8, "NI_list must have 8 amplitude rings for 256-RIS."
    alphas = []

    for i in range(1, 9):  # 1-based index for clarity
        n_i = 4 * (2 * i - 1)
        psi0 = np.pi / (2 * n_i)
        psi = psi0 + 2 * np.pi * np.arange(n_i) / n_i
        ring = np.sqrt(NI_list[i - 1]) * np.exp(1j * psi)
        alphas.append(ring)

    return np.concatenate(alphas)

def rate_per_unit_time(alpha,rr, beta):
    a_mag = abs(alpha)
    b_mag = abs(beta)
    mean_whole = (a_mag**2 + b_mag**2 - 2.0 * rr * a_mag * b_mag * np.cos( np.angle(alpha) - np.angle(beta)))
    rate = max(mean_whole / T_full, 0.0)
    return rate





def posterior_update_TRM(area_eff,prior, alpha_list, rr, beta, t_observed, tau):
    M = len(alpha_list)
    rates = np.array([area_eff*rate_per_unit_time(alpha_list[k], rr, beta) for k in range(M)])
    if t_observed is None:
        likes = np.exp(-rates * tau)
    else:
        t = t_observed
        likes = rates * np.exp(-rates * t)
    post_unnorm = prior * likes
    s = post_unnorm.sum()
    if s <= 0:
        post = np.ones_like(prior) / len(prior)
    else:
        post = post_unnorm / s
    return post



def adaptive_hybrid_multiarea_receiver(
    alpha_list, true_index,
    rr=1.0,
    beta_candidates=None,
    prior_init=None,
    area_efficiency=None,    # efficiency per area (multi-color)
          # maximum time bin before adaptation
    feedback_delay=1,#30*1e-3,     # feedback latency between adaptations   # adapt after this many detections
    rng=np.random.default_rng()
):
    """
    Multi-area adaptive photon receiver.
    Adapts beta after receiving N photons or reaching tau_bin_max.
    Uses posterior update based on photon arrival times,
    selecting the one that yields the maximum posterior.
    """
    tau_bin_max = T_full/10
    M = len(alpha_list)
    if prior_init is None:
        prior = np.ones(M) / M
    else:
        prior = prior_init.copy()

    if beta_candidates is None:
        beta_candidates = list(alpha_list)


    elapsed = 0.0

    log_lambda = []
    log_deltaT = []
    log_elapsed = []


    xtx = 0
    while elapsed < T_full - 1e-12:
        xtx += 1
        if xtx > 200:
            break

        remaining = T_full - elapsed
        best_idx_trm = int(np.argmax(prior))
        chosen_beta = beta_candidates[best_idx_trm]

        # Calculate multi-area photon rates
        lam_true = rate_per_unit_time(alpha_list[true_index], rr, chosen_beta)
        lam_true_eff_list = [lam_true * area_efficiency[area_i] for area_i in range(numS)]
        # Draw one photon waiting time from each area
        # Draw waiting times for each area (exponential inter-arrival distribution)

        # Draw waiting times for each area
        wait_t_list = [rng.exponential(1.0 / lam) if lam > 0 else np.inf for lam in lam_true_eff_list]

        # Sort times but keep track of which area each photon came from
        wait_t_with_idx = sorted(enumerate(wait_t_list), key=lambda x: x[1])
        wait_t_sort = np.array([t for _, t in wait_t_with_idx])
        wait_t_area_idx = np.array([i for i, _ in wait_t_with_idx])
        # print(wait_t_list,'wait--t',wait_t_sort,wait_t_area_idx)


        tau_bin_max_here = min(remaining, tau_bin_max)


        # # After system has stabilized — consider all photon arrivals within the bin
        if xtx < int(99*T_full/100):#1027---int(99*T_full/100):


            # After system has stabilized — consider all photon arrivals within the bin
            photon_times = wait_t_sort[wait_t_sort <= tau_bin_max_here]
            photon_areas = wait_t_area_idx[wait_t_sort <= tau_bin_max_here]
            #print(xtx,'pho-timws--', photon_times)
        else:
            #early phase — only first photon
            if wait_t_sort[0] <= tau_bin_max_here:
                photon_times = np.array([wait_t_sort[0]])
                photon_areas = np.array([wait_t_area_idx[0]])
            else:
                photon_times = np.array([])
                photon_areas = np.array([])
                #print(xtx, '-first-', photon_times)


        # Posterior update
        #print('photon-collect',photon_times)
        if len(photon_times) == 0:
            # No photon within bin
            prior = posterior_update_TRM(max(area_efficiency),prior, alpha_list, rr, chosen_beta, None, tau_bin_max_here)
            elapsed += tau_bin_max_here + feedback_delay

            deltaT = tau_bin_max_here + feedback_delay

            log_lambda.append(lam_true)
            log_deltaT.append(deltaT)
            log_elapsed.append(elapsed)
        else:
            # Photon detected: update with each photon arrival
            prior_list = []
            maxprior_list = []
            for i in range(len(photon_times)):
                t_obs = photon_times[i]
                area_idx = photon_areas[i]
                p_tmp = posterior_update_TRM(area_efficiency[area_idx],prior.copy(), alpha_list, rr, chosen_beta, t_obs, tau_bin_max_here)
                prior_list.append(p_tmp)
                maxprior_list.append(np.max(p_tmp))

            # Choose posterior with maximum posterior probability
            idx_areap = int(np.argmax(maxprior_list))
            prior = prior_list[idx_areap]
            #print('prior--', prior)
            # Advance time by latest photon + feedback delay
            t_max = np.max(photon_times)
            elapsed += t_max + feedback_delay

            # Update chosen beta (MAP)
            best_idx_map = int(np.argmax(prior))
            chosen_beta = beta_candidates[best_idx_map]

            deltaT = t_max + feedback_delay

            log_lambda.append(lam_true)
            log_deltaT.append(deltaT)

            log_elapsed.append(elapsed)

        if elapsed >= T_full:
            break

    m_hat = int(np.argmax(prior))
    result = {
        'm_hat': m_hat,
        'log_lambda':log_lambda,
        'log_deltaT': log_deltaT,
        'log_elapsed': log_elapsed,

    }
    return result


# ---------------------------
# Example run with provided RIS symbols
# ---------------------------
if __name__ == "__main__":

    numBit = 8
    # parameters

    rr = 0.9995#0.9993##1027---0.9995#0.997#1#0.997#1##1#0.98



    numS  = 2
    n0 = 1/numS
    area_efficiency =  [0.99,0.69]
    # [0.99,0.89]#[0.60]#[0.57, 0.55, 0.60]#[0.60 for _ in range(numS)]#[0.59, 0.58, 0.60, 0.57]#


    num_trials = 10000

    Tf_list = [30]#[10**i for i in range(-1,6)]
    n = 4000#4000

    # Containers for results
    Pe_avg_list = []
    T_avg_list = []
    L_list = []
    # ---- Loop over different n0 values ----
    for tfu in Tf_list:

        T_full = tfu  # 100
        NI_list = np.linspace(10*n, 160*n, 8)  # amplitude levels
        alphas00 = build_RIS_NPM_symbols_256(NI_list)
        print(f"\n===== Running simulation for RIS-num = {n} =====")

        alphas = alphas00 * np.sqrt(n0)
        beta_candidates = list(alphas)

        Pe_list = []
        T_list = []
        # ---- Monte Carlo loop ----
        for trial in range(num_trials):
            true_index = rng.integers(0, len(alphas))
            if trial%500==0:
                print(np.mean(Pe_list),f"n={tfu} | Trial={trial} | True symbol index={true_index}")


            result = adaptive_hybrid_multiarea_receiver(
                alphas, true_index,rr=rr,beta_candidates=beta_candidates,area_efficiency= area_efficiency)

            m_hat = result['m_hat']
            L_list.append(result)

            Pe = float(m_hat != true_index)
            Pe_list.append(Pe)

        # ---- Compute averages ----
        Pe_avg = np.mean(Pe_list)
        T_avg = np.mean(T_list)
        Pe_avg_list.append(Pe_avg)
        T_avg_list.append(T_avg)

        save_dict = {
            'L_list':L_list
        }

        filename = f"Llist_1122_results_bit{numBit}_rr{rr}_numS{numS}.mat"
        savemat(filename, save_dict)
        print("Saved to:", filename)

        print(f"Tfu={tfu}: Average Pe={Pe_avg:.4e}, Avg Time={T_avg:.3f}")

        # ---- Sort and save ----
        Pe_list = np.array(Pe_list)


    # ---- Save overall results ----
    overall_save = {
        'Tfull_list': Tf_list,
        'Pe_avg_list': np.array(Pe_avg_list),
        'T_avg_list': np.array(T_avg_list)
    }
    savemat(f"L1122_Bit{numBit}_rr{rr}_numS{numS}_num_trials{num_trials}.mat", overall_save)
    print("\nAll simulations completed and summary saved.")

