from HFsuper import HFsuper, assert_real, interpolation, spin_observables
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import argparse
import yaml
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.expand_frame_repr", False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', required=True)
    parser.add_argument('--plot', '-p', action='store_true')
    args = parser.parse_args()

    

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    path_ = config['path']
    nu_ = config['nu']
    N_ = config['N']
    U0_ = config['U0']
    Un_ = config['Un']
    V_ = config['V0']
    Vn_ = config['Vn']
    C0_modify_ = config['C0_modify']
    
    kTmin_ = config['kTmin']
    kTmax_ = config['kTmax']
    kTstep_ = config['kTstep']

    for kT_ in np.arange(kTmin_, kTmax_+1e-10, kTstep_):

        log_file = f"log_{args.config.replace('.yaml', '.log')}"
        log_info = []

        model = HFsuper(path=path_, nu=nu_, N=N_, U0=U0_, Un=Un_, V=V_, Vn=Vn_, kT=kT_)
        now_int = int(np.round(datetime.datetime.now().timestamp() * 1e6))
        result = model.solve(max_iter=10000, 
                            alpha=0.8, 
                            tol_dC=1e-6,
                            verbose=True, 
                            random_seed=now_int, 
                            subtract_reference=C0_modify_,
                            energy_stall_iters=np.inf, 
                            energy_decrease_tol=1e-10,
                            )
        h_k = result['h_k']
        e_hf = result['e_hf']
        e_mean = result['e_mean']
        Ck = result['Ck']
        mu = result['mu']
        converged = result['converged']
        it_ = result['iteration']
        
        
        print(f'convergence: {converged} / iteration: {it_}')

        effective_hopping = model.build_effective_hopping(h_k)
        print(f'U0 = {np.round(model.U0, 6)}')
        print(f'Un = {np.round(model.Un, 6)}')
        print(f'V0 = {np.round(model.V, 6)}')
        print(f'Vn = {np.round(model.Vn, 6)}')
        print(f'kT = {np.round(model.kT, 6)}')
        print(f'C0_modify_ = {C0_modify_}')

        log_info.append(f'U0 = {np.round(model.U0, 6)}')
        log_info.append(f'Un = {np.round(model.Un, 6)}')
        log_info.append(f'V0 = {np.round(model.V, 6)}')
        log_info.append(f'Vn = {np.round(model.Vn, 6)}')
        log_info.append(f'kT = {np.round(model.kT, 6)}')
        log_info.append(f'C0_modify_ = {C0_modify_}')
        
        
        log_info.append(f'e_mean = {e_mean:.8f}')
        chern, energy = model.total_chern_number_energy(effective_hopping, 45)
        log_info.append(f"Total Chern number (filled bands) = {chern:.8f}")
        energy_diff = assert_real(energy[:, :, model.nuSuper] - energy[:, :, model.nuSuper-1])
        assert np.all(energy_diff >= 0)
        log_info.append(f"Energy diff {model.nuSuper} and {model.nuSuper+1} = {np.min(energy_diff):.8f}")
        energy_gap = np.min(energy[:, :, model.nuSuper]) - np.max(energy[:, :, model.nuSuper-1])
        log_info.append(f"Energy gap {model.nuSuper} and {model.nuSuper+1} = {assert_real(energy_gap):.8f}")

        Ck_sum = np.sum(Ck, axis=0)
        n_sum = np.diag(Ck_sum)
        n_up_total, n_down_total = 0, 0
        for j in range(model.numSub):
            sl_up = slice(model.dim * j, model.dim * (j + 1), 2)
            sl_down = slice(model.dim * j + 1, model.dim * (j + 1), 2)
            n_up_total += np.sum(n_sum[sl_up])
            n_down_total += np.sum(n_sum[sl_down])
        spin_components = assert_real(np.sort([n_up_total, n_down_total]))
        log_info.append(f'Polarization = {np.round(spin_components[1]/model.N**2/model.nuSuper, 5)}')

        for idx, grid in model.indexToKGrid.items():
            k = grid[0] * model.Gs[1] / model.N + grid[1] * model.Gs[2] / model.N
            eigvals1 = np.linalg.eigh(h_k[idx, :, :])[0]
            eigvals2 = np.linalg.eigh(model.HKtbEff(k, effective_hopping))[0]
            assert np.allclose(eigvals1, eigvals2)
        log_info.append(f'trace of Ck = {np.round(np.trace(np.sum(Ck, axis=0)), 2)}')

        N_high_symmetry = 100
        band_structure = np.zeros((3*N_high_symmetry+1, model.dimSuper))
        for i in range(N_high_symmetry):
            k = interpolation(model.GammaSuper, model.Ksuper, N_high_symmetry, i)
            eigvals = np.linalg.eigh(model.HKtbEff(k, effective_hopping))[0]
            band_structure[i, :] = eigvals - e_hf

            k = interpolation(model.Ksuper, model.Msuper, N_high_symmetry, i)
            eigvals = np.linalg.eigh(model.HKtbEff(k, effective_hopping))[0]
            band_structure[N_high_symmetry+i, :] = eigvals - e_hf

            k = interpolation(model.Msuper, model.GammaSuper, N_high_symmetry, i)
            eigvals = np.linalg.eigh(model.HKtbEff(k, effective_hopping))[0]
            band_structure[2*N_high_symmetry+i, :] = eigvals - e_hf
        k = model.GammaSuper
        eigvals = np.linalg.eigh(model.HKtbEff(k, effective_hopping))[0]
        band_structure[3*N_high_symmetry, :] = eigvals - e_hf

        
        rho = np.mean(Ck, axis=0)
        df_dict = {}
        for qIdx in range(3):
            for orbIdx in range(model.numOrb):
                sl = slice(model.dim*qIdx, model.dim*(qIdx+1))
                rho_spin = rho[sl, sl][2*orbIdx:2*(orbIdx+1), 2*orbIdx:2*(orbIdx+1)]
                df_dict[qIdx, orbIdx] = spin_observables(rho_spin)
        df = pd.DataFrame(df_dict).T
        log_info.append('=========================================================================================================')
        log_info.append(df.__str__())
        log_info.append('=========================================================================================================')
        log_info.append(f'sum of N = {np.round(df["N"].sum(), 3)}')

        with open(log_file, 'a') as f:
            f.write('\n')
            f.write('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n')
            for line in log_info:
                f.write(line)
                f.write('\n')
            f.write('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n')
            f.write('\n')

        if args.plot:
            for bnd in range(model.dimSuper):
                plt.plot(np.arange(3*N_high_symmetry+1), band_structure[:, bnd], color='k')
            plt.hlines(mu, 0, 3*N_high_symmetry+1, colors='b', linestyles='--', alpha=0.5, linewidth=0.7)
            plt.xticks([0, N_high_symmetry, 2*N_high_symmetry, 3*N_high_symmetry], [r'$\Gamma_s$', r'$K_s$', r'$M_s$', r'$\Gamma_s$'])
            plt.title(f'U0={np.round(model.U0, 6)}, Un={np.round(model.Un, 6)}, V={np.round(model.V, 6)}, Vn={np.round(model.Vn, 6)}, kT={np.round(model.kT, 6)}')
            plt.show()


