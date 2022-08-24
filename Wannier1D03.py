import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
import matplotlib.colors as colors
import scipy.special as sps
import os as os
import Wannier1DLib as wlib

print("")
print("  4. Time Propagation - Wannier Basis")
print("")

N_t = 8192
N_bands = 2

PROP_TYPE = 0
SEP_L     = 0

omega0 = 0.014225
E0     = 0.002
T0     = 2.0 * np.pi / omega0
tau0   = 8.0 * T0
phi    = 0.0

T2     = 2.5

paramsT = np.array([N_t, N_bands, PROP_TYPE, omega0, E0, T0, tau0, phi, T2])

# Load parameters

params = np.load("./Data/Wannier1D00/params.npy")

N_x    = int(params[0])
N_L    = int(params[1])
N_k    = int(params[2])
N_b    = int(params[3])

a0     = params[4]
v0     = params[5]

k      = np.load("./Data/Wannier1D00/k.npy")
dk     = k[1] - k[0]

x      = np.load("./Data/Wannier1D00/x.npy")
dx     = x[1] - x[0]

X      = np.load("./Data/Wannier1D00/xL.npy")

D1     = np.load("./Data/Wannier1D00/D1.npy")
D2     = np.load("./Data/Wannier1D00/D2.npy")

v      = np.load("./Data/Wannier1D00/v.npy")

states = np.load("./Data/Wannier1D00/states.npy")
energy = np.load("./Data/Wannier1D00/energy.npy")

DK1, DK2 = wlib.getDerivativeMatrices (N_k, dk)

H_x = np.load("./Data/Wannier1D01/H_x.npy")
P_x = np.load("./Data/Wannier1D01/P_x.npy")
D_x = np.load("./Data/Wannier1D01/D_x.npy")
P_L = np.load("./Data/Wannier1D01/P_L.npy")

for idx in range(2 * N_L):
    
    H_x[ N_L // 2 + idx : N_L * 2 * 3 // 4 + idx, idx] = 0.0

    H_x[ idx,  N_L // 2 + idx : N_L * 2 * 3 // 4 + idx] = 0.0

    P_x[ N_L // 2 + idx : N_L * 2 * 3 // 4 + idx, idx] = 0.0

    P_x[ idx,  N_L // 2 + idx : N_L * 2 * 3 // 4 + idx] = 0.0

    # P_x[ idx, N_L // 2 + idx : N_L * 3 // 4] = 0.0

# plt.imshow(np.log10(np.absolute(H_x)))
# plt.colorbar()
# plt.show()

# exit()


w_B = np.load("./data/Wannier1D02/w.npy")
J_B = np.load("./data/Wannier1D02/j_spec.npy")

# Preliminary Calculations

gap    = energy[2, :] - energy[1, :]
minGap = np.min(gap)
maxGap = np.max(gap)

kdxMax = np.argmin(np.absolute(E0 / omega0 - k))

peakKDX = np.argmin(np.absolute(k - E0 / omega0 / 2.0))

cutOff = gap[kdxMax]

T2    *= 41.341374575751 

if T2 > 0.0:

    decoherenceConstant = 1.0 / T2

else:

    decoherenceConstant = 0.0

LSD = np.linspace(0, 30, 16, dtype=int)

print(LSD)

for LSDX in range(np.size(LSD)):

    t      = np.linspace(- 2.0 * tau0, 2.0 * tau0, N_t)
    dt     = t[1] - t[0]

    t      = np.append(t, np.array([t[-1] + dt]))

    w      = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(N_t, dt))
    dw     = w[1] - w[0]

    j      = np.zeros((N_L, N_t), dtype=complex)
    j_spec = np.zeros((N_L, N_t), dtype=complex)
    l      = np.linspace(0, N_L - 1, N_L)
    W, L   = np.meshgrid(w, l)

    A      = wlib.A (t[0 : N_t], E0, omega0, tau0, phi=0)

    rho    = np.zeros((N_L * N_bands, N_L * N_bands), dtype=complex)

    decoherenceMatrix = np.ones_like (H_x)

    P_xo = np.copy(P_x)

    for bdx in range (N_bands):
            for bbdx in range(N_bands):
                for ldx in range(N_L):
                    for lldx in range(N_L):
                        if ( ldx - lldx > 2 * N_L):
                            P_xo[ldx * N_bands + bdx, lldx * N_bands + bbdx] = 0.0
                        if (bdx == bbdx):
                            P_xo[ldx * N_bands + bdx, lldx * N_bands + bbdx] = 0.0
                            P_L[:, ldx * N_bands + bdx, lldx * N_bands + bbdx] = 0.0
                        if (T2 < 0.0):
                            decoherenceMatrix[ldx * N_bands + bdx, lldx * N_bands + bbdx] = 1.0
                        else:
                            LS = LSD[LSDX]
                            sigmaL = 32
                            if (abs(ldx - lldx) >= LS):
                                decoherenceMatrix[ldx * N_bands + bdx, lldx * N_bands + bbdx] = np.exp( - np.power( (abs(ldx - lldx) - LS) / sigmaL, 2.0 ) * dt )

    LR = np.linspace(0, N_L - 1, N_L)


    for ldx in range(N_L):

        rho[ldx * N_bands, ldx * N_bands] = 1.0

    for tdx in range(1, N_t + 1):

        wlib.printProgressBar (tdx, N_t - 1, prefix="     4.1 Progress: ")

        K1 = wlib.RK4Propagator (t[tdx], rho, H_x, P_x, D_x, E0, omega0, tau0, phi, PROP_TYPE=PROP_TYPE)

        K2 = wlib.RK4Propagator (t[tdx] + 0.5 * dt, rho + 0.5 * dt * K1, H_x, P_x, D_x, E0, omega0, tau0, phi, PROP_TYPE=PROP_TYPE)

        K3 = wlib.RK4Propagator (t[tdx] + 0.5 * dt, rho + 0.5 * dt * K2, H_x, P_x, D_x, E0, omega0, tau0, phi, PROP_TYPE=PROP_TYPE)

        K4 = wlib.RK4Propagator (t[tdx] + dt, rho + dt * K3, H_x, P_x, D_x, E0, omega0, tau0, phi, PROP_TYPE=PROP_TYPE)

        rho = rho + (1.0 / 6.0) * (K1 + 2.0 * K2 + 2.0 * K3 + K4) * dt
        rho = decoherenceMatrix * rho

        if SEP_L == 1:    

            for ldx in range(N_L):
                j[ldx, tdx - 1] = np.trace(np.dot(rho, P_L[ldx, :, :]))
        
        else:

            if PROP_TYPE == 0:
                j[0, tdx - 1] = np.trace(np.dot(rho, P_xo))
            else:
                j[0, tdx - 1] = np.trace(np.dot(rho, D_x))

    t = t[0 : N_t]

    for ldx in range(N_L):

        j_norm = np.sqrt(np.sum(np.power(np.absolute(j[ldx, :]), 2.0) * dt))

        j_spec[ldx, :] = np.power(w, 1.0) * np.fft.fftshift(np.fft.fft(np.fft.fftshift(j[ldx, :] * wlib.fftFilter (t, tau0))))

        j_specNorm = np.sqrt(np.sum(np.power(np.absolute(j_spec[ldx, :]), 2.0) * dw))

    Z = np.power(np.absolute(j_spec), 2.0)

    DIRECTORY = "./Data/Wannier1D03/LS_" + str(LSD[LSDX]).zfill(2)

    if not os.path.isdir(DIRECTORY):

        os.mkdir(DIRECTORY)
    
    np.save (DIRECTORY + "/t.npy", t)
    np.save (DIRECTORY + "/w.npy", w)  
    np.save (DIRECTORY + "/j.npy", j)  
    np.save (DIRECTORY + "/j_spec.npy", j_spec)  
    np.save (DIRECTORY + "/A.npy", A)  
    np.save (DIRECTORY + "/paramsT.npy", paramsT)  

    j = np.sum(j, axis=0).real * wlib.fftFilter (t, tau0)
    j_spec = - np.power(w, 1.0) * np.fft.fftshift(np.fft.fft(np.fft.fftshift(j)))

    # Generate Figures

    # # Light

    plt.style.use("ggbLight")

    # # # PNG

    fig, ax = plt.subplots(1, 1)
    ax.plot(t, A.real)
    ax.set_xlabel(r'Time (a.u.)')
    ax.set_ylabel("Vector Potential (a.u.)")
    ax.set_xlim(t[0], t[-1])

    fmax = np.max(np.absolute(A.real))
    ax.set_ylim( - 1.1 * fmax, 1.1 * fmax )

    plt.tight_layout()

    fig.savefig("./Figures/Wannier1D03/light/PNG/vectorPotential.png", format="PNG")
    fig.savefig("./Figures/Wannier1D03/light/PDF/vectorPotential.pdf", format="PDF")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    ax.plot(t, j.real)
    ax.set_xlabel(r'Time (a.u.)')
    ax.set_ylabel("Current (a.u.)")
    ax.set_xlim(t[0], t[-1])

    fmax = np.max(np.absolute(j.real))
    ax.set_ylim( - 1.1 * fmax, 1.1 * fmax )

    plt.tight_layout()

    fig.savefig("./Figures/Wannier1D03/light/PNG/current.png", format="PNG")
    fig.savefig("./Figures/Wannier1D03/light/PDF/current.pdf", format="PDF")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    ax.semilogy(w / omega0, np.power(np.absolute(j_spec), 2.0))
    ax.set_xlabel("Harmonic Order")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_xlim(0.0, 65.0)

    plt.tight_layout()

    fig.savefig("./Figures/Wannier1D03/light/PNG/spectrum.png", format="PNG")
    fig.savefig("./Figures/Wannier1D03/light/PDF/spectrum.pdf", format="PDF")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    ax.semilogy(w / omega0, np.power(np.absolute(j_spec), 2.0), label="Wannier Basis")
    ax.semilogy(w_B / omega0, 1E-1 * np.power(np.absolute(J_B), 2.0), label="Bloch Basis")
    ax.set_xlabel("Harmonic Order")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_xlim(0.0, 65.0)

    ax.legend()

    plt.tight_layout()

    fig.savefig("./Figures/Wannier1D03/light/PNG/spectrumComparison.png", format="PNG")
    fig.savefig("./Figures/Wannier1D03/light/PDF/spectrumComparison.pdf", format="PDF")
    plt.close(fig)

    # # # Dark

    plt.style.use("ggbDark")

    fig, ax = plt.subplots(1, 1)
    ax.plot(t, A.real)
    ax.set_xlabel(r'Time (a.u.)')
    ax.set_ylabel("Vector Potential (a.u.)")
    ax.set_xlim(t[0], t[-1])

    fmax = np.max(np.absolute(A.real))
    ax.set_ylim( - 1.1 * fmax, 1.1 * fmax )

    plt.tight_layout()

    fig.savefig("./Figures/Wannier1D03/dark/vectorPotential.png", format="PNG")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    ax.plot(t, j.real)
    ax.set_xlabel(r'Time (a.u.)')
    ax.set_ylabel("Current (a.u.)")
    ax.set_xlim(t[0], t[-1])

    fmax = np.max(np.absolute(j.real))
    ax.set_ylim( - 1.1 * fmax, 1.1 * fmax )

    plt.tight_layout()

    fig.savefig("./Figures/Wannier1D03/dark/current.png", format="PNG")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    ax.semilogy(w / omega0, np.power(np.absolute(j_spec), 2.0))
    ax.fill_between (w / omega0, np.power(np.absolute(j_spec), 2.0), alpha=0.30)
    ax.axvline(minGap / omega0, linewidth=0.5, alpha=0.5, color='white')
    ax.axvline(maxGap / omega0, linewidth=0.5, alpha=0.5, color='white')
    ax.axvline(cutOff / omega0, linestyle='--', linewidth=0.5, alpha=0.5, color="white")
    ax.set_xlabel("Harmonic Order")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_xlim(0.0, 60.0)
    ax.set_ylim(1E-15, 1E7)

    plt.tight_layout()

    fig.savefig("./Figures/Wannier1D03/dark/spectrum.png", format="PNG")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    ax.semilogy(w / omega0, np.power(np.absolute(j_spec), 2.0), label="Wannier Basis")
    ax.fill_between (w / omega0, np.power(np.absolute(j_spec), 2.0), alpha=0.30)
    ax.semilogy(w_B / omega0, np.power(w_B * np.absolute(J_B), 2.0), label="Bloch Basis")
    ax.axvline(minGap / omega0, linewidth=1.0, alpha=0.75, color='white')
    ax.axvline(maxGap / omega0, linewidth=1.0, alpha=0.75, color='white')
    ax.axvline(cutOff / omega0, linestyle='--', linewidth=1.0, alpha=0.5, color="white")
    ax.set_xlabel("Harmonic Order")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_xlim(0.0, 85.0)
    ax.set_ylim(1E-35, 1E8)
    ax.legend()

    plt.tight_layout()

    fig.savefig("./Figures/Wannier1D03/dark/spectrumComparison.png", format="PNG")
    plt.close(fig)

    print()
    print("  5. Calculation finished.")
    print()