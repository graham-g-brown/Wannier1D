import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
import Wannier1DLib as wlib
import os as os

import sys as sys

# import gplotLib as gplot

os.system("clear")
print("  One-Dimensional Wannier Basis HHG Simulation")
print()
print("  1. Ground State Calculation")
print()

# Simulation parameters

N_x = 512
N_k = 32
N_L = N_k
N_b = 3

a0  = 8.0
v0  = 0.37

# Preliminary calculations and variable definitions
    
k      = np.linspace(- np.pi / a0, np.pi / a0, N_k, endpoint=False) 
dk     = k[1] - k[0]
k     += 0.5 * dk

x      = np.linspace(- 0.5, 0.5, N_x, endpoint=False) * a0
dx     = x[1] - x[0]

print(x[0], x[-1])

X      = np.linspace(- 0.5 * N_L, 0.5 * N_L, N_L * N_x, endpoint=False) * a0

D1, D2 = wlib.getDerivativeMatrices (N_x, dx)

v      = - v0 * (1.0 + np.cos(2.0 * np.pi * x / a0))

V      = np.diag(v)

states = np.zeros((N_b, N_k, N_x), dtype=complex)
energy = np.zeros((N_b, N_k))

# Calculate ground state

# phys rev a 62 032706

for kdx in range(N_k):

    wlib.printProgressBar (kdx, N_k - 1, prefix="     1.1 Calculating Bloch States: ")

    H          = 0.5 * (- D2 - 2j * k[kdx] * D1 + np.power(k[kdx], 2.0) * np.eye(N_x)) + V

    vals, vecs = np.linalg.eig(H)

    idx        = vals.argsort()
    vals       = vals[idx]
    vecs       = vecs[:, idx]

    for bdx in range(N_b):

        energy[bdx, kdx] = vals[bdx].real
        phase0           = np.angle(vecs[N_x // 2, bdx])

        if (bdx % 2 == 1):

            phase0 += - np.sign(k[kdx]) * np.pi / 2.0
        
        u_nk = vecs[:, bdx]

        # norm = np.sqrt(np.sum(np.power(np.absolute(u_nk), 2.0) * dx))
        # print(norm)
        
        # u_nk /= norm 

        states[bdx, kdx, :] = u_nk * np.exp(- 1j * phase0)

plt.plot(np.tile(np.roll(states[1, N_k // 2, :], N_x // 2), 1024))
plt.show()

statesFT = np.fft.fftshift(np.fft.fft(np.fft.fftshift(np.tile(states[1, N_k // 2, :], 1024))))

statesFT[0 : 512 * N_x] = 0.0
stateNew = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(statesFT)))

X = np.linspace(- 0.5, 0.5, 1024 * N_x) * 1024 * a0

phase = np.unwrap(np.angle(stateNew))

plt.plot(X / a0, phase) # np.unwrap(np.angle(states[1, 0, :])))
plt.show()

# print("\r     1.1 Bloch basis Hamiltonian and momentum matrices finished                                ")
# print("     1.2 Saving data and generating figures")

# params = np.array([N_x, N_L, N_k, N_b, a0, v0])

# np.save("./Data/Wannier1D00/params.npy", params)
# np.save("./Data/Wannier1D00/x.npy", x)
# np.save("./Data/Wannier1D00/xL.npy", X)
# np.save("./Data/Wannier1D00/k.npy", k)
# np.save("./Data/Wannier1D00/D1.npy", D1)
# np.save("./Data/Wannier1D00/D2.npy", D2)
# np.save("./Data/Wannier1D00/v.npy", v)
# np.save("./Data/Wannier1D00/states.npy", states)
# np.save("./Data/Wannier1D00/energy.npy", energy)

# # Generate Figures

# # # Light

# # # # PNG

# plt.style.use("ggbLight")

# # # # # Potential

# fig, ax = plt.subplots(1, 1)

# ax.plot(x / a0, v)

# ax.set_xlabel(r'Position ($a_0$)')
# ax.set_ylabel("Potential (a.u.)")
# ax.set_xlim(x[0] / a0, x[-1] / a0)

# plt.tight_layout()

# fig.savefig("./figures/Wannier1D00/light/potential.png", format="PNG")

# plt.close(fig)

# # # # # Band Structure

# fig, ax = plt.subplots(1, 1)

# for bdx in range(N_b):
#     ax.plot(k / (np.pi / a0), 27.2114 * energy[bdx, :], label=r'$n = $' + str(bdx))
# ax.set_xlabel(r'Crystal Momentum ($\pi / a_0$)')
# ax.set_ylabel("Energy (eV)")
# ax.set_xlim(- 1, 1)

# plt.tight_layout()

# fig.savefig("./figures/Wannier1D00/light/bandStructure.png", format="PNG")

# plt.close(fig)

# # # # PDF

# # # # # Potential

# fig, ax = plt.subplots(1, 1)

# ax.plot(x / a0, v)
# ax.set_xlabel(r'Position ($a_0$)')
# ax.set_ylabel("Potential (a.u.)")
# ax.set_xlim(x[0] / a0, x[-1] / a0)
# ax.set_yticks([- 0.75, -0.5, -0.25, 0.0])

# plt.tight_layout()

# fig.savefig("./figures/Wannier1D00/light/PDF/potential.pdf", format="PDF")

# plt.close(fig)

# # # # # Band Structure

# fig, ax = plt.subplots(1, 1)

# for bdx in range(N_b):
#     ax.plot(k / (np.pi / a0), 27.2114 * energy[bdx, :], label=r'$n = $' + str(bdx))
# ax.set_xlabel(r'Crystal Momentum ($\pi / a_0$)')
# ax.set_ylabel("Energy (eV)")
# ax.set_xlim(- 1, 1)

# plt.tight_layout()

# fig.savefig("./figures/Wannier1D00/light/PDF/bandStructure.pdf", format="PDF")

# plt.close(fig)

# # # # Dark

# plt.style.use("ggbDark")

# fig, ax = plt.subplots(1, 1)

# ax.plot(x / a0, v)
# ax.set_xlabel(r'Position ($a_0$)')
# ax.set_ylabel("Potential (a.u.)")
# ax.set_xlim(x[0] / a0, x[-1] / a0)
# ax.set_yticks([- 0.75, -0.5, -0.25, 0.0])

# fig.savefig("./figures/Wannier1D00/dark/potential.png", format="PNG")

# plt.close(fig)

# fig, ax = plt.subplots(1, 1)

# for bdx in range(N_b):
#     ax.plot(k / (np.pi / a0), 27.2114 * energy[bdx, :], label=r'$n = $' + str(bdx))
# ax.set_xlabel(r'Crystal Momentum ($\pi / a_0$)')
# ax.set_ylabel("Energy (eV)")
# ax.set_xlim(- 1, 1)

# ax.set_xticks([- 1.0, - 0.5, 0.0, 0.5, 1.0])
# ax.set_yticks([- 20, - 10, 0, 10])
# plt.tight_layout()

# fig.savefig("./figures/Wannier1D00/dark/bandStructure.png", format="PNG")

# plt.close(fig)
