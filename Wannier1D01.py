import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt

import Wannier1DLib as wlib

print()
print("  2. Calculate Operators in Bloch and Wannier Bases")
print()

N_bands = 2

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

# Calculation

L   = np.linspace(- N_L // 2, N_L // 2 - 1, N_L)

H_k = np.zeros( (N_k * N_bands, N_k * N_bands), dtype=complex )
P_k = np.zeros( (N_k * N_bands, N_k * N_bands), dtype=complex )
D_k = np.zeros( (N_k * N_bands, N_k * N_bands), dtype=complex )

P_k_temp = np.zeros((N_bands, N_bands, N_k), dtype=complex)
D_k_temp = np.zeros((N_bands, N_bands, N_k), dtype=complex)

H_x = np.zeros( ( N_bands * N_L , N_bands * N_L ) , dtype = complex )
P_x = np.zeros( ( N_bands * N_L , N_bands * N_L ) , dtype = complex )
P_L = np.zeros( ( N_L, N_bands * N_L , N_bands * N_L ) , dtype = complex )
D_x = np.zeros( ( N_bands * N_L , N_bands * N_L ) , dtype = complex )

P_x_temp = np.zeros((N_bands, N_bands, N_L), dtype=complex)
D_x_temp = np.zeros((N_bands, N_bands, N_L), dtype=complex)

statesDK = np.zeros_like(states)

for bdx in range(N_b):
    for xdx in range(N_x):
        statesDK[bdx, :, xdx] = np.dot(- 1j * DK1, states[bdx, :, xdx] * np.exp(1j * k * x[xdx]))


for kdx in range(N_k):

    wlib.printProgressBar (kdx, N_k - 1, prefix="     2.1 Bloch Basis H and P: ")

    for bdx in range (N_b - 1):

        H_k[ kdx * (N_b - 1) + bdx , kdx * (N_b - 1) + bdx ] = energy[bdx + 1, kdx]

        for bbdx in range (N_b - 1):

            if (bdx == bbdx):
                delta_nm = 1.0
            else:
                delta_nm = 0.0

            if (bdx == bbdx):
                P_k[ kdx * (N_b - 1) + bdx, kdx * (N_b - 1) + bbdx ] = delta_nm * k[kdx] + np.sum( np.conj(states[bdx + 1, kdx, :]) * np.dot( - 1j * D1, states[bbdx + 1, kdx, :]) ) 
            else:
                P_k[ kdx * (N_b - 1) + bdx, kdx * (N_b - 1) + bbdx ] = np.sum( np.conj(states[bdx + 1, kdx, :]) * np.dot( - 1j * D1, states[bbdx + 1, kdx, :]) ) 
            
            P_k_temp[bdx, bbdx, kdx] = P_k[ kdx * (N_b - 1) + bdx, kdx * (N_b - 1) + bbdx ]

            if (bdx != bbdx):

                D_k[ kdx * (N_b - 1) + bdx , kdx * (N_b - 1) + bbdx ] = 1j * P_k_temp[bdx, bbdx, kdx] / (energy[bdx + 1, kdx] - energy[bbdx + 1, kdx])
                D_k_temp[ bdx, bbdx, kdx ] = D_k[ kdx * (N_b - 1) + bdx , kdx * (N_b - 1) + bbdx ]
            
            else:

                D_k[ kdx * (N_b - 1) + bdx , kdx * (N_b - 1) + bbdx ] = np.sum( np.conj(states[bdx + 1, kdx, :] * np.exp(1j * k[kdx] * x)) * statesDK[bbdx + 1, kdx, :]  )
                D_k_temp[ bdx, bbdx, kdx ] = D_k[ kdx * (N_b - 1) + bdx , kdx * (N_b - 1) + bbdx ]

print("\r     2.1 Bloch basis Hamiltonian and momentum matrices finished                           ")

for bdx in range (N_bands):

    for ldx in range(N_L):

        wlib.printProgressBar (ldx, N_L, prefix="     2.2 Wannier Basis H(" + str(bdx) + "): ")

        for lldx in range(N_L):

            H_x[ldx * N_bands + bdx, lldx * N_bands + bdx] = np.sum(np.exp(1j * k * (ldx - lldx) * a0) * energy[bdx + 1, :])
        
print("\r     2.2 Wannier basis Hamiltonian finished                                             ")

for bdx in range(N_bands):

    for bbdx in range(N_bands):

        for ldx in range(N_L):
            
            wlib.printProgressBar (ldx, N_L, prefix="     2.3 Wannier Basis D (" + str(bdx) + ", " + str(bbdx) + "): ")

            for lldx in range(N_L):

                if (ldx != lldx):
                    k0 = 0.5 * np.pi / (ldx - lldx) / a0
                else:
                    k0 = 0.0

                P_x[ ldx * N_bands + bdx, lldx * N_bands + bbdx ] = np.sum ( np.exp( 1j * k * (ldx - lldx) * a0) * P_k_temp[bdx, bbdx, :] )                    

                P_L[ abs(ldx - lldx), ldx * N_bands + bdx, lldx * N_bands + bbdx] = P_x[ ldx * N_bands + bdx, lldx * N_bands + bbdx ]

                if (bdx != bbdx):

                    D_x[ ldx * N_bands + bdx, lldx * N_bands + bbdx ] = np.sum( np.exp(1j * k * (ldx - lldx) * a0) * D_k_temp[bdx, bbdx, :] )
                
                else:

                    D_x[ ldx * N_bands + bdx, lldx * N_bands + bbdx ] = np.sum( np.exp(1j * k * (ldx - lldx) * a0) * D_k_temp[bdx, bbdx, :] )

H_x /= N_k
P_x /= N_k
P_L /= N_k
D_x /= N_k

print("\r     2.3 Wannier basis momentum operator finished                                           ")

print("     2.4 Saving Bloch and Wannier Hamiltonian and momentum matrices")

np.save("./Data/Wannier1D01/H_k.npy", H_k)
np.save("./Data/Wannier1D01/P_k.npy", P_k)
np.save("./Data/Wannier1D01/D_k.npy", D_k)

np.save("./Data/Wannier1D01/H_x.npy", H_x)
np.save("./Data/Wannier1D01/P_x.npy", P_x)
np.save("./Data/Wannier1D01/P_L.npy", P_L)
np.save("./Data/Wannier1D01/D_x.npy", D_x)

print("     2.5 Generating figures")

p_VV = np.zeros(N_L, dtype=complex)
p_VC = np.zeros(N_L, dtype=complex)
p_CV = np.zeros(N_L, dtype=complex)
p_CC = np.zeros(N_L, dtype=complex)

d_VV = np.zeros(N_L, dtype=complex)
d_VC = np.zeros(N_L, dtype=complex)
d_CV = np.zeros(N_L, dtype=complex)
d_CC = np.zeros(N_L, dtype=complex)

for ldx in range(N_L):
    d_VC[ldx] = D_x[ (N_L // 2) * N_bands + 0, ldx * N_bands + 1 ]
    d_VV[ldx] = D_x[ (N_L // 2) * N_bands + 0, ldx * N_bands + 0 ]
    d_CC[ldx] = D_x[ (N_L // 2) * N_bands + 1, ldx * N_bands + 1 ]

    p_VV[ldx] = P_x[ (N_L // 2) * N_bands + 0, ldx * N_bands + 0 ]
    p_CC[ldx] = P_x[ (N_L // 2) * N_bands + 1, ldx * N_bands + 1 ]
    p_VC[ldx] = P_x[ (N_L // 2) * N_bands + 0, ldx * N_bands + 1 ] 
    p_CV[ldx] = P_x[ (N_L // 2) * N_bands + 1, ldx * N_bands + 0 ]

# Generate Figures

# # Light

plt.style.use("ggbLight")

cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

# # # PNG

# # # # Bloch

# # # # # Momentum

# # # # # # VC

fig, ax = plt.subplots(1, 1)
ax.plot(k / (np.pi / a0), np.real(P_k_temp[0, 1, :]), label="Real")
ax.plot(k / (np.pi / a0), np.imag(P_k_temp[0, 1, :]), label="Imag.")
ax.set_xlabel(r'Crystal Momentum ($\pi / a_0$)')
ax.set_ylabel("Momentum Matrix Element (a.u.)")
ax.set_xlim(- 1, 1)

ax.legend(ncol=2)

plt.tight_layout()

fig.savefig("./figures/Wannier1D01/light/PNG/Bloch/P/P_VC.png", format="PNG")
fig.savefig("./figures/Wannier1D01/light/PDF/Bloch/P/P_VC.pdf", format="PDF")
plt.close(fig)

# # # # # # CV

fig, ax = plt.subplots(1, 1)
ax.plot(k / (np.pi / a0), np.real(P_k_temp[1, 0, :]), label="Real")
ax.plot(k / (np.pi / a0), np.imag(P_k_temp[1, 0, :]), label="Imag.")
ax.set_xlabel(r'Crystal Momentum ($\pi / a_0$)')
ax.set_ylabel("Momentum Matrix Element (a.u.)")
ax.set_xlim(- 1, 1)

ax.legend(ncol=2)

plt.tight_layout()

fig.savefig("./figures/Wannier1D01/light/PNG/Bloch/P/P_CV.png", format="PNG")
fig.savefig("./figures/Wannier1D01/light/PDF/Bloch/P/P_CV.pdf", format="PDF")
plt.close(fig)

# # # # # # CC

fig, ax = plt.subplots(1, 1)
ax.plot(k / (np.pi / a0), np.real(P_k_temp[1, 1, :]), label="Real")
ax.plot(k / (np.pi / a0), np.imag(P_k_temp[1, 1, :]), label="Imag.")
ax.set_xlabel(r'Crystal Momentum ($\pi / a_0$)')
ax.set_ylabel("Momentum Matrix Element (a.u.)")
ax.set_xlim(- 1, 1)

ax.legend(ncol=2)

plt.tight_layout()

fig.savefig("./figures/Wannier1D01/light/PNG/Bloch/P/P_CC.png", format="PNG")
fig.savefig("./figures/Wannier1D01/light/PDF/Bloch/P/P_CC.pdf", format="PDF")
plt.close(fig)

# # # # # # VV

fig, ax = plt.subplots(1, 1)
ax.plot(k / (np.pi / a0), np.real(P_k_temp[0, 0, :]), label="Real")
ax.plot(k / (np.pi / a0), np.imag(P_k_temp[0, 0, :]), label="Imag.")
ax.set_xlabel(r'Crystal Momentum ($\pi / a_0$)')
ax.set_ylabel("Momentum Matrix Element (a.u.)")
ax.set_xlim(- 1, 1)

ax.legend(ncol=2)

plt.tight_layout()

fig.savefig("./figures/Wannier1D01/light/PNG/Bloch/P/P_VV.png", format="PNG")
fig.savefig("./figures/Wannier1D01/light/PDF/Bloch/P/P_VV.pdf", format="PDF")
plt.close(fig)

# # # # # Dipole

# # # # # # VC

fig, ax = plt.subplots(1, 1)
ax.plot(k / (np.pi / a0), np.absolute(D_k_temp[0, 1, :]))
ax.set_xlabel(r'Crystal Momentum ($\pi / a_0$)')
ax.set_ylabel("Dipole Matrix Element (a.u.)")
ax.set_xlim(- 1, 1)

plt.tight_layout()

fig.savefig("./figures/Wannier1D01/light/PNG/Bloch/D/D_VC.png", format="PNG")
fig.savefig("./figures/Wannier1D01/light/PDF/Bloch/D/D_VC.pdf", format="PDF")
plt.close(fig)

fig, ax = plt.subplots(1, 1)
ax.plot(k / (np.pi / a0), np.absolute(D_k_temp[1, 0, :]))
ax.set_xlabel(r'Crystal Momentum ($\pi / a_0$)')
ax.set_ylabel("Dipole Matrix Element (a.u.)")
ax.set_xlim(- 1, 1)

plt.tight_layout()

fig.savefig("./figures/Wannier1D01/light/PNG/Bloch/D/D_CV.png", format="PNG")
fig.savefig("./figures/Wannier1D01/light/PDF/Bloch/D/D_CV.pdf", format="PDF")
plt.close(fig)

fig, ax = plt.subplots(1, 1)
ax.plot(k / (np.pi / a0), np.absolute(D_k_temp[0, 0, :]))
ax.set_xlabel(r'Crystal Momentum ($\pi / a_0$)')
ax.set_ylabel("Dipole Matrix Element (a.u.)")
ax.set_xlim(- 1, 1)

plt.tight_layout()

fig.savefig("./figures/Wannier1D01/light/PNG/Bloch/D/D_VV.png", format="PNG")
fig.savefig("./figures/Wannier1D01/light/PDF/Bloch/D/D_VV.pdf", format="PDF")
plt.close(fig)

fig, ax = plt.subplots(1, 1)
ax.plot(k / (np.pi / a0), np.absolute(D_k_temp[1, 1, :]), linewidth=0.6)
ax.set_xlabel(r'Crystal Momentum ($\pi / a_0$)')
ax.set_ylabel("Dipole Matrix Element (a.u.)")
ax.set_xlim(- 1, 1)

plt.tight_layout()

fig.savefig("./figures/Wannier1D01/light/PNG/Bloch/D/D_CC.png", format="PNG")
fig.savefig("./figures/Wannier1D01/light/PDF/Bloch/D/D_CC.pdf", format="PDF")
plt.close(fig)

# # # # Wannier

# # # # # Momentum

# # # # VV

fig, ax = plt.subplots(1, 1)

ax.semilogy(L, np.absolute(p_VV), color = cycle[0])
ax.semilogy(L, np.absolute(p_VV), marker='o', color = cycle[0])

ax.set_xlabel("Lattice Site")
ax.set_ylabel("Momentum Matrix Element")

if (N_L < 64):
    ax.set_xlim(- 8, 8)
else:
    ax.set_xlim(- 40, 40)

plt.tight_layout()

fig.savefig("./figures/Wannier1D01/light/PNG/Wannier/P/P_VV.png", format="PNG")
fig.savefig("./figures/Wannier1D01/light/PDF/Wannier/P/P_VV.pdf", format="PDF")
plt.close(fig)

fig, ax = plt.subplots(1, 1)

ax.semilogy(L, np.absolute(p_CC), color = cycle[0])
ax.semilogy(L, np.absolute(p_CC), marker='o', color = cycle[0])

ax.set_xlabel("Lattice Site")
ax.set_ylabel("Momentum Matrix Element")


if (N_L < 64):
    ax.set_xlim(- 8, 8)
else:
    ax.set_xlim(- 40, 40)

plt.tight_layout()

fig.savefig("./figures/Wannier1D01/light/PNG/Wannier/P/P_CC.png", format="PNG")
fig.savefig("./figures/Wannier1D01/light/PDF/Wannier/P/P_CC.pdf", format="PDF")
plt.close(fig)

fig, ax = plt.subplots(1, 1)

ax.semilogy(L, np.absolute(p_VC), color = cycle[0])
ax.semilogy(L, np.absolute(p_VC), marker='o', color = cycle[0])

ax.set_xlabel("Lattice Site")
ax.set_ylabel("Momentum Matrix Element")
ax.set_ylim(1E-3 / N_L, 1E3 / N_L)

if (N_L < 64):
    ax.set_xlim(- 8, 8)
else:
    ax.set_xlim(- 40, 40)

plt.tight_layout()

fig.savefig("./figures/Wannier1D01/light/PNG/Wannier/P/P_VC.png", format="PNG")
fig.savefig("./figures/Wannier1D01/light/PDF/Wannier/P/P_VC.pdf", format="PDF")
plt.close(fig)

fig, ax = plt.subplots(1, 1)
ax.semilogy(L, np.absolute(p_CV), color = cycle[0])
ax.semilogy(L, np.absolute(p_CV), marker='o', color = cycle[0])
ax.set_xlabel("Lattice Site")
ax.set_ylabel("Momentum Matrix Element")
ax.set_ylim(1E-3 / N_L, 1E3 / N_L)

if (N_L < 64):
    ax.set_xlim(- 8, 8)
else:
    ax.set_xlim(- 40, 40)

plt.tight_layout()

fig.savefig("./figures/Wannier1D01/light/PNG/Wannier/P/P_CV.png", format="PNG")
fig.savefig("./figures/Wannier1D01/light/PDF/Wannier/P/P_CV.pdf", format="PDF")
plt.close(fig)

# # # # # # Dipole

fig, ax = plt.subplots(1, 1)
ax.semilogy(L, np.absolute(d_VC), color = cycle[0])
ax.semilogy(L, np.absolute(d_VC), marker='o', color = cycle[0])
ax.set_xlabel("Lattice Site")
ax.set_ylabel("Dipole Matrix Element")
ax.set_ylim(1E-3 / N_L, 1E3 / N_L)

if (N_L < 64):
    ax.set_xlim(- 8, 8)
else:
    ax.set_xlim(- 40, 40)

plt.tight_layout()

fig.savefig("./figures/Wannier1D01/light/PNG/Wannier/D/D_VC.png", format="PNG")
fig.savefig("./figures/Wannier1D01/light/PDF/Wannier/D/D_VC.pdf", format="PDF")
plt.close(fig)

fig, ax = plt.subplots(1, 1)
ax.semilogy(L, np.absolute(d_VV), color = cycle[0])
ax.semilogy(L, np.absolute(d_VV), marker='o', color = cycle[0])
ax.set_xlabel("Lattice Site")
ax.set_ylabel("Dipole Matrix Element")
ax.set_ylim(1E-3 / N_L, 1E3 / N_L)

if (N_L < 64):
    ax.set_xlim(- 8, 8)
else:
    ax.set_xlim(- N_L // 2, N_L // 2)

plt.tight_layout()

fig.savefig("./figures/Wannier1D01/light/PNG/Wannier/D/D_VV.png", format="PNG")
fig.savefig("./figures/Wannier1D01/light/PDF/Wannier/D/D_VV.pdf", format="PDF")
plt.close(fig)

fig, ax = plt.subplots(1, 1)
ax.semilogy(L, np.absolute(d_CC), color = cycle[0])
ax.semilogy(L, np.absolute(d_CC), marker='o', color = cycle[0])
ax.set_xlabel("Lattice Site")
ax.set_ylabel("Dipole Matrix Element")
ax.set_ylim(1E-3 / N_L, 1E3 / N_L)

if (N_L < 64):
    ax.set_xlim(- 8, 8)
else:
    ax.set_xlim(- 40, 40)

plt.tight_layout()

fig.savefig("./figures/Wannier1D01/light/PNG/Wannier/D/D_CC.png", format="PNG")
fig.savefig("./figures/Wannier1D01/light/PDF/Wannier/D/D_CC.pdf", format="PDF")
plt.close(fig)

# # Dark

plt.style.use("ggbDark")

cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

# # Bloch

# # # Momentum

# # # # VC

fig, ax = plt.subplots(1, 1)
ax.plot(k / (np.pi / a0), np.real(P_k_temp[0, 1, :]), label="Real")
ax.plot(k / (np.pi / a0), np.imag(P_k_temp[0, 1, :]), label="Imag.")
ax.set_xlabel(r'Crystal Momentum ($\pi / a_0$)')
ax.set_ylabel("Momentum Matrix Element (a.u.)")
ax.set_xlim(- 1, 1)

ax.legend(ncol=2)

plt.tight_layout()

fig.savefig("./figures/Wannier1D01/dark/Bloch/P/P_VC.png", format="PNG")

plt.close(fig)

# # # # # CV

fig, ax = plt.subplots(1, 1)
ax.plot(k / (np.pi / a0), np.real(P_k_temp[1, 0, :]), label="Real")
ax.plot(k / (np.pi / a0), np.imag(P_k_temp[1, 0, :]), label="Imag.")
ax.set_xlabel(r'Crystal Momentum ($\pi / a_0$)')
ax.set_ylabel("Momentum Matrix Element (a.u.)")
ax.set_xlim(- 1, 1)

ax.legend(ncol=2)

plt.tight_layout()

fig.savefig("./figures/Wannier1D01/dark/Bloch/P/P_CV.png", format="PNG")

plt.close(fig)

# # # # # CC

fig, ax = plt.subplots(1, 1)
ax.plot(k / (np.pi / a0), np.real(P_k_temp[1, 1, :]), label="Real")
ax.plot(k / (np.pi / a0), np.imag(P_k_temp[1, 1, :]), label="Imag.")
ax.set_xlabel(r'Crystal Momentum ($\pi / a_0$)')
ax.set_ylabel("Momentum Matrix Element (a.u.)")
ax.set_xlim(- 1, 1)

ax.legend(ncol=2)

plt.tight_layout()

fig.savefig("./figures/Wannier1D01/dark/Bloch/P/P_CC.png", format="PNG")

plt.close(fig)

# # # # # VV

fig, ax = plt.subplots(1, 1)
ax.plot(k / (np.pi / a0), np.real(P_k_temp[0, 0, :]), label="Real")
ax.plot(k / (np.pi / a0), np.imag(P_k_temp[0, 0, :]), label="Imag.")
ax.set_xlabel(r'Crystal Momentum ($\pi / a_0$)')
ax.set_ylabel("Momentum Matrix Element (a.u.)")
ax.set_xlim(- 1, 1)

ax.legend(ncol=2)

plt.tight_layout()

fig.savefig("./figures/Wannier1D01/dark/Bloch/P/P_VV.png", format="PNG")

plt.close(fig)

# # # # Dipole

# # # # # VC

fig, ax = plt.subplots(1, 1)
ax.plot(k / (np.pi / a0), np.absolute(D_k_temp[0, 1, :]))
ax.set_xlabel(r'Crystal Momentum ($\pi / a_0$)')
ax.set_ylabel("Dipole Matrix Element (a.u.)")
ax.set_xlim(- 1, 1)

plt.tight_layout()

fig.savefig("./figures/Wannier1D01/dark/Bloch/D/D_VC.png", format="PNG")
plt.close(fig)

fig, ax = plt.subplots(1, 1)
ax.plot(k / (np.pi / a0), np.absolute(D_k_temp[1, 0, :]))
ax.set_xlabel(r'Crystal Momentum ($\pi / a_0$)')
ax.set_ylabel("Dipole Matrix Element (a.u.)")
ax.set_xlim(- 1, 1)

plt.tight_layout()

fig.savefig("./figures/Wannier1D01/dark/Bloch/D/D_CV.png", format="PNG")
plt.close(fig)

fig, ax = plt.subplots(1, 1)
ax.plot(k / (np.pi / a0), np.absolute(D_k_temp[0, 0, :]))
ax.set_xlabel(r'Crystal Momentum ($\pi / a_0$)')
ax.set_ylabel("Dipole Matrix Element (a.u.)")
ax.set_xlim(- 1, 1)

plt.tight_layout()

fig.savefig("./figures/Wannier1D01/dark/Bloch/D/D_VV.png", format="PNG")
plt.close(fig)

fig, ax = plt.subplots(1, 1)
ax.plot(k / (np.pi / a0), np.absolute(D_k_temp[1, 1, :]), linewidth=0.6)
ax.set_xlabel(r'Crystal Momentum ($\pi / a_0$)')
ax.set_ylabel("Dipole Matrix Element (a.u.)")
ax.set_xlim(- 1, 1)

plt.tight_layout()

fig.savefig("./figures/Wannier1D01/dark/Bloch/D/D_CC.png", format="PNG")
plt.close(fig)

# # Wannier

# # # Momentum

# # VV

fig, ax = plt.subplots(1, 1)

ax.semilogy(L, np.absolute(p_VV), color = cycle[0])
ax.semilogy(L, np.absolute(p_VV), marker='o', color = cycle[0])

ax.set_xlabel("Lattice Site")
ax.set_ylabel("Momentum Matrix Element")

if (N_L < 64):
    ax.set_xlim(- 8, 8)
else:
    ax.set_xlim(- 40, 40)

plt.tight_layout()

fig.savefig("./figures/Wannier1D01/dark/Wannier/P/P_VV.png", format="PNG")
plt.close(fig)

fig, ax = plt.subplots(1, 1)

ax.semilogy(L, np.absolute(p_CC), color = cycle[0])
ax.semilogy(L, np.absolute(p_CC), marker='o', markersize=0.5, color = cycle[0])

ax.set_xlabel("Lattice Site")
ax.set_ylabel("Momentum Matrix Element")


if (N_L < 64):
    ax.set_xlim(- 8, 8)
else:
    ax.set_xlim(- 40, 40)

plt.tight_layout()

fig.savefig("./figures/Wannier1D01/dark/Wannier/P/P_CC.png", format="PNG")
plt.close(fig)

fig, ax = plt.subplots(1, 1)

ax.semilogy(L, np.absolute(p_VC), color = cycle[0])
ax.semilogy(L, np.absolute(p_VC), marker='o', color = cycle[0])

ax.set_xlabel("Lattice Site")
ax.set_ylabel("Momentum Matrix Element")
ax.set_ylim(1E-3 / N_L, 1E3 / N_L)

if (N_L < 64):
    ax.set_xlim(- 8, 8)
else:
    ax.set_xlim(- 40, 40)

plt.tight_layout()

fig.savefig("./figures/Wannier1D01/dark/Wannier/P/P_VC.png", format="PNG")
plt.close(fig)

fig, ax = plt.subplots(1, 1)
ax.semilogy(L, np.absolute(p_CV), color = cycle[0])
ax.semilogy(L, np.absolute(p_CV), marker='o', color = cycle[0])
ax.set_xlabel("Lattice Site")
ax.set_ylabel("Momentum Matrix Element")
ax.set_ylim(1E-8 / N_L, 1E3 / N_L)

if (N_L < 64):
    ax.set_xlim(- 8, 8)
else:
    ax.set_xlim(- 40, 40)

plt.tight_layout()

fig.savefig("./figures/Wannier1D01/dark/Wannier/P/P_CV.png", format="PNG")
plt.close(fig)

# # # # # Dipole

fig, ax = plt.subplots(1, 1)
ax.semilogy(L, np.absolute(d_VC), color = cycle[0])
ax.semilogy(L, np.absolute(d_VC), marker='o', color = cycle[0])
ax.set_xlabel("Lattice Site")
ax.set_ylabel("Dipole Matrix Element")
ax.set_ylim(1E-3 / N_L, 1E3 / N_L)

if (N_L < 64):
    ax.set_xlim(- 8, 8)
else:
    ax.set_xlim(- 40, 40)

plt.tight_layout()

fig.savefig("./figures/Wannier1D01/dark/Wannier/D/D_VC.png", format="PNG")
plt.close(fig)

fig, ax = plt.subplots(1, 1)
ax.semilogy(L, np.absolute(d_VV), color = cycle[0])
ax.semilogy(L, np.absolute(d_VV), marker='o', color = cycle[0])
ax.set_xlabel("Lattice Site")
ax.set_ylabel("Dipole Matrix Element")
ax.set_ylim(1E-3 / N_L, 1E3 / N_L)

if (N_L < 64):
    ax.set_xlim(- 8, 8)
else:
    ax.set_xlim(- N_L // 2, N_L // 2)

plt.tight_layout()

fig.savefig("./figures/Wannier1D01/dark/Wannier/D/D_VV.png", format="PNG")
plt.close(fig)

fig, ax = plt.subplots(1, 1)
ax.semilogy(L, np.absolute(d_CC), color = cycle[0])
ax.semilogy(L, np.absolute(d_CC), marker='o', color = cycle[0])
ax.set_xlabel("Lattice Site")
ax.set_ylabel("Dipole Matrix Element")
ax.set_ylim(1E-3 / N_L, 1E3 / N_L)

if (N_L < 64):
    ax.set_xlim(- 8, 8)
else:
    ax.set_xlim(- 40, 40)

plt.tight_layout()

fig.savefig("./figures/Wannier1D01/dark/Wannier/D/D_CC.png", format="PNG")
plt.close(fig)