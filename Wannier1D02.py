import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt

import Wannier1DLib as wlib

print("")
print("  3. Time Propagation - Bloch Basis")
print("")

N_t = 16384

N_bands = 2

PROP_TYPE = 0

omega0 = 0.014225
E0     = 0.002
T0     = 2.0 * np.pi / omega0
tau0   = 16.0 * T0
phi    = 0.0

T2     = 2.0

# Preliminary Calculations

T2 *= 41.341374575751 

if T2 > 0.0:
    decoherenceConstant = 1.0 / T2
else:
    decoherenceConstant = 0.0

t  = np.linspace(- 2.0 * tau0, 2.0 * tau0, N_t)
dt = t[1] - t[0]

t  = np.append(t, np.array([t[-1] + dt]))

j  = np.zeros(N_t, dtype=complex)

w  = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(N_t, dt))
dw = w[1] - w[0]

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

gap    = energy[2, :] - energy[1, :]
minGap = np.min(gap)
maxGap = np.max(gap)

kdxMax = np.argmin(np.absolute(E0 / omega0 - k))

cutOff = gap[kdxMax]

A = wlib.A (t[0 : N_t], E0, omega0, tau0, phi=0)

# Calculation

H0 = np.load("./Data/Wannier1D01/H_k.npy")
P0 = np.load("./Data/Wannier1D01/P_k.npy")
X0 = np.load("./Data/Wannier1D01/D_k.npy")

rho = np.zeros( ( N_t + 1, (N_b - 1) * N_k, (N_b - 1) * N_k ), dtype=complex )

decoherenceMatrix = np.zeros_like (H0)

dR = np.linspace(0, N_k - 1, N_k)
fR = np.zeros(N_k)


fR = np.exp(- np.power( (dR) / 8, 2.0 ))

fSum = np.sum(2.0 * fR)

for bdx in range(N_bands):
    for bbdx in range(N_bands):
        for kdx in range(N_k):
            if (bdx != bbdx):
                decoherenceMatrix[kdx * N_bands + bdx, kdx * N_bands + bbdx] = 1.0

rho = np.zeros( ( 2, (N_b - 1) * N_k, (N_b - 1) * N_k ), dtype=complex )

for kdx in range(N_k):

    for bdx in range(1, 2):

        rho[0, kdx *  (N_b - 1) + bdx - 1, kdx * (N_b - 1) + bdx - 1] = 1.0

rho[1, :, :] = np.copy(rho[0, :, :])

rho0 = np.copy(rho[0, :, :])
rho1 = np.copy(rho[1, :, :])

crho = np.zeros((N_t, N_k))

# plt.ion()
# fig, ax = plt.subplots()
# fig.canvas.draw()


# ax.set_ylim(0.0, 1.0)
# ax.set_xlim(k[0], k[-1])


for tdx in range(1, N_t + 1):

    wlib.printProgressBar       (tdx, N_t - 1, prefix="     3.1 Progress: ")

    K1 = wlib.RK4Propagator (t[tdx]           , rho0                , H0, P0, X0, E0, omega0, tau0, phi, PROP_TYPE=PROP_TYPE)

    K2 = wlib.RK4Propagator (t[tdx] + 0.5 * dt, rho0 + 0.5 * dt * K1, H0, P0, X0, E0, omega0, tau0, phi, PROP_TYPE=PROP_TYPE)

    K3 = wlib.RK4Propagator (t[tdx] + 0.5 * dt, rho0 + 0.5 * dt * K2, H0, P0, X0, E0, omega0, tau0, phi, PROP_TYPE=PROP_TYPE)

    K4 = wlib.RK4Propagator (t[tdx] + dt      , rho0 + dt * K3      , H0, P0, X0, E0, omega0, tau0, phi, PROP_TYPE=PROP_TYPE)

    drho = (1.0 / 6.0) * (K1 + 2.0 * K2 + 2.0 * K3 + K4) * dt
    rho0 = rho0 + drho
    rho0 = rho0 - decoherenceConstant * decoherenceMatrix * rho0 * dt

    # crho[tdx, :] = np.diag(drho)[1::2].real

    # if (tdx % 16 == 0):
    #     ax.clear()
    #     # ax.set_ylim(0.0, 1.0)
    #     ax.plot(k, crho[tdx, :])
    #     fig.canvas.flush_events()
    #     plt.pause(0.01)


    j[tdx - 1] = np.trace(np.dot(rho0, P0))

j_spec = np.fft.fftshift(np.fft.fft(np.fft.fftshift(wlib.fftFilter (t[0 : N_t], tau0) * j)))

print("\r     3.1 Time propagation in Bloch basis finished                                           ")

t = t[0 : N_t]

j *= wlib.fftFilter (t, tau0)

j_norm = np.sqrt(np.sum(np.power(np.absolute(j), 2.0) * dt))

j_spec = np.fft.fftshift(np.fft.fft(np.fft.fftshift(j)))

j_specNorm = np.sqrt(np.sum(np.power(np.absolute(j_spec), 2.0) * dw))

j_spec *= j_norm / j_specNorm

np.save ("./Data/Wannier1D02/t.npy", t)
np.save ("./Data/Wannier1D02/w.npy", w)
np.save ("./Data/Wannier1D02/j.npy", j)
np.save ("./Data/Wannier1D02/j_spec.npy", j_spec)
np.save ("./Data/Wannier1D02/A.npy", A)

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

fig.savefig("./Figures/Wannier1D02/light/PNG/vectorPotential.png", format="PNG")
fig.savefig("./Figures/Wannier1D02/light/PDF/vectorPotential.pdf", format="PDF")
plt.close(fig)

fig, ax = plt.subplots(1, 1)
ax.plot(t, j.real)
ax.set_xlabel(r'Time (a.u.)')
ax.set_ylabel("Current (a.u.)")
ax.set_xlim(t[0], t[-1])

fmax = np.max(np.absolute(j.real))
ax.set_ylim( - 1.1 * fmax, 1.1 * fmax )

plt.tight_layout()

fig.savefig("./Figures/Wannier1D02/light/PNG/current.png", format="PNG")
fig.savefig("./Figures/Wannier1D02/light/PDF/current.pdf", format="PDF")
plt.close(fig)

fig, ax = plt.subplots(1, 1)
ax.semilogy(w / omega0, np.power(np.absolute(j_spec), 2.0))
ax.set_xlabel("Harmonic Order")
ax.set_ylabel("Intensity (a.u.)")
ax.set_xlim(0.0, 65.0)

plt.tight_layout()

fig.savefig("./Figures/Wannier1D02/light/PNG/spectrum.png", format="PNG")
fig.savefig("./Figures/Wannier1D02/light/PDF/spectrum.pdf", format="PDF")
plt.close(fig)

# # Dark

plt.style.use("ggbDark")

fig, ax = plt.subplots(1, 1)
ax.plot(t, A.real)
ax.set_xlabel(r'Time (a.u.)')
ax.set_ylabel("Vector Potential (a.u.)")
ax.set_xlim(t[0], t[-1])

fmax = np.max(np.absolute(A.real))
ax.set_ylim( - 1.1 * fmax, 1.1 * fmax )

plt.tight_layout()
fig.savefig("./Figures/Wannier1D02/dark/vectorPotential.png", format="PNG")
plt.close(fig)

fig, ax = plt.subplots(1, 1)
ax.plot(t, j.real)
ax.set_xlabel(r'Time (a.u.)')
ax.set_ylabel("Current (a.u.)")
ax.set_xlim(t[0], t[-1])

fmax = np.max(np.absolute(j.real))
ax.set_ylim( - 1.1 * fmax, 1.1 * fmax )

plt.tight_layout()
fig.savefig("./Figures/Wannier1D02/dark/current.png", format="PNG")
plt.close(fig)

fig, ax = plt.subplots(1, 1)
ax.semilogy(w / omega0, np.power(np.absolute(j_spec), 2.0))
ax.fill_between (w / omega0, np.power(np.absolute(j_spec), 2.0), alpha=0.30)
ax.axvline(minGap / omega0, linewidth=0.5, alpha=0.5, color='white')
ax.axvline(maxGap / omega0, linewidth=0.5, alpha=0.5, color='white')
ax.axvline(cutOff / omega0, linestyle='--', linewidth=0.5, alpha=0.5, color="white")
ax.set_xlabel("Harmonic Order")
ax.set_ylabel("Intensity (a.u.)")
ax.set_xlim(0.0, 80.0)
ax.set_ylim(1E-24, 1E5)

plt.tight_layout()
fig.savefig("./Figures/Wannier1D02/dark/spectrum.png", format="PNG")
plt.close(fig)