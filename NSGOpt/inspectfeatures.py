#%%
import numpy as np
import glob
import os

os.chdir("/home/dbd/PlotkinCollab/Optimization")
import PlotkinD1PatchMatrix as waves

dataname = "D1_Patch_Sample_2"
exp_to_fit = waves.data[dataname][[0, 1, 2, 3]]


# datadirs = glob.glob('./output/fitd1d2-D1-D1_Patch_Sample_2_vary_chans_include_CaCC_2tmp_62938/tmp*')
datadirs = glob.glob(
    "/tmp/fitd1d2-D1-D1_Patch_Sample_2_real_morph_full_charging_curve_tmp_357/tmp*"
)

datadirs.sort(key=os.path.getmtime)

ddir = datadirs[-1]
# traces = []
# for t in ['ivdata--2e-10.npy', 'ivdata-1.5e-10.npy','ivdata-1.25e-10.npy','ivdata-1.75e-10.npy']:
#    traces.append(np.load(ddir+'/'+t))

#%%
from matplotlib import pyplot as plt

for s in range(1, 2):  # (len(datadirs)):
    ddir = datadirs[-s]
    traces = []
    ddirs = []
    for t in [
        "ivdata--2e-10.npy",
        "ivdata-1.5e-10.npy",
        "ivdata-1.25e-10.npy",
        "ivdata-1.75e-10.npy",
    ]:
        try:
            traces.append(np.load(ddir + "/" + t))
            ddirs.append(ddir)
        except FileNotFoundError:
            print("fnf")
    plt.figure()
    for t, ddir in zip(traces, ddirs):
        sim_t = np.linspace(0, exp_to_fit[0].wave.x[-1], len(traces[0]))

        plt.plot(sim_t, t, "r")
    for w in exp_to_fit.waves:
        plt.plot(w.wave.x, w.wave.y, "b")
    plt.title(ddir)
plt.show()

#%%
params = np.load(ddir + "/params.pickle")
for k, v in params.items():
    print(k, type(v))
paramlist = [(v.value, k) for k, v in params.items() if hasattr(v, "value")]
# paramlist


#%%
#%%
import ajustador as aju

aju.features.Feature(exp_to_fit.waves[2]).plot()

dir(aju.features.Feature)

dir(exp_to_fit.waves[1])
exp_to_fit.waves[1]._attributes
exp_to_fit.waves[1].baseline_before

w = exp_to_fit.waves[3].wave
w
from matplotlib import pyplot as plt

plt.figure()
plt.plot(w.x, w.y)


#%%
import ajustador as aju

ivfile = ddir + "/" + "ivdata-1.75e-10.npy"
simtime = 0.7
junction_potential = -5e-3
simresult = aju.optimize.load_simulation(
    ivfile, simtime, junction_potential, exp_to_fit.features
)
plt.figure()
plt.plot(simresult.wave.x, simresult.wave.y)
plt.plot(simresult.charging_curve.x, simresult.charging_curve.y)
plt.axhline(simresult.spike_threshold)
plt.plot(exp_to_fit.waves[2].wave.x, exp_to_fit.waves[2].wave.y)
plt.axhline(exp_to_fit.waves[2].spike_threshold, c="tab:green")
plt.plot(exp_to_fit.waves[2].charging_curve.x, exp_to_fit.waves[2].charging_curve.y)

#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

#%matplotlib qt5
# Number of samplepoints
ex = exp_to_fit.waves[3]
N = len(simresult.charging_curve.x)
Nex = len(ex.charging_curve.x)
# sample spacing
T = simresult.charging_curve.x[1] - simresult.charging_curve.x[0]
Tex = ex.charging_curve.x[1] - ex.charging_curve.x[0]
y = simresult.charging_curve.y
yex = ex.charging_curve.y
yf = scipy.fftpack.fft(y)
yfex = scipy.fftpack.fft(yex)
xf = np.linspace(0.0, 1.0 / (2.0 * T), N / 2)
xfex = np.linspace(0.0, 1.0 / (2.0 * Tex), Nex / 2)
plt.subplot(2, 1, 1)
plt.plot(xf, 2.0 / N * np.abs(yf[0 : int(N / 2)]))
plt.plot(xfex, 2.0 / Nex * np.abs(yfex[0 : int(Nex / 2)]))

plt.subplot(2, 1, 2)
plt.plot(xf[1:], 2.0 / N * np.abs(yf[0 : int(N / 2)])[1:])
plt.plot(xfex[1:], 2.0 / Nex * np.abs(yfex[0 : int(Nex / 2)])[1:])

#%%
def interpolate(wave1, wave2):
    "Interpolate wave1 to wave2.x"
    y = np.interp(wave2.x, wave1.x, wave1.y, left=np.nan, right=np.nan)
    return np.rec.fromarrays((wave2.x, y), names="x,y")
    # from scipy.interpolate import interp1d
    # interp = interp1d(wave1.x,wave1.y,bounds_error=False,fill_value = 'extrapolate')
    # y = interp()
    # return np.rec.fromarrays((wave2.x,interp(wave2.x)),names='x,y')


# wavei = interpolate(simresult.charging_curve,ex.charging_curve)
wavei = interpolate(ex.charging_curve, simresult.charging_curve)
plt.figure()
plt.plot(wavei.x, wavei.y, ":")
plt.plot(ex.charging_curve.x, ex.charging_curve.y, ".-")
plt.plot(simresult.charging_curve.x, simresult.charging_curve.y, "--")

#%%

spikes = [
    simresult.wave[
        (simresult.wave.x < spike.x + 0.01) & (simresult.wave.x > spike.x - 0.01)
    ]
    for spike in simresult.spikes
]
plt.figure()
for spike in spikes:
    plt.plot(spike.x - spike.x[0], spike.y, c="tab:blue")
spikes = [
    ex.wave[(ex.wave.x < spike.x + 0.01) & (ex.wave.x > spike.x - 0.01)]
    for spike in ex.spikes
]
for spike in spikes:
    plt.plot(spike.x - spike.x[0], spike.y, c="tab:orange")

#%%
from ajustador import fitnesses

plt.figure()
for i, ww in enumerate(simresult.spike_ahp_window):
    w = fitnesses.ahp_curve_centered(simresult, i)
    plt.plot(w.wave.x - w.wave.x[0], w.wave.y, c="tab:blue")
for i, ww in enumerate(ex.spike_ahp_window):
    w = fitnesses.ahp_curve_centered(ex, i)

    plt.plot(w.wave.x - w.wave.x[0], w.wave.y, c="tab:orange")

#%%
plt.figure()
for e in exp_to_fit:
    plt.plot(
        e.post_injection_curve.x,
        e.post_injection_curve.y,
        label=e.post_injection_curve_fit,
    )
    plt.legend()

#%%
#!%matplotlib

#%%
