#%%[markdown]
"""
TODO:
- [ ] Function to evaluate npz file
- [ ] Function to evaluate dill obj
- [x] Compare time to save npz/sas/dill using save params
    - saving the npz/sas params is 2x slower, probably due to recalculating all the fitnesses



"""
#%%
#!%matplotlib
from matplotlib import pyplot as plt
from ajustador.helpers import save_params

#%%
fname = "/data1/dbd/NSGOptData/output.tar-1/NSGOpt/outputD1_Patch_Sample_3_NSG_full_tmp_1754/fitd1d2-D1-D1_Patch_Sample_3_NSG_full_tmp_1754_persist_dill.obj"

fname = "/data1/dbd/NSGOptData/output(1)/NSGOpt/outputD1_Matrix_Sample_2_NSG_full_tmp_1755/fitd1d2-D1-D1_Matrix_Sample_2_NSG_full_tmp_1755_persist_dill.obj"


fname = "/data1/dbd/NSGOptData/output.tar-7/NSGOpt/outputD1_Patch_Sample_3_NSG_full_tmp_1754/fitd1d2-D1-D1_Patch_Sample_3_NSG_full_tmp_1754_persist_dill.obj"


fname = "/data1/dbd/NSGOptData"

fnames = [
    "/home/dandorman/PatchSample4NSGOutput/NSGOpt/outputD1_Patch_Sample_4_NSG_full_tmp_1756/fitd1patchsample2-D1-D1_Patch_Sample_4_NSG_full_tmp_1756_persist_dill.obj",
    "/home/dandorman/Downloads/PatchSample5/NSGOpt/outputD1_Patch_Sample_5_NSG_full_tmp_1756/fitd1patchsample2-D1-D1_Patch_Sample_5_NSG_full_tmp_1756_persist_dill.obj",
]

fnames = [
    "/mnt/ExternalData/NSGOptOutput/output/NSGOpt/outputD1_Patch_Sample_3_NSG_full_tmp_1753/fitd1patchsample2-D1-D1_Patch_Sample_3_NSG_full_tmp_1753_persist_dill.obj"
]

fits = [save_params.load_persist(fname) for fname in fnames]

# from ajustador import drawing
# for fit in fits:
#    drawing.plot_history(fit, fit.measurement)

#%%
npzfile = "/mnt/ExternalData/NSGOptOutput/output/NSGOpt/outputD1_Patch_Sample_3_NSG_full_tmp_1753/fitd1patchsample2-D1-D1_Patch_Sample_3_NSG_full_tmp_1753.npz"

npzfile = "/mnt/ExternalData/NSGOptOutput/matrixsample3/NSGOpt/outputD1_Matrix_Sample_3_NSG_full_tmp_1753/fitd1patchsample2-D1-D1_Matrix_Sample_3_NSG_full_tmp_1753.npz"

#npzfile = "/mnt/ExternalData/NSGOptOutput/matrixsample2/NSGOpt/outputD1_Matrix_Sample_2_NSG_full_tmp_1757/fitd1patchsample2-D1-D1_Matrix_Sample_2_NSG_full_tmp_1757.npz"

import numpy as np

fitnpz = np.load(npzfile)

#%%
n_best = 100

best = np.argmin(fit._history)

best_output = fit[best].waves


from matplotlib import pyplot as plt

plt.ion()
for w in best_output:
    plt.plot(w.wave.x, w.wave.y)

for w in fit.measurement.waves:
    plt.plot(w.wave.x, w.wave.y)

# plt.plot(fit._history,'.')
#%%
#plt.figure()

locs = []
labels = []

for feat in range(fitnpz["fitvals"].shape[1]):
    #plt.barh(feat, fitnpz["fitvals"][best, feat])
    plt.figure()
    plt.plot(fitnpz["fitvals"][:,feat],".")
    plt.title(fitnpz['features'][feat])
    locs.append(feat)
    labels.append(fitnpz["features"][feat])

    plt.tight_layout()
#%%
arr = np.zeros((len(fit), len(fit.fitness_func(fit[0], fit.measurement, full=1))))
for i, f in enumerate(fit):
    arr[i, :] = fit.fitness_func(f, fit.measurement, full=1)

for i in range(arr.shape[1]):
    plt.figure()
    plt.plot(arr[:, i], ".")
    plt.title(fitnpz["features"][i])

#%%
min_index = fitnpz["fitvals"][:,-1].argmin()

min_tmpdir = fitnpz["tmpdirs"][min_index]

print(min_tmpdir)

#%%
min_tmpdir = "/mnt/ExternalData/NSGOptOutput/matrixsample3/NSGOpt/output/tmp/fitd1patchsample2-D1-D1_Matrix_Sample_3_NSG_full_tmp_1753/tmpyrp5815b"

import PlotkinD1PatchMatrix as waves

dataname='D1_Matrix_Sample_2'
exp_to_fit = waves.data[dataname][[0,1,2,3]]


#%%
import glob
plt.figure()
traces = []
for t in glob.glob(min_tmpdir+'/*npy'):
    try:
        traces.append(np.load(t))
    except FileNotFoundError:
        print('fnf')
    for t in traces:
        sim_t = np.linspace(0,exp_to_fit[0].wave.x[-1],len(traces[0]))
        plt.plot(sim_t,t-fitnpz['params'][min_index][0],'r')
    for w in exp_to_fit.waves:
        plt.plot(w.wave.x,w.wave.y,'b')
#   plt.title(ddir)


#%%
# Small fit object to test saving speed

test_obj_path = "/home/dandorman/ajustadorTest/SPN_opt/D1_FitCaPoolToDifShell_pas3_1234567/fitd1d2-D1-D1_FitCaPoolToDifShell_pas3_1234567_persist_dill.obj"

test_obj = save_params.load_persist(test_obj_path)



#%%
#!%%time
save_params.persist(test_obj,'/tmp')

#%%
#%%
#!%%time
save_params.save_params(test_obj,fn='tmp')


#%%
from ajustador.helpers.copy_param import create_npz_param

create_npz_param.create_npz_param(npzfile, 'd1matrixsample2','D1',store_param_path='/home/dandorman/moose_nerp/d1matrixsample2_seed1757')


#%%
# Compare NSG output to local test:
nsg_path = "/mnt/ExternalData/NSGOptOutput/matrixsample3/NSGOpt/output/tmp/fitd1patchsample2-D1-D1_Matrix_Sample_3_NSG_full_tmp_1753/tmpyrp5815b/"
#local_path = "/mnt/ExternalData/NSGOptOutput/matrixsample3/NSGOpt/output/test/"
local_path = "/mnt/ExternalData/NSGOptOutput/matrixsample3/NSGOpt/"


ivstring = 'ivdata-1e-10.npy'

nsgtrace = np.load(nsg_path+ivstring)
localtrace = np.load(local_path+ivstring)
plt.figure()
plt.plot(nsgtrace,'--')
plt.plot(localtrace)
#%%
