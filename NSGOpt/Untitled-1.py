# This file contains the minimal amount of code required to produce the code cell you gathered.
#%%
from ajustador.helpers import save_params
fname = '/data1/dbd/NSGOptData'
fnames = ['/mnt/ExternalData/NSGOptOutput/output/NSGOpt/outputD1_Patch_Sample_3_NSG_full_tmp_1753/fitd1patchsample2-D1-D1_Patch_Sample_3_NSG_full_tmp_1753_persist_dill.obj']
fits = [save_params.load_persist(fname) for fname in fnames]

#%%
fit = fits[0]

#%%
fit.fitness_func.report(fit[0],fit.measurement)

#%%
fit.fitness_func(fit[0],fit.measurement,full=1)

#%%
fit.fitness_func(fit[0],fit.measurement)

#%%
fit.fitness_func(fit[0],fit.measurement,full=True)

#%%
fit.fitness_func.report(fit[0],fit.measurement)

#%%
print(fit.fitness_func.report(fit[0],fit.measurement))

#%%
print(fit.fitness_func._parts(fit[0],fit.measurement))

#%%
list(fit.fitness_func._parts(fit[0],fit.measurement))

#%%
print(fit.fitness_func.report(fit[1],fit.measurement))


#%%
import ajustador as aju 

#%%
aju.fitnesses.spike_time_fitness(fit[1],fit.measurement)

#%%
