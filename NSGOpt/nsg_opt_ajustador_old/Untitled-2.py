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
fitness = aju.fitnesses.combined_fitness('empty',
                                             response=5,
                                             baseline_pre=2,
                                             baseline_post=2,
                                             rectification=1,
                                             falling_curve_time=1,
                                             spike_time=3,
                                             spike_width=2,
                                             spike_height=10,
                                             spike_latency=2,
                                             spike_count=1,
                                             spike_ahp=2,
                                             ahp_curve=1,
                                             #charging_curve_time=2,
                                             charging_curve_full=10,
                                             spike_range_y_histogram=1,
                                             mean_isi=1,
                                             isi_spread=1,
                                             spike_threshold=.25,
                                             #response_variance=0.05,
                                             post_injection_curve_tau = 2)

#%%
aju.fitnesses.spike_time_fitness(fit[0],fit.measurement)

#%%
fitness(fit[0],fit.measurement)

#%%
fitness(fit[0],fit.measurement,full=True)

#%%
fitness.report(fit[0],fit.measurement,full=True)

#%%
print(fitness.report(fit[0],fit.measurement,full=True))

#%%
aju.fitnesses.spike_time_fitness(fit[0],fit.measurement)
#%%

