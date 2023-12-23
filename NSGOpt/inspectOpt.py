#%%
import numpy as np
import glob
import os
os.chdir('/home/dbd/PlotkinCollab/Optimization')
import PlotkinD1PatchMatrix as waves

dataname='D1_Patch_Sample_2'
exp_to_fit = waves.data[dataname][[0,1,2,3]]


#datadirs = glob.glob('./output/fitd1d2-D1-D1_Patch_Sample_2_vary_chans_include_CaCC_2tmp_62938/tmp*')
datadirs = glob.glob('/tmp/fitd1d2-D1-D1_Patch_Sample_2_real_morphtmp_356/tmp*')

datadirs.sort(key=os.path.getmtime)

ddir = datadirs[-1]
#traces = []
#for t in ['ivdata--2e-10.npy', 'ivdata-1.5e-10.npy','ivdata-1.25e-10.npy','ivdata-1.75e-10.npy']:
#    traces.append(np.load(ddir+'/'+t))

#%%
from matplotlib import pyplot as plt

for s in range(64):#(len(datadirs)):
    ddir = datadirs[-s]
    traces = []
    ddirs=[]
    for t in ['ivdata--2e-10.npy', 'ivdata-1.5e-10.npy','ivdata-1.25e-10.npy','ivdata-1.75e-10.npy']:
        try:
            traces.append(np.load(ddir+'/'+t))
            ddirs.append(ddir)
        except FileNotFoundError:
            print('fnf')
    plt.figure()
    for t,ddir in zip(traces,ddirs):
        sim_t = np.linspace(0,exp_to_fit[0].wave.x[-1],len(traces[0]))

        plt.plot(sim_t,t,'r')
    for w in exp_to_fit.waves:
        plt.plot(w.wave.x,w.wave.y,'b')
    plt.title(ddir)
plt.show()

#%%
params = np.load(ddir+'/params.pickle')
for k,v in params.items():
    print(k, type(v))
paramlist = [(v.value, k) for k,v in params.items() if hasattr(v,'value')]
#paramlist


#%%

import logging
import numpy as np
import fileinput
import sys
import re
import moose_nerp
from pathlib import Path
from collections import defaultdict
from ajustador.helpers.loggingsystem import getlogger
from ajustador.helpers.copy_param.process_common import create_path
from ajustador.helpers.copy_param.process_common import check_version_build_file_path
from ajustador.helpers.copy_param.process_common import get_file_abs_path
from ajustador.helpers.copy_param.process_common import clone_file
from ajustador.helpers.copy_param.process_common import  write_header
from ajustador.helpers.copy_param.process_morph import clone_and_change_morph_file
from ajustador.helpers.copy_param.process_npz import get_least_fitness_params
from ajustador.helpers.copy_param.process_npz import make_new_file_name_from_npz
from ajustador.helpers.copy_param.process_npz import get_params
from ajustador.helpers.copy_param.process_param_cond import get_namedict_block_start
from ajustador.helpers.copy_param.process_param_cond import get_block_end
from ajustador.helpers.copy_param.process_param_cond import update_morph_file_name_in_cond
from ajustador.helpers.copy_param.process_param_cond import update_conductance_param
from ajustador.helpers.copy_param.process_param_cond import reshape_conds_to_dict
from ajustador.helpers.copy_param.process_param_chan import create_chan_param_relation
from ajustador.helpers.copy_param.process_param_chan import reshape_chans_to_dict
from ajustador.helpers.copy_param.process_param_chan import import_param_chan
from ajustador.helpers.copy_param.process_param_chan import update_chan_param
from ajustador.helpers.copy_param.process_param_chan import chan_param_locator
from ajustador.regulate_chan_kinetics import scale_voltage_dependents_tau_muliplier
from ajustador.regulate_chan_kinetics import offset_voltage_dependents_vshift

logger = getlogger(__name__)
logger.setLevel(logging.INFO)

model = 'd1d2'
neuron_type = 'D1'
fitnum=None
cond_file= 'param_cond.py'
chan_file='param_chan.py'
store_param_path = None

model_path = Path(moose_nerp.__file__.rpartition('/')[0])/model

#logger.info("START STEP 1!!!\n loading npz file: {}.".format(npz_file))
#data = np.load(npz_file)

#logger.info("START STEP 2!!! Prepare param conductances.")
#fit_number, param_data_list = get_least_fitness_params(data, fitnum)
# l#ogger.info("*******Data directory for fit: {}".format(data['tmpdirs'][fit_number]))
#header_line = "# Generated from npzfile: {} of fit number: {}\n".format(
#             npz_file.rpartition('/')[2], fit_number)
#sample_label = npz_file.rpartition('/')[2].rstrip('.npz').split('_')[-1]

fit_number = 0
param_data_list = paramlist
logger.debug("Param_data: {}".format(param_data_list))
conds = get_params(param_data_list, 'Cond_')
non_conds = get_params(param_data_list, 'Cond_', exclude_flag=True)

# Create new path to save param_cond.py and *.p
new_param_path = create_path(store_param_path) if store_param_path else create_path(model_path/'conductance_save')

if cond_file is None:
    cond_file = 'param_cond.py'
#new_param_cond = make_new_file_name_from_npz(data, npz_file,
#                     str(new_param_path), neuron_type, cond_file)
#new_cond_file_name = check_version_build_file_path(str(new_param_cond), neuron_type, fit_number)
#logger.info("START STEP 3!!! Copy \n source : {} \n dest: {}".format(get_file_abs_path(model_path,cond_file), new_cond_file_name))
new_cond_file_name = '/home/dandorman/moose_nerp/moose_nerp/d1d2/conductance_save/param_cond.py'
new_param_cond = clone_file(src_path=model_path, src_file=cond_file, dest_file=new_cond_file_name)

#%%
#logger.info("START STEP 4!!! Extract and modify morph_file from {}".format(new_param_cond))
#sample_label = 'test
# F'
morph_file = clone_and_change_morph_file(new_param_cond, model_path, model, neuron_type, non_conds)

#NOTE: created param_cond.py file in conductance_save directory of moose_nerp squid model.
#NOTE: created and updated morph file.

#logger.info("START STEP 5!!! Renaming morph file after checking version.")
#new_morph_file_name = check_version_build_file_path(morph_file, neuron_type, fit_number)
#Path(str(new_param_path/morph_file)).rename(str(new_morph_file_name))

#logger.info("START STEP 6!!! Renaming morph file after checking version.")

#update_morph_file_name_in_cond(new_cond_file_name, neuron_type, #new_morph_file_name.rpartition('/')[2])
header_line = '\n'
write_header(header_line, new_param_cond)
start_param_cond_block = get_namedict_block_start(new_param_cond, neuron_type)
end_param_cond_block = get_block_end(new_param_cond, start_param_cond_block, r"\)")
conds_dict = reshape_conds_to_dict(conds)
update_conductance_param(new_param_cond, conds_dict, start_param_cond_block, end_param_cond_block)

logger.info("STEP 8!!! start channel processing.")
chans = get_params(param_data_list, 'Chan_')
logger.debug('{}'.format(chans))

if chan_file is None:
    chan_file = 'param_chan.py'
#new_param_chan = make_new_file_name_from_npz(data, npz_file,
#                     str(new_param_path), neuron_type, chan_file)
#new_chan_file_name = check_version_build_file_path(str(new_param_chan), neuron_type, fit_number)
new_chan_file_name = '/home/dandorman/moose_nerp/moose_nerp/d1d2/conductance_save/param_chan.py'
#logger.info("START STEP 9!!! Copy \n source : {} \n dest: {}".format(get_file_abs_path(model_path,chan_file), new_chan_file_name))
new_param_chan = clone_file(src_path=model_path, src_file=chan_file, dest_file=new_chan_file_name)

#logger.info("START STEP 10!!! Preparing channel and gateparams relations.")
start_param_chan_block = get_namedict_block_start(new_param_chan, 'Channels')
end_param_chan_block = get_block_end(new_param_chan, start_param_chan_block, r"^(\s*\))")
chans_dict = reshape_chans_to_dict(chans)
#logger.info("START STEP 11!!! import parameters from param_chan.py. and apply scale Tau and delay SS")
py_param_chan = import_param_chan(model) # import param_chan.py file from model.
chanset = py_param_chan.Channels # Get Channels set from the imported param_chan.py.
for key,value in chans_dict.items():
    chan_name, opt, gate = key
    if opt == 'taumul':
        scale_voltage_dependents_tau_muliplier(chanset, chan_name, gate, np.float(value))
        if 'KaF' in chan_name:
            print(value, chanset)
    elif opt == 'vshift':
        offset_voltage_dependents_vshift(chanset, chan_name, gate, np.float(value))
chan_param_name_relation = create_chan_param_relation(new_param_chan, start_param_chan_block, end_param_chan_block)
param_location = chan_param_locator(new_param_chan, chan_param_name_relation)
update_chan_param(new_param_chan, chan_param_name_relation, chanset, param_location) #Update new param_chan files with new channel params.
write_header(header_line, new_param_chan) # Write header to the new param_chan.py
#logger.info("THE END!!! New files names \n morph: {1} \n param_cond file: {0} \n param_chan file: {2}".format(new_cond_file_name, new_morph_file_name, new_chan_file_name))



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
plt.plot(w.x,w.y)

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#def func(x, a, b, c):
#  return a * np.exp(-b * x) + c
#  #return a * np.log10(b * x) + c
def simple_exp(x, amp, tau):
    return float(amp) * np.exp(-(x-x[0]) / float(tau))
def negative_exp(x, amp, tau):
    return float(amp) * (1-np.exp(-(x-x[0]) / float(tau)))
def double_negative_exp(x, amp1, amp2, tau1, tau2):
    return float(amp1) * (1-np.exp(-(x-x[0]) / float(tau1))) + amp2 * (1-np.exp(-(x-x[0]) / float(tau2)))
func = double_negative_exp
#x = np.linspace(1,5,50)   # changed boundary conditions to avoid division by 0
#y = func(x, 2.5, 1.3, 0.5)
#yn = y + 0.2*np.random.normal(size=len(x))

x = w.x[(w.x>.101) & (w.x<.15)]
yn = w.y[(w.x>.101) & (w.x<.15)]
x = x-x[0]
yn=yn-yn[0]
popt, pcov = curve_fit(func, x, yn, (1,1,1,2),maxfev = 10000)

plt.figure()
plt.plot(x, yn, 'ko', label="Original Noised Data")
plt.plot(x, func(x, *popt), 'r-', label="Fitted Curve")
plt.legend()
plt.show()


#%%
