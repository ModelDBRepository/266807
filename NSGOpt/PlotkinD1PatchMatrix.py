import pathlib
import glob
from ajustador.loader import CSVSeries

class Params:
    requires = ()
    provides = ('baseline_before', 'baseline_after',
                'steady_after', 'steady_before', 'steady_cutoff',
                'falling_curve_window', 'rectification_window',
                'injection_start', 'injection_end',
                'injection_interval')

    #injection_start = 20e-3 # Current clamp experimental injection current start in time. measured(0 to injection_start in seconds).
    injection_start = 100e-3 # Current clamp experimental injection current start in time. measured(0 to injection_start in seconds).
    #injection_end = 60e-3 # Current clamp experimental injection current end in time. measured(0 to injection_end in seconds).
    injection_end = 600e-3 # Current clamp experimental injection current end in time. measured(0 to injection_end in seconds).

    injection_interval = injection_end - injection_start

    #baseline_before = 20e-3 # Resting membrane potential(RMP), measured (0 to baseline_before time in seconds).
    baseline_before = 30e-3 # Resting membrane potential(RMP), measured (0 to baseline_before time in seconds).
    baseline_after = injection_end + 80e-3 # Membrane potential revert back to RMP after injection_end (time in seconds).

    #steady_after = 40e-3 # Time interval to wait before measuring steady state depolarization
                        # After injection current. measured(0 to steady_after in seconds). cannot be greater that _end.
    steady_after = 300e-3 # Time interval to wait before measuring steady state depolarization
    steady_cutoff = 80
    steady_before = injection_end

    falling_curve_window = 20
    rectification_window = 11

dirname = pathlib.Path(__file__).parent / 'PlotkinD1PatchMatrixWaves'
csvs = sorted(glob.glob('{}/*.csv'.format(dirname)))

params = Params()
data = [CSVSeries(name, params, voltage_units='mV') for name in csvs]
data = {series.name:series for series in data}
