"""Run INSIDE the original notebook after its artifact detection and settings cells.
In a Jupyter cell: exec(open('test_real_neural_reviewer.py').read())
Only a tiny subset of real neural data is used to keep memory low.
"""
import numpy as np
import pandas as pd
import mne
from neural_artifact_reviewer_standalone import review_artifact_trials

required = ['epochs', 'artifact_events', 'freqs', 'n_cycles', 'section_time_range', 'artifact_buffer_ms', 'section_included_bands']
missing = [k for k in required if k not in globals()]
if missing:
    raise RuntimeError('Run the original notebook detection/settings cells first. Missing: ' + ', '.join(missing))

# Choose a channel/trial that actually contains an artifact event, if one exists.
if not artifact_events.empty:
    example_ch = int(artifact_events.iloc[0]['channel'])
    example_tr = int(artifact_events.iloc[0]['trial'])
else:
    example_ch, example_tr = 0, 0
ch_ids = list(range(len(epochs.ch_names)))
start = max(0, min(example_tr - 2, len(epochs) - 100))
tr_ids = list(range(start, min(start + 100, len(epochs))))

# Important: slice BEFORE computing the spectrogram; do not compute all channels.
raw_subset = epochs[tr_ids].get_data(picks=ch_ids)
decim = 4
small_tfr = mne.time_frequency.tfr_array_morlet(
    raw_subset, sfreq=epochs.info['sfreq'], freqs=freqs,
    n_cycles=n_cycles, output='power', decim=decim, n_jobs=1)
small_tfr = small_tfr.astype(np.float32)

# Original artifact_events uses full-recording indices. Remap to our subset.
subset_events = artifact_events.loc[
    artifact_events['channel'].isin(ch_ids) & artifact_events['trial'].isin(tr_ids)
].copy()
subset_events['channel'] = subset_events['channel'].map({v:i for i,v in enumerate(ch_ids)})
subset_events['trial'] = subset_events['trial'].map({v:i for i,v in enumerate(tr_ids)})

artifact_matrix_subset = np.zeros((len(ch_ids), len(tr_ids)), dtype=int)
for _, ev in subset_events.iterrows():
    artifact_matrix_subset[int(ev['channel']), int(ev['trial'])] = 1

print('Original channels:', ch_ids, 'Original trials:', tr_ids)
print('Artifact events in this test:', len(subset_events))
controller = review_artifact_trials(
    artifact_matrix=artifact_matrix_subset,
    artifact_intervals=subset_events,
    tfr=small_tfr,
    buffer_ms=artifact_buffer_ms,
    section_time_range=section_time_range,
    artifact_events=subset_events,
    threshold_time_range=analysis_time_range,  # Detection window for display markers
    tfr_times=epochs.times[::decim],
    freqs=freqs,
    raw_data=raw_subset,
    raw_times=epochs.times,
    channel_names=[epochs.ch_names[i] for i in ch_ids],
    included_bands=section_included_bands,
    review_csv='real_neural_test_decisions.csv',
)
print('Reviewer opened. Test All Trials / Algorithm Flagged / navigation / Accept-Reject / Settings / hover.')
