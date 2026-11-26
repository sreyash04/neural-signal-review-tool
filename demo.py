"""Demonstrate the neural artifact reviewer using entirely synthetic data."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neural_artifact_reviewer_standalone import review_artifact_trials


def main():
    rng = np.random.default_rng(42)
    n_trials, n_channels = 12, 4
    times = np.linspace(-1, 2, 301)
    freqs = np.geomspace(4, 100, 40)
    raw_times = np.linspace(-1, 2, 1501)

    # Synthetic power and raw waveforms (not derived from patient recordings).
    tfr = rng.lognormal(0, 0.3, (n_trials, n_channels, len(freqs), len(times)))
    raw = rng.normal(0, 0.15, (n_trials, n_channels, len(raw_times)))
    raw += 0.3 * np.sin(2 * np.pi * 10 * raw_times)
    matrix = np.zeros((n_channels, n_trials), dtype=int)
    events = []
    for ch, tr, band, hz, start, end in [
        (0, 0, 'Low', 8, 0.1, 0.5),
        (1, 3, 'Mid', 25, 0.3, 0.7),
        (2, 6, 'High', 75, 0.6, 0.9),
        (3, 9, 'Mid', 30, 0.9, 1.2),
    ]:
        fmask = np.abs(np.log(freqs / hz)) < 0.25
        tmask = (times >= start) & (times <= end)
        tfr[tr, ch][np.ix_(fmask, tmask)] *= 12
        rmask = (raw_times >= start) & (raw_times <= end)
        raw[tr, ch, rmask] += 2 * np.sin(2 * np.pi * hz * raw_times[rmask])
        matrix[ch, tr] = 1
        events.append(dict(channel=ch, trial=tr, band=band, start_s=start, end_s=end))

    events = pd.DataFrame(events)
    controller = review_artifact_trials(
        artifact_matrix=matrix,
        artifact_intervals=events,
        tfr=tfr,
        buffer_ms={'Low': 100, 'Mid': 100, 'High': 100},
        section_time_range=(0, 1),
        artifact_events=events,
        threshold_time_range=(-0.2, 1.2),
        tfr_times=times,
        freqs=freqs,
        raw_data=raw,
        raw_times=raw_times,
        channel_names=[f'Synthetic Ch {i+1}' for i in range(n_channels)],
        review_csv=Path('demo_review_decisions.csv'),
    )
    plt.show(block=True)
    return controller


if __name__ == '__main__':
    main()
