"""Standalone interactive neural artifact trial reviewer.

Requires numpy, pandas, matplotlib; Qt backend recommended for a separate window.
TFR input: ndarray [trials, channels, frequencies, times] of linear power.
For large TFRs, pass np.load("tfr.npy", mmap_mode="r") to avoid loading
the full array into RAM.
Artifact matrix: ndarray [channels, trials], numeric/bool, used for context only.
"""
from pathlib import Path
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, TextBox


def review_artifact_trials(
    artifact_matrix,
    artifact_intervals,
    tfr,
    buffer_ms,
    section_time_range,
    artifact_events,
    threshold_time_range,
    *,
    tfr_times,
    freqs,
    raw_data=None,
    raw_times=None,
    channel_names=None,
    channel_labels=None,
    included_bands=("Low", "Mid", "High"),
    review_csv="review_decisions.csv",
    initial_channel=0,
    initial_trial=0,
    color_range=5.0,
):
    """Launch a Matplotlib reviewer; return its controller.

    Required inputs
    ---------------
    artifact_matrix : [channel, trial] array; informational algorithm summary
    artifact_intervals : DataFrame or iterable of dicts with channel,trial,band,start_s,end_s;
        may be None to use artifact_events for intervals
    tfr : [trial, channel, frequency, time] linear-power ndarray, precomputed
    buffer_ms : dict with Low/Mid/High buffer widths (milliseconds)
    section_time_range : (start,end) seconds, used for green selection overlay
    artifact_events : DataFrame containing channel,trial,band,start_s,end_s;
        presence of (channel,trial) determines 'Algorithm Flagged' mode
    threshold_time_range : (start,end) seconds, used for red threshold markers
    tfr_times, freqs : 1-D axes matching the TFR

    Optional raw_data : [trial,channel,sample], raw_times matching samples
    included_bands : bands whose buffers are drawn (does not change raw event filter)
    review_csv : path for human decisions, independent of artifact_events

    Notes: channel/trial indices are zero-based. Event intervals and axes use seconds.
    This function does not compute or mutate detection results.
    """
    # Keep the caller's original dtype/memory map; never copy the full TFR.
    power = np.asanyarray(tfr)
    if not np.issubdtype(power.dtype, np.number) or np.iscomplexobj(power):
        raise ValueError("tfr must contain real numeric power values")
    matrix = np.asarray(artifact_matrix)
    times = np.asarray(tfr_times, dtype=float)
    frequencies = np.asarray(freqs, dtype=float)
    if power.ndim != 4:
        raise ValueError('tfr must have shape (trials, channels, frequencies, times)')
    n_trials, n_channels, n_freq, n_times = power.shape
    if matrix.shape != (n_channels, n_trials):
        raise ValueError('artifact_matrix must have shape (channels, trials)')
    if times.shape != (n_times,) or frequencies.shape != (n_freq,):
        raise ValueError('tfr_times and freqs must match the last two TFR dimensions')
    if not np.all(np.diff(times) > 0) or not np.all(np.diff(frequencies) > 0) or np.any(frequencies <= 0):
        raise ValueError('TFR times must increase; frequencies must be increasing and positive')
    # Validate one channel when it is loaded, rather than scanning/copying
    # the entire (possibly memory-mapped) dataset on startup.
    if set(buffer_ms) != {'Low','Mid','High'} or any(not np.isfinite(v) or v < 0 for v in buffer_ms.values()):
        raise ValueError('buffer_ms must have nonnegative finite Low, Mid, High values')
    if not set(included_bands).issubset(buffer_ms):
        raise ValueError('included_bands contains an unknown band')
    for label, window in [('section_time_range', section_time_range), ('threshold_time_range', threshold_time_range)]:
        if len(window) != 2 or not np.isfinite(window).all() or window[0] >= window[1]:
            raise ValueError(f'{label} must be an increasing (start, end) pair')
    if not (0 <= initial_channel < n_channels and 0 <= initial_trial < n_trials):
        raise ValueError('Initial channel or trial index out of range')
    events = pd.DataFrame(artifact_events).copy()
    needed = {'channel','trial','band','start_s','end_s'}
    if not needed.issubset(events.columns):
        raise ValueError(f'artifact_events requires columns: {sorted(needed)}')
    intervals = events.copy() if artifact_intervals is None else pd.DataFrame(artifact_intervals).copy()
    if not needed.issubset(intervals.columns):
        raise ValueError(f'artifact_intervals requires columns: {sorted(needed)}')
    for label, frame in [('artifact_events',events),('artifact_intervals',intervals)]:
        if frame.empty:
            continue
        if frame[['channel','trial','start_s','end_s']].isna().any().any():
            raise ValueError(f'{label} has missing channel/trial/start/end values')
        ch = frame['channel'].to_numpy(dtype=int)
        tr = frame['trial'].to_numpy(dtype=int)
        if (ch < 0).any() or (ch >= n_channels).any() or (tr < 0).any() or (tr >= n_trials).any():
            raise ValueError(f'{label} has channel/trial indices outside TFR shape')
    if raw_data is not None:
        raw = np.asarray(raw_data)
        rt = np.asarray(raw_times) if raw_times is not None else None
        if raw.ndim != 3 or raw.shape[:2] != (n_trials, n_channels) or rt is None or rt.shape != (raw.shape[-1],):
            raise ValueError('raw_data must be (trials,channels,samples) and raw_times must match samples')
    else:
        raw, rt = None, None
    if channel_names is None:
        channel_names = [f'Ch{i+1}' for i in range(n_channels)]
    if len(channel_names) != n_channels or (channel_labels is not None and len(channel_labels) != n_channels):
        raise ValueError('channel_names/channel_labels must match number of channels')
    review_path = Path(review_csv)
    decisions = {}
    if review_path.exists():
        with review_path.open(newline='') as f:
            for row in csv.DictReader(f):
                decisions[(int(row['channel_index']),int(row['trial_index']))] = row['decision']
    state = {'channel': int(initial_channel), 'trial': int(initial_trial), 'mode': 'all'}
    settings = {'buffer_ms': dict(buffer_ms), 'included_bands': list(included_bands)}
    flagged = {ch: set() for ch in range(n_channels)}

    def recalculate_flags():
        """Match the notebook's selected-window event-overlap/buffer rule.

        This reclassifies existing detected events; it does NOT rerun detection.
        """
        for ch in flagged:
            flagged[ch].clear()
        selected = settings['included_bands']
        if events.empty:
            return
        for row in events.itertuples(index=False):
            if row.band not in selected:
                continue
            # Distance between the detected interval and chosen section, in seconds.
            gap = max(0., section_time_range[0] - float(row.end_s),
                      float(row.start_s) - section_time_range[1])
            if gap * 1000 <= settings['buffer_ms'][row.band] + 1e-9:
                flagged[int(row.channel)].add(int(row.trial))

    recalculate_flags()
    # Only compute log-power and normalization for the selected channel.
    # Normalization is identical to the original: median/std over all trials
    # and time points, separately for each frequency of that channel.
    channel_cache = {'channel': None, 'db': None, 'center': None, 'spread': None}

    def get_channel_data(ch):
        if channel_cache['channel'] != ch:
            # This creates arrays for ONE channel, not the entire TFR dataset.
            channel_power = np.asarray(power[:, ch, :, :], dtype=np.float64)
            if not np.isfinite(channel_power).all() or (channel_power < 0).any():
                raise ValueError(f'tfr channel {ch} contains invalid power values')
            channel_db = 10 * np.log10(np.maximum(channel_power, np.finfo(float).tiny))
            center_ch = np.median(channel_db, axis=(0, 2))
            spread_ch = np.std(channel_db, axis=(0, 2))
            spread_ch = np.where(spread_ch > 1e-12, spread_ch, 1.0)
            channel_cache.update(channel=ch, db=channel_db,
                                 center=center_ch, spread=spread_ch)
        return channel_cache['db'], channel_cache['center'], channel_cache['spread']
    fig = plt.figure(figsize=(14, 9))
    controller = {'fig': fig, 'state': state, 'decisions': decisions}
    palette = {'Low': 'darkorange', 'Mid': 'magenta', 'High': 'deepskyblue'}

    def save():
        review_path.parent.mkdir(parents=True, exist_ok=True)
        with review_path.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['channel_index','trial_index','decision'])
            for (ch,tr), decision in sorted(decisions.items()):
                writer.writerow([ch,tr,decision])

    def visible():
        return list(range(n_trials)) if state['mode'] == 'all' else sorted(flagged[state['channel']])

    def align():
        choices = visible()
        if choices and state['trial'] not in choices:
            state['trial'] = next((x for x in choices if x >= state['trial']), choices[0])
        return choices

    def set_mode(mode):
        state['mode'] = mode
        draw()

    def move_trial(delta):
        choices = align()
        if choices:
            i = choices.index(state['trial'])
            state['trial'] = choices[max(0, min(len(choices)-1, i+delta))]
        draw()

    def move_channel(delta):
        state['channel'] = max(0, min(n_channels-1, state['channel'] + delta))
        draw()

    def decide(decision):
        decisions[(state['channel'],state['trial'])] = decision
        save()
        draw()

    def clear_decision():
        decisions.pop((state['channel'],state['trial']), None)
        save()
        draw()

    settings_window = {'fig': None}

    def open_settings():
        """Standalone Matplotlib settings window; no notebook globals or PyQt required."""
        old = settings_window['fig']
        if old is not None and plt.fignum_exists(old.number):
            plt.figure(old.number)
            old.canvas.draw_idle()
            return
        sf = plt.figure(figsize=(6.8, 4.2))
        settings_window['fig'] = sf
        sf.suptitle('Selected-window artifact settings', fontsize=12)
        sf.text(.06, .84, 'Include band', fontsize=10)
        sf.text(.53, .84, 'Buffer (ms)', fontsize=10)
        bands = ('Low', 'Mid', 'High')
        checks_ax = sf.add_axes([.06, .34, .38, .44])
        checks = CheckButtons(checks_ax, bands, [b in settings['included_bands'] for b in bands])
        boxes = {}
        for i, band in enumerate(bands):
            ax_text = sf.add_axes([.55, .68 - .16*i, .34, .085])
            boxes[band] = TextBox(ax_text, band, initial=str(settings['buffer_ms'][band]))
        status = sf.text(.06, .17, 'Changes adjust event classification, not detection.', fontsize=9)
        apply_ax = sf.add_axes([.57, .045, .20, .085])
        apply_button = Button(apply_ax, 'Apply')

        def apply(_):
            chosen = [b for b, active in zip(bands, checks.get_status()) if active]
            try:
                if not chosen:
                    raise ValueError('Select at least one band.')
                new_buffers = {b: float(boxes[b].text) for b in bands}
                if any(not np.isfinite(v) or v < 0 for v in new_buffers.values()):
                    raise ValueError('Buffers must be finite and nonnegative.')
            except ValueError as exc:
                status.set_text(f'Error: {exc}')
                sf.canvas.draw_idle()
                return
            settings['buffer_ms'] = new_buffers
            settings['included_bands'] = chosen
            recalculate_flags()
            draw()
            plt.close(sf)
            settings_window['fig'] = None

        apply_button.on_clicked(apply)
        settings_window['widgets'] = (checks, boxes, apply_button, status)
        sf.show()

    # Single persistent hover callback; draw() updates the current heatmap axes.
    hover_target = {'axis': None, 'annotation': None, 'trial': None, 'channel': None}

    def on_hover(event):
        ax = hover_target['axis']
        annotation = hover_target['annotation']
        if ax is None or annotation is None:
            return
        if event.inaxes is not ax or event.xdata is None or event.ydata is None:
            if annotation.get_visible():
                annotation.set_visible(False)
                fig.canvas.draw_idle()
            return
        ti = int(np.argmin(np.abs(times - event.xdata)))
        fi = int(np.argmin(np.abs(frequencies - event.ydata)))
        ch, tr = hover_target['channel'], hover_target['trial']
        annotation.xy = (times[ti], frequencies[fi])
        channel_db, center_ch, spread_ch = get_channel_data(ch)
        annotation.set_text(
            f'Time: {times[ti]:.3f} s\nFrequency: {frequencies[fi]:.1f} Hz\n'
            f'Power: {power[tr, ch, fi, ti]:.4g}\n'
            f'Normalized: {(channel_db[tr,fi,ti]-center_ch[fi])/spread_ch[fi]:.2f} SD'
        )
        annotation.set_visible(True)
        fig.canvas.draw_idle()

    controller['hover_connection'] = fig.canvas.mpl_connect('motion_notify_event', on_hover)
    controller['settings'] = settings
    controller['channel_cache'] = channel_cache
    controller['open_settings'] = open_settings

    def add_button(position, title, callback, color='0.9'):
        ax = fig.add_axes(position)
        button = Button(ax, title, color=color, hovercolor='0.8')
        button.label.set_fontsize(8)
        button.on_clicked(lambda _: callback())
        return button

    def draw():
        choices = align()
        ch, tr = state['channel'], state['trial']
        fig.clear()
        hover_target.update(axis=None, annotation=None, trial=None, channel=None)
        if choices:
            if raw is None:
                ax = fig.add_subplot(111)
                ax_raw = None
                fig.subplots_adjust(left=.08, right=.80, top=.88, bottom=.21)
            else:
                ax, ax_raw = fig.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios':[3,1]})
                fig.subplots_adjust(left=.08, right=.80, top=.88, bottom=.21, hspace=.23)
            channel_db, center_ch, spread_ch = get_channel_data(ch)
            z = (channel_db[tr] - center_ch[:, None]) / spread_ch[:, None]
            im = ax.pcolormesh(times, frequencies, z, shading='auto', vmin=-color_range, vmax=color_range)
            annotation = ax.annotate('', xy=(0, 0), xytext=(14, 14),
                                     textcoords='offset points',
                                     bbox=dict(boxstyle='round', fc='white', alpha=.95),
                                     fontsize=9, zorder=20)
            annotation.set_visible(False)
            hover_target.update(axis=ax, annotation=annotation, trial=tr, channel=ch)
            ax.set_yscale('log')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_xlim(times[0],times[-1])
            ax.set_xlabel('Time (s)')
            fig.colorbar(im, ax=ax, label='Normalized log power (SD from channel median)')
            axes = [ax] + ([ax_raw] if ax_raw is not None else [])
            for axis in axes:
                axis.axvline(threshold_time_range[0], color='red', ls=':', lw=1)
                axis.axvline(threshold_time_range[1], color='red', ls=':', lw=1)
                axis.axvspan(section_time_range[0], section_time_range[1], color='green', alpha=.07)
                for band in settings['included_bands']:
                    buffer = settings['buffer_ms'][band] / 1000
                    if buffer:
                        axis.axvspan(section_time_range[0]-buffer, section_time_range[0], color='gray', alpha=.08)
                        axis.axvspan(section_time_range[1], section_time_range[1]+buffer, color='gray', alpha=.08)
            matches = intervals.loc[(intervals['channel'] == ch) & (intervals['trial'] == tr)]
            for row in matches.itertuples(index=False):
                color = palette.get(row.band, 'purple')
                for axis in axes:
                    axis.axvspan(float(row.start_s), float(row.end_s), color=color, alpha=.15)
            if ax_raw is not None:
                ax_raw.plot(rt, raw[tr,ch], color='black', linewidth=.5)
                ax_raw.set_ylabel('Amplitude')
                ax_raw.set_xlabel('Time (s)')
            decision = decisions.get((ch,tr), 'NOT REVIEWED')
            algorithm = 'FLAGGED' if tr in flagged[ch] else 'UNFLAGGED'
            anatomy = '' if channel_labels is None else f' ({channel_labels[ch]})'
            ax.set_title(f'Trial {tr+1}/{n_trials} | {channel_names[ch]}{anatomy} | Algorithm: {algorithm} | Reviewer: {decision}', fontsize=10)
        else:
            fig.text(.5,.53,f'No artifact-flagged trials for channel {ch+1}',ha='center',va='center',fontsize=16)
        # Always recreate controls, including when the filtered list is empty.
        buttons = [
            add_button([.02,.95,.12,.035],'All Trials',lambda:set_mode('all'), '#B8E6C3' if state['mode']=='all' else '0.9'),
            add_button([.15,.95,.18,.035],'Algorithm Flagged',lambda:set_mode('flagged'), '#B8E6C3' if state['mode']=='flagged' else '0.9'),
            add_button([.60,.95,.105,.035],'Settings',open_settings),
            add_button([.72,.95,.12,.035],'Prev Channel',lambda:move_channel(-1)),
            add_button([.85,.95,.12,.035],'Next Channel',lambda:move_channel(1)),
        ]
        if choices:
            buttons += [
                add_button([.30,.09,.18,.045],'Previous Trial',lambda:move_trial(-1)),
                add_button([.52,.09,.18,.045],'Next Trial',lambda:move_trial(1)),
                add_button([.28,.02,.16,.045],'Accept Trial',lambda:decide('Accepted'),'#B8E6C3'),
                add_button([.46,.02,.16,.045],'Reject Trial',lambda:decide('Rejected'),'#F3B5B5'),
                add_button([.66,.02,.16,.045],'Clear Decision',clear_decision),
            ]
        controller['buttons'] = buttons  # prevent matplotlib widgets from being collected
        fig.canvas.draw_idle()

    controller.update({'recalculate_flags':recalculate_flags, 'draw':draw, 'set_mode':set_mode, 'move_trial':move_trial,
                       'move_channel':move_channel, 'save':save, 'visible_trials':visible})
    draw()
    plt.show(block=False)
    return controller
