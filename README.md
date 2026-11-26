# Neural Signal Review Tool

An interactive Python-based software tool designed to accelerate large-scale neural signal quality control in neuroscience research.

## Interface Preview

<img width="1400" height="901" alt="Screenshot 2026-11-26 at 12 51 52 AM" src="https://github.com/user-attachments/assets/153eb0af-0631-4189-bb21-d0c896115dbd" />



## Motivation

While conducting neuroscience research in Professor Karen Moxon's laboratory at UC Davis, I identified a significant bottleneck in our research workflow. Before neural recordings could be used for downstream machine learning experiments, researchers needed to manually inspect thousands of channel-trial combinations.

One dataset alone contained 163 channels and 144 trials, resulting in 23,472 combinations requiring review. The existing workflow was estimated to take approximately two months.

Recognizing this bottleneck, I independently took the initiative to design and develop a software solution.

## Solution

I built an interactive neural signal review interface using Python, Matplotlib, and PyQt that streamlined the manual quality control process.

Key features include:

- Interactive channel and trial navigation.
- Keyboard shortcuts for rapid acceptance and rejection.
- Automatic CSV persistence and crash recovery.
- Bulk rejection of trials.
- Configurable signal visualization and frequency-band exclusions.
- Reusable standalone functionality across research datasets.

## Impact

- Reduced an estimated two-month review workflow to under one week.
- Improved researcher productivity and data review efficiency.
- Received positive feedback from laboratory researchers, who requested that the tool be generalized for additional datasets.
- Refactored the original notebook implementation into a reusable standalone function.

## Technologies

Python, NumPy, Matplotlib, PyQt, Jupyter Notebook, CSV and scientific signal processing.
