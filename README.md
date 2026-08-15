# Harbinger.Deprojector
Developed by **Special Research Group 263/4, Confoederatio Research Division** (SRG263-CRD, SRG264-CRD).

---

<div align = "center">
<img src = "https://i.postimg.cc/LX4MvXrQ/31-georeferencer.jpg" width = "50%">
</div>

---

_Don't know what projection it is? No labels? No problem._ Deprojector autonomously converts any arbitrary projection into any other arbitrary projection based off coastline/area extent features alone.

> [!WARNING]
> **Deprojector** is extremely compute hungry, and takes ~5m to run per projection on a modern workstation, and ~10-15m over Colab. You will either need parallel compute, or a lot of patience. Confoederatio developers are working on getting this compute time down.

Plots should show up in Python IDEs like Spyder to track your progress. The Colab version of this local script can be forked [here](https://colab.research.google.com/drive/14aB1gkgp0dbLxjLJhRnHYYWsJVQL85RS?usp=sharing). Unfortunately, since this was first made on Google Colab, the main file is extremely unwieldy, and has not yet been split up. Prepare to see ~4000LOC in your code editor.

This script is not recommended for production due to large compute times. It will also take some more time during its first run as it attempts to install feature-matching libraries (RoMA/LoFTR), which it uses alongside RANSAC in an ensemble model.

## Installation:

Requirements:
1. Ensure Anaconda is installed such that you have access to Anaconda Prompt:
2. Run Anaconda As Administrator > `conda init cmd.exe` to tie it to your Command Prompt system.
3. Restart Command Prompt (Administrator)
4. `conda env create -f environment.yml` (Installs all Python dependencies)
5. `conda activate sam_env` (Ensures accurate dependencies once installed)

<details>
  <summary>Installing dependencies from scratch:</summary>
  
  1. This setup assumes you already have Anaconda and pip installed.
  2. `conda activate sam_env`
  3. `conda install -c conda-forge opencv numpy scipy matplotlib pillow spyder`
  4. Installing core ML libraries:
    1. If you have an NVIDIA GPU: `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`
    2. Otherwise: `conda install pytorch torchvision torchaudio cpuonly -c pytorch`
  5. `pip install kornia`
  6. `pip install romatch`
  
</details>

## Usage:

Deprojector is designed to be run from the command line. It takes in `from_projection.png` as its source image, and will try to mesh warp it to a black and white `to_projection.png` image. Deprojector versions can be selected from the base folder as `deprojector_<version>/`.

Input files:
- `from_projection.png`
- `to_projection.png`
Output files:
- `extent.png`
- `output.png`

It can be run from the command line once input files are replaced by simply calling `python app.py`.
