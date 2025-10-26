# VGG6 CIFAR-10 Sweep & Model Loader

This repository contains code to run hyperparameter sweeps (using Weights & Biases) and train/evaluate VGG-6 on CIFAR-10. It also includes utilities to load saved models and evaluate them.

This README contains exact commands, environment details, pinned dependency versions (tested), and the seed configuration used to make runs reproducible.

## Tested environment
- OS: Windows 11 (Command Prompt / cmd.exe).
- Python: 3.10
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU (CUDA Version: 12.7).

## Dependency versions (tested)
The project was developed and tested with these package versions. Use the `requirements.txt` below to reproduce the environment.

- python: 3.10
- torch==2.1.0
- torchvision==0.16.0
- numpy==1.26.0
- tqdm==4.65.0
- wandb==0.15.6
- pillow==9.5.0

Note: If you rely on a specific CUDA version, install the matching `torch` / `torchvision` builds from https://pytorch.org.

## Quick setup (Command Prompt)
Open Command Prompt (cmd.exe) or Windows Terminal (cmd profile) and run:

```cmd
python -m venv .venv
REM Activate the venv in cmd
call .venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt
```

## WandB API key
Place your WandB API key in the file `Wandb_API_Key.txt` located at the repository root (same folder as `vgg6_sweep.py`). The scripts will automatically try to read this file and call `wandb.login(key=...)`. Example file contents (single line):

<2aee863945a04807a4a6e22058270bbdacbb02d0>

If the file isn't present, the script will fall back to interactive `wandb.login()`.

## Seed configuration (reproducibility)
This repository sets seeds and deterministic flags to reduce stochasticity between runs. The exact configuration used in `vgg6_sweep.py` (function `setup_seeds`) is:

```python
torch.backends.cudnn.deterministic = True
random.seed(hash("Setting seeds manually") % 2**32 - 1)
np.random.seed(hash("to improve reproducibility") % 2**32 - 1)
torch.manual_seed(hash("and remove stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so that runs are repeatable") % 2**32 - 1)
```

Notes:
- The script uses hashed strings to derive seed values. This provides non-zero seeds stable across runs as long as the code doesn't change.

## How to run

1) Interactive sweep / manual run (recommended for development)

```cmd
# Run the sweep script; it prompts whether to run a sweep or a manual run
python .\vgg6_sweep.py
```

- Choose `(s)`weep to create/use a WandB sweep and optionally start an agent.
- Choose `(m)`anual to run a single configuration (`manual_config`) for quick sanity testing of the sweep.

2) Start a sweep agent (if you already created a sweep)

When `vgg6_sweep.py` asks for a sweep ID, you can provide the ID created by WandB. To start an agent from the script, follow the prompt. The agent is started via the Python API (wandb.agent) so no external CLI call is required.

3) Load and test a saved model

```cmd
# Example: evaluate a saved model file
python .\load_model_from_file.py --model .\models\wise-sweep-9_model.pth --batch-size 64
```

## Notes on tqdm and terminal behavior
- For best behavior (nested progress bars update in-place) use Windows Terminal or PowerShell Core (pwsh). Legacy console hosts may not support the cursor control needed for in-place updates.
- The scripts use `tqdm.write(...)` for status messages so they won't corrupt active progress bars.

## Files of interest
- `vgg6_sweep.py` – main training, sweep configuration, train/eval functions, and interactive runner.
- `load_and_test_model.py` – alternate script (if present) to load/test models.
- `Wandb_API_Key.txt` – place your WandB API key here (single line)

## Troubleshooting
- If you get out-of-memory errors, reduce batch size (`--batch-size`) or use CPU mode by ensuring `DEVICE` resolves to CPU.
- If wandb login fails, run `wandb login` in your terminal or check `Wandb_API_Key.txt` contents.
- If tqdm bars print new lines, try `pwsh` / Windows Terminal or set single-bar mode.

## Reproducing exact environment (requirements file)
Use the provided `requirements.txt` (next to this README) and the virtual environment steps above.

