# AI651 Assignment 1

Course: Deep Learning for Space, Time and Graphs, Fall 2026.

This repo has two tasks. Both are about forecasting a time series.

| Item | Where |
| --- | --- |
| Assignment handout | `DL4STG-PA1.pdf` |
| Report (LaTeX source) | `latex_source.txt` |
| Task 1 | `Question 1/` |
| Task 2 | `Question 2 - Leaderboard/` |

## Setup

You need Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r "Question 1/requirements.txt"
```

The same setup works for both tasks. A GPU makes training faster, but the code also runs on a CPU.

## Task 1: forecasting vibration sensors

| File or folder | What it is |
| --- | --- |
| `Question 1/Assignment1.ipynb` | The finished notebook, already run, with all outputs |
| `Question 1/results/design/` | Every table (CSV, TeX) and figure (PDF) from the notebook |
| `Question 1/checkpoints/` | The four trained models |
| `Question 1/harness/` | Helper code given by the course, not changed |

To run it again:

```bash
cd "Question 1"
PA1_PRESET=full jupyter lab
```

Then open `Assignment1.ipynb` and run all cells. The saved models are loaded, so it is fast.

## Task 2: leaderboard forecast

The goal is to forecast the next 168 values of one series with an Autoformer model.

| File | What it does |
| --- | --- |
| `autoformer.py` | The model |
| `task2.py` | Loads the data, trains the model, runs the experiments, writes the forecast |
| `report.py` | Makes the tables and figures for the report |
| `Data/` | The three data files given by the course |
| `results/submission.txt` | The final forecast: 168 values |
| `results/submission_info.json` | P, E and the settings of the final model |
| `results/` (the rest) | Results of every experiment, plus tables and figures |

### The final model

- It is the average of 10 small Autoformer models.
- **P = 89,330** trainable parameters (10 models, 8,933 each).
- **E = 40** training epochs (10 models, 4 epochs each).

### How to run

Run these from the `Question 2 - Leaderboard/` folder.

| Command | What it does | Time |
| --- | --- | --- |
| `python task2.py check` | Tests that the code and the saved results are correct | under 1 minute |
| `python task2.py submit` | Builds the final forecast and writes `results/submission.txt` | a few seconds with the saved models, about 3 minutes on a GPU without them |
| `python task2.py study` | Runs every experiment again | a few minutes with the saved runs, about 2 hours on a GPU without them |
| `python task2.py report` | Rebuilds the tables and figures | under 1 minute |

`submit` only needs the three files in `Data/`.
