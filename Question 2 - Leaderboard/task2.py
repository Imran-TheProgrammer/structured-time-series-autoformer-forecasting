"""Task 2 pipeline: data, training, validation studies, figures and the leaderboard forecast.

Run from this folder with the dl4stg-pa1 environment:

    python task2.py check     # checks of the model, the data handling and the selected model
    python task2.py submit    # trains the selected model (if not stored) and writes the 168
                              # forecasts for time_idx 43657..43824 -> results/submission.txt
    python task2.py study     # every multi-seed study behind the selection -> results/runs.csv
    python task2.py report    # tables (CSV + TeX) and numbered PDF figures from the stored runs

The selected model is written down in SELECTED. `submit` needs nothing but the three files in Data/.

One epoch is one pass over every training window. P is the total trainable parameter count and E
the total number of epochs of the model or models that produce the submitted forecast.

Two selection rules are used. First studies (A to I): options within one seed-spread of the best
mean validation RMSE are tied and the cheapest wins. Later studies (S1 to S6): the lowest mean
wins, because the handout ranks accuracy first (section 2.7). They come in two stages, and the pick
of a stage only replaces the configuration kept so far if its five-seed average is better by more
than one seed-spread.
"""
import hashlib
import json
import itertools
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from autoformer import (Autoformer, SeriesDecomposition, aggregate_delays, count_parameters,
                        delay_scores)

HERE = Path(__file__).resolve().parent
DATA, RESULTS = HERE / "Data", HERE / "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PRED_LEN = 168                                   # forecast horizon fixed by the handout
VALIDATION_WEEKS = 26                            # the last 26 blocks of 168 steps are validation
ORIGIN_STEP = 24                                 # a validation forecast starts every 24 steps
CONTINUOUS = [f"feature_{c}" for c in "ABCDEF"]
SKEWED = ["feature_D", "feature_E", "feature_F"]  # strongly right-skewed -> log1p before scaling
FLAGS = [f"feature_{c}" for c in "GHIJ"]         # mutually exclusive 0/1 indicators
VARIABLES = {"all": CONTINUOUS + FLAGS, "continuous": CONTINUOUS, "flags": FLAGS}

BASE = dict(seq_len=168, d=32, kernel=25, factor=1, dropout=0.1, trend_start="fade",
            target="raw", external="past+future", variables="all", daily=True,
            window_norm=False, epochs=3, batch=64, lr=3e-3)

# (name, setting, options, cost rank of each option, seeds, what the study asks)
# Cost rank: more parameters, epochs, inputs or computation = higher; equal rank = same cost.
STUDIES = [
    ("A", "trend_start", ["mean", "fade", "anchor"], [0, 1, 1], 5,
     "Starting trend: window mean (paper), fading level, or fading gap to the last value"),
    ("B", "target", ["raw", "log"], [0, 0], 3, "Target scale: raw or log1p"),
    ("C", "seq_len", [96, 168, 336], [0, 1, 2], 3, "History length"),
    ("D", "d", [8, 16, 32, 64], [0, 1, 2, 3], 3, "Model width (sets the parameter count P)"),
    ("E", "epochs", [1, 2, 3, 4, 6], [0, 1, 2, 3, 4], 3, "Training epochs (sets E)"),
    ("G1", "factor", [1, 2, 3], [0, 1, 2], 3, "Number of delays, k = floor(c ln L)"),
    ("G2", "window_norm", [False, True], [0, 1], 3, "Window-relative normalisation"),
    ("G3", "daily", [False, True], [0, 1], 3, "Daily sin/cos position features"),
    ("F1", "external", ["none", "past", "past+future"], [0, 1, 1], 5,
     "Optional file: none, past only, past + future"),
    ("F2", "variables", ["flags", "continuous", "all"], [0, 1, 2], 3, "Optional file: which variables"),
    # Two more studies aimed at the largest errors, which sit on the highest values.
    ("B2", "target", ["raw", "log"], [0, 0], 5, "Target scale again, now at the final settings"),
    ("I", "variables", ["all", "all+roll", "all+diff", "all+roll+diff"], [0, 1, 1, 2], 3,
     "Extra features from the optional file: 24-step mean and 24-step change"),
]
FINAL_SEEDS = 5

# Settings added after the first studies. A run stores one only when it differs from the
# default below, so the runs trained before it existed keep their keys.
LATER = dict(e_layers=1, marks_net="linear", marks_hidden=1, marks_layers=2)
ROLLING = "all+roll6+roll24+diff6+diff24"        # means and changes over the last 6 and 24 steps

# Later studies, aimed at a lower validation RMSE. They start from the configuration the first
# studies picked. Each option is a set of changes to the configuration picked so far.
# (name, what the study asks, seeds, [(option, changes), ...])
# The studies come in stages. After each stage a five-seed check decides whether its pick
# replaces the configuration kept so far; the next stage starts from whatever was kept.
STAGE_1 = [
    ("S1", "Width and epochs together", 3,
     [(f"width {d}, {e} epochs", dict(d=d, epochs=e)) for d in (16, 32, 64) for e in (3, 10, 20)]),
    ("S1b", "Encoder layers, at the pair S1 picks", 3,
     [("1 encoder layer", dict(e_layers=1)), ("2 encoder layers", dict(e_layers=2))]),
    ("S2", "Target scale, at the configuration S1 picks", 5,
     [("raw", dict(target="raw")), ("log", dict(target="log")),
      ("average of raw and log", dict(target="raw+log"))]),
    ("S3", "How the optional variables enter, at the configuration S2 picks", 5,
     [("one linear layer (as now)", dict(marks_net="linear")),
      ("two-layer network", dict(marks_net="mlp")),
      ("linear layer + means and changes over 6 and 24 steps", dict(variables=ROLLING))]),
]
STAGE_2 = [     # S3 showed that the gain comes from how the optional variables enter
    ("S4", "Two-layer network with and without the means and changes over 6 and 24 steps", 5,
     [("two-layer network (as now)", dict(variables="all")),
      ("two-layer network + means and changes", dict(variables=ROLLING))]),
    ("S5", "Hidden width of that network, at the configuration S4 picks", 5,
     [("model width (as now)", dict(marks_hidden=1)), ("twice the model width", dict(marks_hidden=2)),
      ("four times the model width", dict(marks_hidden=4))]),
    ("S5b", "Two against three layers in that network, at the width S5 picks", 5,
     [("2 layers", dict(marks_layers=2)), ("3 layers", dict(marks_layers=3))]),
    ("S6", "Epochs again, at the configuration S5 picks", 5,
     [(f"{e} epochs", dict(epochs=e)) for e in (2, 3, 4, 6)]),
]
STAGE_3 = [     # the forecasts are still too high on average and too low on the largest values
    ("S7", "More features from the optional file", 5,
     [("(a) means and changes over 6 and 24 steps (as now)", dict(variables=ROLLING)),
      ("(b) as (a), plus 3 and 12 steps", dict(variables=ROLLING + "+roll3+roll12+diff3+diff12")),
      ("(c) as (a), plus 48 steps", dict(variables=ROLLING + "+roll48+diff48")),
      ("(d) as (a), plus spread, minimum, maximum over 24 steps",
       dict(variables=ROLLING + "+std24+min24+max24"))]),
    ("S8a", "Learning rate, at the configuration S7 picks", 5,
     [(f"learning rate {lr}", dict(lr=lr)) for lr in (0.001, 0.003, 0.006)]),
    ("S8b", "Dropout, at the configuration S8a picks", 5,
     [(f"dropout {rate}", dict(dropout=rate)) for rate in (0.0, 0.1, 0.2)]),
    ("S8c", "Decomposition window, at the configuration S8b picks", 5,
     [(f"window {kernel}", dict(kernel=kernel)) for kernel in (13, 25, 49)]),
]
# (stage, its studies, whether its check also asks "not worse in either half of validation" and
#  whether the deployment studies S9 and S10 are run at its pick)
STAGES = [("S1-S3", STAGE_1, False), ("S4-S6", STAGE_2, False), ("S7-S8", STAGE_3, True)]
TWO_YEARS = 17520                                # rows in the "two most recent years" option of S9

# The selected model: what the studies above ended with, and what `submit` trains and averages.
# One unit is two models (raw target and log target). The forecast is the average of five units:
# 10 models of 8,933 parameters and 4 epochs each, so P = 89,330 and E = 40.
SELECTED = dict(seq_len=96, d=16, kernel=25, factor=1, dropout=0.1, trend_start="anchor",
                target="raw+log", external="past+future", variables=ROLLING, daily=True,
                window_norm=False, epochs=4, batch=64, lr=3e-3, marks_net="mlp", marks_hidden=2)
SELECTED_UNITS = 5
SELECTED_P, SELECTED_E = 89330, 40

# The handout's ablation of the optional file, measured on the selected model (chooses nothing).
ABLATION = [("none", dict(external="none")), ("past", dict(external="past")), ("past+future", {}),
            ("flags only", dict(variables="flags")),
            ("continuous only", dict(variables=ROLLING.replace("all", "continuous", 1)))]


# ----------------------------------------------------------------------------- data
class Data:
    """The three supplied files, the chronological split and leakage-safe scaling."""

    def __init__(self):
        train = pd.read_csv(DATA / "student_train.csv")
        test = pd.read_csv(DATA / "student_test.csv")
        external = pd.read_csv(DATA / "optional_external_data.csv")
        self.y = train["value"].to_numpy(float)
        self.n, self.total = len(self.y), len(external)
        assert np.array_equal(train["time_idx"], np.arange(1, self.n + 1))
        assert np.array_equal(external["time_idx"], np.arange(1, self.total + 1))
        assert np.array_equal(test["time_idx"], np.arange(self.n + 1, self.n + PRED_LEN + 1))
        assert self.total == self.n + PRED_LEN and not train["value"].isna().any()
        self.test_time_idx = test["time_idx"].to_numpy()
        self.external = external[CONTINUOUS + FLAGS]
        self.val_start = self.n - VALIDATION_WEEKS * PRED_LEN   # first validation row (0-based)
        self.val_origins = np.arange(self.val_start, self.n - PRED_LEN + 1, ORIGIN_STEP)
        self.truth = np.stack([self.y[o:o + PRED_LEN] for o in self.val_origins])

    def target(self, config):
        """Scaled target of length `total`; rows after the history are NaN on purpose."""
        values = np.log1p(self.y) if config["target"] == "log" else self.y
        centre, spread = values[:self.val_start].mean(), values[:self.val_start].std()
        scaled = np.full(self.total, np.nan)
        scaled[:self.n] = (values - centre) / spread

        def restore(z):                          # back to original units, never below zero
            z = z * spread + centre
            return np.clip(np.expm1(z) if config["target"] == "log" else z, 0, None)
        return torch.tensor(scaled, dtype=torch.float32, device=DEVICE), restore

    def marks(self, config, shuffle=None):
        """Known-ahead features [total, M]: optional-file columns first, then daily sin/cos.

        `shuffle` = (columns, row order) scrambles those columns inside the validation rows;
        it is only used to measure how much each variable matters.
        """
        columns, parts = [], []
        if config["external"] != "none":
            base, *extras = config["variables"].split("+")      # e.g. "all+roll+diff"
            names = VARIABLES[base]
            frame = self.external[names].astype(float).copy()
            for name in names:
                if name in SKEWED:
                    frame[name] = np.log1p(frame[name])
                if name in CONTINUOUS:
                    fit = frame[name].iloc[:self.val_start]
                    frame[name] = (frame[name] - fit.mean()) / fit.std(ddof=0)
            for extra in extras:                 # "roll6": mean of the last 6 steps; "diff6": change
                kind, steps = re.fullmatch(r"(roll|diff|std|min|max)(\d*)", extra).groups()
                steps = int(steps or 24)         # over 6 steps; "std", "min", "max" likewise; no number = 24
                for name in [c for c in names if c in CONTINUOUS]:
                    window = frame[name].rolling(steps, min_periods=1)   # this row and the ones before it
                    if kind == "diff":
                        derived = (frame[name] - frame[name].shift(steps)).fillna(0.0)
                    else:
                        derived = {"roll": window.mean, "min": window.min, "max": window.max,
                                   "std": lambda: window.std(ddof=0)}[kind]()
                    fit = derived.iloc[:self.val_start]
                    frame[f"{name}_{extra}"] = (derived - fit.mean()) / fit.std(ddof=0)
            columns = list(frame.columns)
            if shuffle is not None:
                names, order = shuffle
                block = frame.iloc[self.val_start:self.n]
                frame.iloc[self.val_start:self.n, [columns.index(c) for c in names]] = \
                    block[names].to_numpy()[order]
            parts.append(frame.to_numpy())
        if config["daily"]:
            angle = 2 * np.pi * np.arange(1, self.total + 1) / 24
            parts.append(np.column_stack([np.sin(angle), np.cos(angle)]))
        array = np.column_stack(parts) if parts else np.zeros((self.total, 0))
        return torch.tensor(array, dtype=torch.float32, device=DEVICE), len(columns)


def window_inputs(config, target, marks, n_external, origins):
    """Encoder values and features, decoder features, for forecasts starting at `origins`."""
    seq_len, label_len = config["seq_len"], config["seq_len"] // 2
    enc_rows = origins[:, None] + torch.arange(-seq_len, 0, device=DEVICE)
    dec_rows = origins[:, None] + torch.arange(-label_len, PRED_LEN, device=DEVICE)
    mark_dec = marks[dec_rows]
    if config["external"] == "past" and n_external:      # optional values unknown after the origin
        mark_dec = mark_dec.clone()
        mark_dec[:, label_len:, :n_external] = 0
    return target[enc_rows][..., None], marks[enc_rows], mark_dec


def forecast_scaled(model, config, inputs):
    x, mark_enc, mark_dec = inputs
    if not config["window_norm"]:
        return model(x, mark_enc, mark_dec)
    centre, spread = x.mean(1, keepdim=True), x.std(1, keepdim=True) + 1e-5
    return model((x - centre) / spread, mark_enc, mark_dec) * spread[:, 0] + centre[:, 0]


def build(config, n_marks):
    return Autoformer(n_marks, seq_len=config["seq_len"], label_len=config["seq_len"] // 2,
                      pred_len=PRED_LEN, d=config["d"], kernel=config["kernel"],
                      factor=config["factor"], dropout=config["dropout"],
                      trend_start=config["trend_start"],
                      e_layers=config.get("e_layers", LATER["e_layers"]),
                      marks_net=config.get("marks_net", LATER["marks_net"]),
                      marks_hidden=config.get("marks_hidden", LATER["marks_hidden"]),
                      marks_layers=config.get("marks_layers", LATER["marks_layers"])).to(DEVICE)


def members(config):
    """A configuration stands for one model, or for two when the target is "raw+log": the same
    model trained once on the raw target and once on the log target, their forecasts averaged."""
    if config["target"] == "raw+log":
        return [dict(config, target="raw"), dict(config, target="log")]
    return [config]


def identity(config):
    """The configuration without later settings that sit at their default."""
    return {name: value for name, value in config.items()
            if name not in LATER or value != LATER[name]}


# ----------------------------------------------------------------------------- fit and score
def fit(data, config, seed, train_end=None, train_start=None):
    """Train on windows whose 168 targets all lie before `train_end` (default: the validation start)
    and whose history starts at `train_start` or later (default: the first row)."""
    torch.manual_seed(seed)
    train_end = data.val_start if train_end is None else train_end
    target, _ = data.target(config)
    marks, n_external = data.marks(config)
    model = build(config, marks.shape[1])
    origins = torch.arange((train_start or 0) + config["seq_len"], train_end - PRED_LEN + 1, device=DEVICE)
    assert int(origins.max()) + PRED_LEN <= train_end <= data.n, "training target beyond its limit"
    steps = config["epochs"] * math.ceil(len(origins) / config["batch"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config["lr"],
                                                   total_steps=steps, pct_start=0.15)
    generator = torch.Generator().manual_seed(seed)
    horizon = torch.arange(PRED_LEN, device=DEVICE)
    started = time.perf_counter()
    for _ in range(config["epochs"]):
        model.train()
        for index in torch.randperm(len(origins), generator=generator).split(config["batch"]):
            batch = origins[index.to(DEVICE)]
            inputs = window_inputs(config, target, marks, n_external, batch)
            loss = (forecast_scaled(model, config, inputs) - target[batch[:, None] + horizon]).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            schedule.step()
    return model.eval(), time.perf_counter() - started


@torch.no_grad()
def predict(data, config, model, origins, shuffle=None):
    """Forecasts in original units, [len(origins), 168]."""
    target, restore = data.target(config)
    marks, n_external = data.marks(config, shuffle)
    inputs = window_inputs(config, target, marks, n_external,
                           torch.as_tensor(origins, device=DEVICE))
    assert all(torch.isfinite(part).all() for part in inputs), "a forecast input reads missing data"
    return restore(forecast_scaled(model.eval(), config, inputs).cpu().numpy().astype(float))


def smape_terms(prediction, truth):
    total = np.abs(prediction) + np.abs(truth)
    return np.divide(2 * np.abs(prediction - truth), total, out=np.zeros_like(total), where=total > 0)


def metrics(prediction, truth):
    """The handout's three metrics, pooled over every forecast and lead time."""
    error = prediction - truth
    return {"rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(np.abs(error))),
            "smape": float(100 * np.mean(smape_terms(prediction, truth)))}


def halves(prediction, truth):
    """RMSE of the first and of the second half of the validation forecasts, and the mean error."""
    error, middle = prediction - truth, len(truth) // 2
    return {"rmse_half1": float(np.sqrt(np.mean(error[:middle] ** 2))),
            "rmse_half2": float(np.sqrt(np.mean(error[middle:] ** 2))),
            "mean_error": float(error.mean())}


def block_rmse(prediction, truth):
    return np.sqrt(np.mean((prediction - truth) ** 2, axis=1))      # one value per 168-step block


# ----------------------------------------------------------------------------- stored runs
class Runs:
    """Every (configuration, seed) is trained once; its scores, forecasts and weights are kept."""

    def __init__(self):
        for folder in (RESULTS, RESULTS / "predictions", RESULTS / "models"):
            folder.mkdir(parents=True, exist_ok=True)
        self.path = RESULTS / "runs.csv"
        self.table = pd.read_csv(self.path) if self.path.exists() else pd.DataFrame()

    @staticmethod
    def key(config, seed, train_end=None, train_start=None):
        named = {**identity(config), "seed": seed}
        if train_end is not None:                # only runs with a non-standard training part
            named["train_end"] = int(train_end)
        if train_start is not None:
            named["train_start"] = int(train_start)
        return hashlib.sha1(json.dumps(named, sort_keys=True).encode()).hexdigest()[:12]

    def run(self, data, config, seed, train_end=None, train_start=None):
        """Train once and store. Scores use only forecasts that start after the training part."""
        key = self.key(config, seed, train_end, train_start)
        files = (RESULTS / "predictions" / f"{key}.npy", RESULTS / "models" / f"{key}.pt")
        if len(self.table) and key in set(self.table["key"]):
            if all(path.exists() for path in files):
                return key
            self.table = self.table[self.table["key"] != key]        # files lost: train it again
        model, seconds = fit(data, config, seed, train_end, train_start)
        prediction = predict(data, config, model, data.val_origins)
        unseen = data.val_origins >= (data.val_start if train_end is None else train_end)
        scores = {"rmse": np.nan, "mae": np.nan, "smape": np.nan}
        blocks = np.array([np.nan])
        if unseen.any():
            scores = metrics(prediction[unseen], data.truth[unseen])
            blocks = block_rmse(prediction[unseen], data.truth[unseen])
        row = {"key": key, "seed": seed, **LATER, **config, "P": count_parameters(model),
               "seconds": round(seconds, 1), **scores,
               "block_rmse_mean": float(blocks.mean()), "block_rmse_std": float(blocks.std()),
               "fade": float(model.fade.detach()) if config["trend_start"] != "mean" else np.nan,
               "train_end": int(data.val_start if train_end is None else train_end),
               "train_start": np.nan if train_start is None else int(train_start)}
        np.save(RESULTS / "predictions" / f"{key}.npy", prediction.astype(np.float32))
        torch.save(model.state_dict(), RESULTS / "models" / f"{key}.pt")
        self.table = pd.concat([self.table, pd.DataFrame([row])], ignore_index=True)
        self.table.to_csv(self.path, index=False)
        print(f"  {key} seed {seed}: RMSE {row['rmse']:6.2f}  MAE {row['mae']:6.2f}  "
              f"sMAPE {row['smape']:5.1f}  P {row['P']:6d}  {seconds:5.1f}s", flush=True)
        return key

    def rows(self, keys):
        return self.table.set_index("key").loc[list(keys)].reset_index()

    def prediction(self, key):
        return np.load(RESULTS / "predictions" / f"{key}.npy").astype(float)

    def model(self, data, key):
        config = self.config(key)
        model = build(config, data.marks(config)[0].shape[1])
        model.load_state_dict(torch.load(RESULTS / "models" / f"{key}.pt", map_location=DEVICE))
        return model.eval(), config

    def config(self, key):
        row = self.table.set_index("key").loc[key]
        config = {name: type(BASE[name])(row[name]) for name in BASE}
        for name, default in LATER.items():      # absent or empty for runs stored before it existed
            if name in row.index and not pd.isna(row[name]):
                config[name] = type(default)(row[name])
        return identity(config)

    def unit(self, data, config, seed, train_end=None, train_start=None):
        """The stored run or runs behind one forecast (two for the "raw+log" target)."""
        return [self.run(data, member, seed, train_end, train_start) for member in members(config)]

    def unit_prediction(self, unit):
        return np.mean([self.prediction(key) for key in unit], axis=0)

    def unit_scores(self, data, unit):
        """Validation scores of one unit, and its total parameters and epochs."""
        rows = self.rows(unit)
        scores = ({m: float(rows[m].iloc[0]) for m in ("rmse", "mae", "smape")} if len(unit) == 1
                  else metrics(self.unit_prediction(unit), data.truth))
        return {**scores, **halves(self.unit_prediction(unit), data.truth),
                "P": int(rows["P"].sum()), "epochs": int(rows["epochs"].sum())}


def choose(summary, costs):
    """Options within one seed-spread of the best mean RMSE count as tied. Take the cheapest
    of them; if two cost the same, take the one with the lower mean RMSE."""
    best = summary.loc[summary["rmse_mean"].idxmin()]
    tied = summary[summary["rmse_mean"] <= best["rmse_mean"] + best["rmse_std"]].copy()
    tied["cost"] = [costs[option] for option in tied["option"]]
    return tied.sort_values(["cost", "rmse_mean"]).iloc[0]["option"]


def choose_lowest(summary):
    """Rule of the later studies: the option with the lowest mean validation RMSE wins.

    The first rule gave a tie (options within one seed-spread of the best) to the cheapest
    option. The handout says accuracy is the dominant term of the ranking (section 2.7), so a
    tie now goes to the lower mean RMSE. Also returns the options tied with the winner."""
    best = summary.loc[summary["rmse_mean"].idxmin()]
    tied = summary[summary["rmse_mean"] <= best["rmse_mean"] + best["rmse_std"]]["option"]
    return best["option"], [option for option in tied if option != best["option"]]


def summarise(runs, data, units_by_option):
    """Mean and spread over seeds. One seed is one stored run, or a unit of several runs."""
    rows = []
    for option, units in units_by_option.items():
        units = [[unit] if isinstance(unit, str) else unit for unit in units]
        table = pd.DataFrame([runs.unit_scores(data, unit) for unit in units])
        row = {"option": option, "seeds": len(table), "P": int(table["P"].iloc[0]),
               "epochs": int(table["epochs"].iloc[0])}
        for name in ("rmse", "mae", "smape"):
            row[f"{name}_mean"], row[f"{name}_std"] = table[name].mean(), table[name].std(ddof=1)
        for name in ("rmse_half1", "rmse_half2", "mean_error"):
            row[name] = table[name].mean()
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- studies
def baselines(data):
    """Reference points that need no training, scored on the same validation forecasts."""
    history, origins = data.y, data.val_origins
    marks, _ = data.marks(dict(BASE, external="past+future", variables="all", daily=True))
    design = np.column_stack([marks.cpu().numpy(), np.ones(data.total)])
    fit_rows = design[:data.val_start]
    rows = {"training mean": np.full_like(data.truth, history[:data.val_start].mean()),
            "last value": np.stack([np.full(PRED_LEN, history[o - 1]) for o in origins]),
            "mean of last 168": np.stack([np.full(PRED_LEN, history[o - PRED_LEN:o].mean()) for o in origins])}
    for label, forward, back in (("raw target", lambda v: v, lambda v: v),
                                 ("log target", np.log1p, np.expm1)):
        weights = np.linalg.solve(fit_rows.T @ fit_rows + 1e-3 * np.eye(design.shape[1]),
                                  fit_rows.T @ forward(history[:data.val_start]))
        rows[f"same-time linear fit on optional file ({label})"] = np.stack(
            [np.clip(back(design[o:o + PRED_LEN] @ weights), 0, None) for o in origins])
    return pd.DataFrame([{"baseline": name, **metrics(pred, data.truth)} for name, pred in rows.items()])


def study():
    data, runs = Data(), Runs()
    print(f"device {DEVICE}; training rows 1..{data.val_start}; validation rows "
          f"{data.val_start + 1}..{data.n}; {len(data.val_origins)} validation forecasts")
    reference = baselines(data)
    reference.to_csv(RESULTS / "baselines.csv", index=False)
    print(reference.round(2).to_string(index=False))

    config, record = dict(BASE), []
    for name, setting, options, costs, n_seeds, question in STUDIES:
        if setting == "variables" and config["external"] == "none":
            continue                             # nothing to choose if the file is not used
        print(f"\nStudy {name}: {question}", flush=True)
        keys = {}
        for option in options:
            print(f" {setting} = {option}", flush=True)
            keys[str(option)] = [runs.run(data, dict(config, **{setting: option}), seed)
                                 for seed in range(n_seeds)]
        summary = summarise(runs, data, keys)
        winner = choose(summary, {str(o): c for o, c in zip(options, costs)})
        chosen = type(BASE[setting])(next(o for o in options if str(o) == winner))
        config[setting] = chosen
        print(summary.round(2).to_string(index=False))
        print(f" -> chosen: {setting} = {chosen}", flush=True)
        record.append({"study": name, "setting": setting, "question": question, "chosen": str(chosen),
                       "options": [str(o) for o in options], "costs": costs, "keys": keys})

    print("\nConfiguration picked by the first studies:", config, flush=True)
    start_keys = [runs.run(data, config, seed) for seed in range(FINAL_SEEDS)]
    extra = checks(data, runs, config, start_keys)

    # One model or the average of five seeds? Averaged if the gain is larger than one seed-spread.
    single = runs.rows(start_keys)["rmse"]
    average = metrics(np.mean([runs.prediction(key) for key in start_keys], axis=0), data.truth)["rmse"]
    deploy = {"average_seeds": bool(single.mean() - average > single.std(ddof=1)),
              "train_rows": "to validation", "units": FINAL_SEEDS}
    print(f"\nSingle model {single.mean():.2f} ± {single.std(ddof=1):.2f}, average of seeds {average:.2f}"
          f" -> average the seeds: {deploy['average_seeds']}", flush=True)

    # Later studies in stages. After each stage a check decides whether its pick replaces the
    # configuration kept so far.
    final_config, final_units = config, [[key] for key in start_keys]
    later, decision = [], []
    deployment = None
    for stage, studies, strict in STAGES:
        picked, stage_record = later_studies(data, runs, final_config, studies)
        if strict:                               # S9 and S10: how the pick would be trained and averaged
            deployment = deployment_studies(data, runs, picked)
        verdict = accept(data, runs, final_config, picked, stage, strict)
        new_units = verdict.pop("new_units")
        if verdict["keep"]:
            final_config, final_units = picked, new_units
        # S9 and S10 were measured at `picked`. They apply when that is the final configuration:
        # either it was kept, or the studies picked the configuration that was already final.
        if strict and (verdict["keep"] or not verdict["changed"]):
            deploy.update(train_rows=deployment["train_rows"], units=deployment["units"])
            deployment["applied"] = True
        elif strict:
            deployment["applied"] = False
        later += stage_record
        decision.append(verdict)
    pd.concat([verdict.pop("table") for verdict in decision]).to_csv(RESULTS / "acceptance.csv", index=False)

    # The optional file, with and without, on the final configuration.
    print("\nAblation of the optional file on the final configuration", flush=True)
    ablation = {label: [runs.unit(data, dict(final_config, **change), seed) for seed in range(FINAL_SEEDS)]
                for label, change in ABLATION}
    print(summarise(runs, data, ablation).round(2).to_string(index=False, line_width=250), flush=True)

    (RESULTS / "studies.json").write_text(json.dumps(
        {"final_config": identity(final_config), "final_units": final_units, "start_config": config,
         "studies": record, "checks": extra, "later": later, "acceptance": decision,
         "deployment": deployment, "deploy": deploy,
         "ablation": {"study": "ablation", "keys": ablation}}, indent=1))
    analyse(data, runs, final_config, final_units, record, later, ablation)
    same = identity(final_config) == identity(SELECTED) and deploy["units"] == SELECTED_UNITS \
        and deploy["train_rows"] == "to validation" and deploy["average_seeds"]
    print(f"\nThe studies end at the selected model written in SELECTED: {same}", flush=True)


def later_studies(data, runs, config, studies):
    """One stage of later studies. Each one starts from the configuration the one before it picked."""
    record = []
    for name, question, n_seeds, options in studies:
        print(f"\nStudy {name}: {question}", flush=True)
        units = {}
        for label, change in options:
            print(f" {label}", flush=True)
            units[label] = [runs.unit(data, dict(config, **change), seed) for seed in range(n_seeds)]
        summary = summarise(runs, data, units)
        winner, tied = choose_lowest(summary)
        config = dict(config, **dict(options)[winner])
        print(summary.round(2).to_string(index=False, line_width=250))
        print(f" -> chosen: {winner}" + (f"   (within one seed-spread of it: {'; '.join(tied)})" if tied else ""),
              flush=True)
        record.append({"study": name, "question": question, "chosen": winner, "tied_with_chosen": tied,
                       "options": [label for label, _ in options], "changes": dict(options),
                       "keys": units})
    return config, record


def accept(data, runs, current, new, stage, strict=False):
    """Five seeds of each configuration. The new one replaces the current one only if the average
    of its five seeds' forecasts has a lower validation RMSE by more than one seed-spread, taken
    as the larger of the two configurations' spreads of single-seed RMSE. With `strict`, it must
    also be no worse than the current one in each half of validation."""
    print(f"\nFinal check after {stage}: the current configuration against the pick of these studies",
          flush=True)
    rows, units = [], {}
    for label, config in (("current", current), ("new", new)):
        units[label] = [runs.unit(data, config, seed) for seed in range(FINAL_SEEDS)]
        single = pd.DataFrame([runs.unit_scores(data, unit) for unit in units[label]])
        forecast = np.mean([runs.unit_prediction(unit) for unit in units[label]], axis=0)
        average = {**metrics(forecast, data.truth), **halves(forecast, data.truth)}
        rows.append({"stage": stage, "configuration": label, "seeds": FINAL_SEEDS,
                     "P_total": int(single["P"].sum()),
                     "E_total": int(single["epochs"].sum()),
                     "single_rmse_mean": single["rmse"].mean(), "single_rmse_std": single["rmse"].std(ddof=1),
                     "average_rmse": average["rmse"], "average_mae": average["mae"],
                     "average_smape": average["smape"], "average_rmse_half1": average["rmse_half1"],
                     "average_rmse_half2": average["rmse_half2"], "average_mean_error": average["mean_error"]})
    table = pd.DataFrame(rows)
    gain = float(table["average_rmse"].iloc[0] - table["average_rmse"].iloc[1])
    spread = float(table["single_rmse_std"].max())
    changed = identity(new) != identity(current)
    both_halves = bool((table[["average_rmse_half1", "average_rmse_half2"]].iloc[1]
                        <= table[["average_rmse_half1", "average_rmse_half2"]].iloc[0]).all())
    keep = bool(changed and gain > spread and (both_halves or not strict))
    print(table.round(2).to_string(index=False, line_width=250))
    print(f" new configuration differs: {changed}; gain of the five-seed average {gain:.2f}; "
          f"one seed-spread {spread:.2f}; no worse in either half: {both_halves}"
          f"{'' if strict else ' (not part of this check)'} -> keep the new configuration: {keep}", flush=True)
    return {"stage": stage, "keep": keep, "changed": changed, "gain": gain, "one_seed_spread": spread,
            "no_worse_in_either_half": both_halves,
            "new_config": identity(new), "new_units": units["new"],
            "table": table.assign(gain=gain, one_seed_spread=spread, kept=keep)}


def deployment_studies(data, runs, config):
    """S9 and S10, at the configuration the stage picked. They decide which rows the final models
    train on and how many units are averaged. They only take effect if that configuration is kept."""
    # S9: each half of validation is forecast by models trained without the 13 weeks just before
    # it, with them, and on only the two years just before it.
    print("\nStudy S9: recent data, at the configuration S8 picks", flush=True)
    half = (data.n - data.val_start) // 2                  # 13 weeks
    middle = data.val_start + half
    folds = [("first half of validation", data.val_origins + PRED_LEN <= middle, data.val_start),
             ("second half of validation", data.val_origins >= middle, middle)]
    rows, stored = [], {}
    for fold, scored, start in folds:
        for option, end, first in (("without the 13 weeks before it", start - half, None),
                                   ("with the 13 weeks before it", start, None),
                                   ("only the two years before it", start, start - TWO_YEARS)):
            end = None if end == data.val_start and first is None else end     # the stored standard runs
            units = [runs.unit(data, config, seed, end, first) for seed in range(FINAL_SEEDS)]
            forecasts = np.stack([runs.unit_prediction(unit)[scored] for unit in units])
            truth = data.truth[scored]
            scores = pd.DataFrame([{**metrics(f, truth), "mean_error": float((f - truth).mean())}
                                   for f in forecasts])
            rows.append({"fold": fold, "option": option, "seeds": len(units), "forecasts": int(scored.sum()),
                         "trained_on_time_idx": f"{(first or 0) + 1}..{data.val_start if end is None else end}",
                         **{f"{m}_mean": scores[m].mean() for m in ("rmse", "mae", "smape")},
                         **{f"{m}_std": scores[m].std(ddof=1) for m in ("rmse", "mae", "smape")},
                         "mean_error": scores["mean_error"].mean(),
                         "rmse_of_seed_average": metrics(forecasts.mean(axis=0), truth)["rmse"]})
            stored[f"{fold}, {option}"] = units
    recent = pd.DataFrame(rows)
    recent.to_csv(RESULTS / "walk_forward_S9.csv", index=False)
    print(recent.round(2).to_string(index=False, line_width=250))
    score = recent.pivot(index="fold", columns="option", values="rmse_mean")
    newest_better = bool((score["with the 13 weeks before it"] < score["without the 13 weeks before it"]).all())
    two_years_best = bool((score["only the two years before it"] == score.min(axis=1)).all())
    train_rows = "last two years" if two_years_best else "all" if newest_better else "to validation"
    print(f" with the newest 13 weeks better in both halves: {newest_better}; two years best in both halves: "
          f"{two_years_best} -> the final models train on rows: {train_rows}", flush=True)

    # S10: the average of five units against the average of ten. "Five" is every choice of five
    # of the ten units (252 of them); "ten" is one forecast, so it has no spread.
    print("\nStudy S10: average of five units against average of ten units", flush=True)
    units = [runs.unit(data, config, seed) for seed in range(2 * FINAL_SEEDS)]
    forecasts = np.stack([runs.unit_prediction(unit) for unit in units])

    def scored(forecast):
        return {**metrics(forecast, data.truth), **halves(forecast, data.truth)}
    fives = pd.DataFrame([scored(forecasts[list(pick)].mean(axis=0))
                          for pick in itertools.combinations(range(len(units)), FINAL_SEEDS)])
    ten, first_five = scored(forecasts.mean(axis=0)), scored(forecasts[:FINAL_SEEDS].mean(axis=0))
    members_p = int(runs.rows(units[0])["P"].sum())
    members_e = int(runs.rows(units[0])["epochs"].sum())
    averaging = pd.DataFrame([
        {"option": "average of five units", "forecasts_scored": len(fives), "P": 5 * members_p, "E": 5 * members_e,
         **{f"{m}_mean": fives[m].mean() for m in ("rmse", "mae", "smape")},
         **{f"{m}_std": fives[m].std(ddof=1) for m in ("rmse", "mae", "smape")},
         **{m: fives[m].mean() for m in ("rmse_half1", "rmse_half2", "mean_error")}},
        {"option": "average of ten units", "forecasts_scored": 1, "P": 10 * members_p, "E": 10 * members_e,
         **{f"{m}_mean": ten[m] for m in ("rmse", "mae", "smape")},
         **{f"{m}_std": np.nan for m in ("rmse", "mae", "smape")},
         **{m: ten[m] for m in ("rmse_half1", "rmse_half2", "mean_error")}}])
    averaging.to_csv(RESULTS / "ten_units_S10.csv", index=False)
    # The two forecasts that could be submitted are units 0 to 4 and all ten. The one with the
    # lower validation RMSE is used (asked for by the user). The 252 fives show how much a
    # five-unit average moves with the choice of units.
    n_units = 2 * FINAL_SEEDS if ten["rmse"] < first_five["rmse"] else FINAL_SEEDS
    print(averaging.round(2).to_string(index=False, line_width=250))
    print(f" units 0 to 4: RMSE {first_five['rmse']:.2f}; all ten: RMSE {ten['rmse']:.2f} "
          f"-> units to average: {n_units}", flush=True)
    return {"train_rows": train_rows, "units": n_units, "newest_better_in_both_halves": newest_better,
            "two_years_best_in_both_halves": two_years_best, "walk_forward_units": stored,
            "ten_units": units, "ten_units_rmse": ten["rmse"], "units_0_to_4_rmse": first_five["rmse"],
            "five_units_rmse_mean": float(fives["rmse"].mean()),
            "five_units_rmse_std": float(fives["rmse"].std(ddof=1))}


def checks(data, runs, config, final_keys):
    """Runs that choose nothing. They test whether dropout, which was never tuned, changes a result."""
    record = []
    print("\nCheck A2: Study A again without dropout", flush=True)
    keys = {start: [runs.run(data, dict(BASE, trend_start=start, dropout=0.0), seed)
                    for seed in range(FINAL_SEEDS)] for start in ("mean", "fade", "anchor")}
    print(summarise(runs, data, keys).round(2).to_string(index=False))
    record.append({"study": "A2", "setting": "trend_start", "keys": keys,
                   "question": "Starting trend, same settings as Study A but dropout 0"})
    print("\nCheck H: final configuration with and without dropout", flush=True)
    keys = {"0.0": [runs.run(data, dict(config, dropout=0.0), seed) for seed in range(FINAL_SEEDS)],
            str(config["dropout"]): final_keys}
    print(summarise(runs, data, keys).round(2).to_string(index=False))
    record.append({"study": "H", "setting": "dropout", "keys": keys,
                   "question": "Final configuration with dropout 0 and with dropout 0.1"})
    return record


@torch.no_grad()
def dropout_gap(data, runs, keys_by_option):
    """Error on training windows with dropout on (as during training) and off (as when forecasting)."""
    rows = []
    for option, keys in keys_by_option.items():
        values = []
        for key in keys:
            model, config = runs.model(data, key)
            target, _ = data.target(config)
            marks, n_external = data.marks(config)
            origins = torch.arange(config["seq_len"], data.val_start - PRED_LEN + 1, 97, device=DEVICE)
            inputs = window_inputs(config, target, marks, n_external, origins)
            truth = target[origins[:, None] + torch.arange(PRED_LEN, device=DEVICE)]
            torch.manual_seed(0)
            row = []
            for mode in ("train", "eval"):
                getattr(model, mode)()
                error = forecast_scaled(model, config, inputs) - truth
                row += [float(error.square().mean()), float(error.mean())]
            values.append(row)
        mean = np.mean(values, axis=0)
        rows.append({"option": option, "seeds": len(keys),
                     "squared_error_dropout_on": mean[0], "mean_error_dropout_on": mean[1],
                     "squared_error_dropout_off": mean[2], "mean_error_dropout_off": mean[3]})
    return pd.DataFrame(rows)


def analyse(data, runs, config, final_units, record, later=(), ablation=None):
    """Extra evidence from the stored final models; nothing is trained here."""
    final = pd.DataFrame([runs.unit_scores(data, unit) for unit in final_units])
    predictions = np.stack([runs.unit_prediction(unit) for unit in final_units])   # [seed, block, lead]

    # Study A: does a model behave the same with dropout on and off? (scaled units, training windows)
    starts = next(r["keys"] for r in record if r["setting"] == "trend_start")
    dropout_gap(data, runs, starts).to_csv(RESULTS / "dropout_gap.csv", index=False)

    # Which variable matters: scramble one variable inside the validation rows and re-score.
    if config["external"] != "none":
        order = np.random.default_rng(0).permutation(data.n - data.val_start)
        base, *extras = config["variables"].split("+")
        names = VARIABLES[base]                  # a variable is scrambled together with what is derived from it
        groups = [[c] + [f"{c}_{extra}" for extra in extras] for c in names if c in CONTINUOUS]
        if any(c in FLAGS for c in names):
            groups.append([c for c in names if c in FLAGS])
        rows = []
        for group in groups:
            for unit, score in zip(final_units, final["rmse"]):
                parts = []
                for key in unit:
                    model, member = runs.model(data, key)
                    parts.append(predict(data, member, model, data.val_origins, shuffle=(group, order)))
                shuffled = np.mean(parts, axis=0)
                label = "+".join(c[-1] for c in group) if group[0] in FLAGS else group[0][-1]
                rows.append({"variable": label,
                             "rmse_increase": metrics(shuffled, data.truth)["rmse"] - score})
        importance = pd.DataFrame(rows).groupby("variable", sort=False)["rmse_increase"].agg(["mean", "std"])
        importance.reset_index().to_csv(RESULTS / "importance.csv", index=False)

    # Error at each of the 168 lead times, for each way of using the optional file.
    lead = []
    for arm in ("none", "past", "past+future"):
        for seed, unit in enumerate((ablation or {}).get(arm, [])):
            error = runs.unit_prediction(unit) - data.truth
            lead.append(pd.DataFrame({"external": arm, "seed": seed, "lead": np.arange(1, PRED_LEN + 1),
                                      "rmse": np.sqrt((error ** 2).mean(axis=0)),
                                      "mean_error": error.mean(axis=0)}))
    if lead:
        pd.concat(lead).to_csv(RESULTS / "lead_time.csv", index=False)

    # How much one 168-step block says: RMSE of every validation block, for every final seed.
    blocks = np.stack([block_rmse(p, data.truth) for p in predictions])
    blocks_average = block_rmse(predictions.mean(axis=0), data.truth)          # the submitted forecast
    pd.DataFrame(blocks.T, columns=[f"seed_{s}" for s in range(len(final_units))]).assign(
        average=blocks_average, origin_time_idx=data.val_origins + 1).to_csv(RESULTS / "block_rmse.csv", index=False)

    # The highest values: the top 10% of true validation values and the error they carry.
    high = data.truth > np.quantile(data.truth, 0.9)

    def high_values(label, forecasts):           # forecasts: one or several [block, lead] arrays
        parts = []
        for forecast in forecasts:
            error = forecast - data.truth
            parts.append([100 * (error[high] ** 2).sum() / (error ** 2).sum(), error[high].mean(),
                          error[~high].mean(), np.sqrt((error[~high] ** 2).mean()),
                          np.sqrt((error ** 2).mean())])
        share, bias_high, bias_rest, rmse_rest, rmse = np.mean(parts, axis=0)
        return {"forecast": label, "seeds": len(forecasts), "share_of_squared_error_on_top_10pct": share,
                "mean_error_top_10pct": bias_high, "mean_error_rest": bias_rest,
                "rmse_rest": rmse_rest, "rmse": rmse}

    rows = [high_values("final model, single seeds", predictions),
            high_values("final model, average of seeds", [predictions.mean(axis=0)])]
    for item in list(record) + list(later):
        if item["study"] in {"B2", "I", "S2", "S3", "S4", "S7"}:
            for option, units in item["keys"].items():
                units = [[unit] if isinstance(unit, str) else unit for unit in units]
                label = f"{item['setting']} = {option}" if "setting" in item else option
                rows.append(high_values(f"{item['study']}: {label}",
                                        [runs.unit_prediction(unit) for unit in units]))
    pd.DataFrame(rows).assign(threshold=float(np.quantile(data.truth, 0.9))).to_csv(
        RESULTS / "high_values.csv", index=False)

    ensemble = metrics(predictions.mean(axis=0), data.truth)
    summary = {"final_config": identity(config), "P": int(final["P"].iloc[0]),
               "E": int(final["epochs"].iloc[0]), "seeds": list(range(len(final_units))),
               "single_model": {m: [float(final[m].mean()), float(final[m].std(ddof=1))]
                                for m in ("rmse", "mae", "smape")},
               "average_of_seeds": {**ensemble, **halves(predictions.mean(axis=0), data.truth)},
               "block_rmse_average": {"mean": float(blocks_average.mean()), "std": float(blocks_average.std()),
                                      "min": float(blocks_average.min()), "max": float(blocks_average.max()),
                                      "share_within_10_of_mean": float(
                                          (np.abs(blocks_average - blocks_average.mean()) <= 10).mean())}}
    (RESULTS / "final_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


# ----------------------------------------------------------------------------- submission
def submit():
    """Leaderboard forecast of the selected model: the average of SELECTED_UNITS units.

    Each model is trained once on time_idx 1..39288 for a fixed number of epochs: no early
    stopping and no refit. Models already stored by `study` are reused; missing ones are trained.
    Every model behind the forecast counts: P is the sum of their parameters, E of their epochs.
    """
    data, runs = Data(), Runs()
    seeds = list(range(SELECTED_UNITS))
    units = [runs.unit(data, SELECTED, seed) for seed in seeds]
    models, configs, per_unit = [], [], []
    for unit in units:
        parts = []
        for key in unit:
            model, member = runs.model(data, key)
            parts.append(predict(data, member, model, np.array([data.n]))[0])
            models.append(model)
            configs.append(member)
        per_unit.append(np.mean(parts, axis=0))
    per_unit = np.stack(per_unit)
    forecast = per_unit.mean(axis=0)
    validation = np.mean([runs.unit_prediction(unit) for unit in units], axis=0)

    total_p = int(sum(count_parameters(model) for model in models))
    total_e = int(sum(member["epochs"] for member in configs))
    assert (total_p, total_e) == (SELECTED_P, SELECTED_E), (total_p, total_e)
    assert forecast.shape == (PRED_LEN,) and np.isfinite(forecast).all() and (forecast >= 0).all()
    assert data.test_time_idx[0] == data.n + 1 and data.test_time_idx[-1] == data.n + PRED_LEN
    line = ", ".join(f"{value:.3f}" for value in forecast)
    assert len(line.split(",")) == PRED_LEN
    (RESULTS / "submission.txt").write_text(line + "\n")
    torch.save({"states": [model.state_dict() for model in models], "configs": configs,
                "config": SELECTED, "seeds": seeds, "trained_to_time_idx": int(data.val_start)},
               RESULTS / "final_model.pt")
    info = {"P_total_trainable_parameters": total_p, "E_training_epochs": total_e,
            "models": len(models), "units": len(units), "seeds": seeds, "config": SELECTED,
            "trained_on_time_idx": [1, int(data.val_start)],
            "forecast_time_idx": [int(data.test_time_idx[0]), int(data.test_time_idx[-1])],
            "validation_of_this_average": {**metrics(validation, data.truth), **halves(validation, data.truth)},
            "forecast_min_mean_max": [float(forecast.min()), float(forecast.mean()), float(forecast.max())]}
    (RESULTS / "submission_info.json").write_text(json.dumps(info, indent=1))
    pd.DataFrame({"time_idx": data.test_time_idx, "value": forecast}).to_csv(
        RESULTS / "submission_forecast.csv", index=False)
    print(json.dumps(info, indent=1))
    print(f"\n168 values written to {RESULTS / 'submission.txt'}")


# ----------------------------------------------------------------------------- checks
def check():
    torch.manual_seed(0)
    example = torch.tensor([1., 2., 3., 10., 5.]).reshape(1, 5, 1)
    remainder, trend = SeriesDecomposition(3)(example)
    assert torch.allclose(trend.flatten(), torch.tensor([4 / 3, 2., 5., 6., 20 / 3]))
    assert torch.allclose(remainder + trend, example)
    values = torch.tensor([10., 20., 30., 40.]).reshape(1, 1, 1, 4)
    mixed = aggregate_delays(values, torch.tensor([[1, 3]]), torch.tensor([[.75, .25]]))
    assert torch.allclose(mixed.flatten(), torch.tensor([35., 15., 25., 25.]))
    q, k = torch.randn(2, 1, 3, 16, dtype=torch.float64), torch.randn(2, 1, 3, 16, dtype=torch.float64)
    qc, kc = q - q.mean(-1, keepdim=True), k - k.mean(-1, keepdim=True)
    direct = torch.stack([(qc * torch.roll(kc, tau, -1)).sum(-1).mean((1, 2)) for tau in range(16)], -1)
    assert torch.allclose(delay_scores(q, k), direct)
    print("Task 1 worked examples and delay scores: ok")

    for start in ("mean", "fade", "anchor"):
        for seq_len, marks in ((96, 0), (168, 12), (336, 6)):
            label_len = seq_len // 2
            model = Autoformer(marks, seq_len, label_len, PRED_LEN, d=16, trend_start=start)
            out = model(torch.randn(3, seq_len, 1), torch.randn(3, seq_len, marks),
                        torch.randn(3, label_len + PRED_LEN, marks))
            assert out.shape == (3, PRED_LEN)
            out.square().mean().backward()
            starved = [n for n, p in model.named_parameters() if p.grad is None]
            assert not starved, f"no gradient for {starved}"
    for e_layers, marks_net, hidden, layers in ((2, "linear", 1, 2), (1, "mlp", 1, 2), (2, "mlp", 2, 3),
                                                (1, "mlp", 4, 3)):
        model = Autoformer(12, 96, 48, PRED_LEN, d=16, trend_start="anchor", e_layers=e_layers,
                           marks_net=marks_net, marks_hidden=hidden, marks_layers=layers)
        out = model(torch.randn(3, 96, 1), torch.randn(3, 96, 12), torch.randn(3, 48 + PRED_LEN, 12))
        assert out.shape == (3, PRED_LEN) and len(model.encoder) == e_layers
        out.square().mean().backward()
        assert all(p.grad is not None for p in model.parameters())
    print("Model output shape [batch, 168] and gradients to every parameter: ok")

    data = Data()
    assert (data.n, data.total, data.val_start) == (43656, 43824, 39288)
    assert len(data.val_origins) == 176 and data.val_origins[-1] + PRED_LEN == data.n
    config = dict(BASE)
    target, restore = data.target(config)
    assert torch.isnan(target[data.n:]).all() and torch.isfinite(target[:data.n]).all()
    assert np.allclose(restore(target[:data.n].cpu().numpy().astype(float)), data.y, atol=1e-2)
    assert abs(float(target[:data.val_start].mean())) < 1e-4, "target scaling must use training rows only"
    marks, n_external = data.marks(config)
    assert marks.shape == (data.total, 12) and n_external == 10
    continuous = marks[:data.val_start, :6]
    assert continuous.mean(0).abs().max() < 1e-3 and (continuous.std(0) - 1).abs().max() < 1e-3
    assert bool((marks[:, 6:10].sum(1) == 1).all()), "the four flags must sum to one"

    origin = torch.tensor([data.n], device=DEVICE)       # the leaderboard forecast
    x, mark_enc, mark_dec = window_inputs(config, target, marks, n_external, origin)
    assert torch.isfinite(x).all() and x.shape == (1, 168, 1) and mark_dec.shape == (1, 84 + 168, 12)
    assert torch.equal(mark_dec[0, 84:], marks[data.n:data.total]), "decoder must see rows 43657..43824"
    past = dict(config, external="past")
    _, _, masked = window_inputs(past, target, marks, n_external, origin)
    assert bool((masked[0, 84:, :10] == 0).all()) and torch.equal(masked[0, :84], mark_dec[0, :84])
    assert torch.equal(masked[0, 84:, 10:], mark_dec[0, 84:, 10:])
    none = data.marks(dict(config, external="none"))
    assert none[0].shape == (data.total, 2) and none[1] == 0
    last_training_origin = data.val_start - PRED_LEN
    assert last_training_origin + PRED_LEN <= data.val_start < data.val_origins[0] + 1
    print("Data sizes, split, scaling on training rows only, and no look-ahead: ok")

    # Later settings: defaults keep the keys of earlier runs; "raw+log" stands for two models.
    assert Runs.key(dict(BASE), 0) == Runs.key(dict(BASE, **LATER), 0)
    assert Runs.key(dict(BASE), 0) != Runs.key(dict(BASE, e_layers=2), 0)
    assert Runs.key(dict(BASE), 0) != Runs.key(dict(BASE, marks_net="mlp"), 0)
    assert Runs.key(dict(BASE, marks_net="mlp"), 0) != Runs.key(dict(BASE, marks_net="mlp", marks_hidden=2), 0)
    assert Runs.key(dict(BASE, marks_net="mlp"), 0) != Runs.key(dict(BASE, marks_net="mlp", marks_layers=3), 0)
    assert [m["target"] for m in members(dict(BASE, target="raw+log"))] == ["raw", "log"]
    assert members(config) == [config]
    # Means and changes over the last 6 and 24 steps: same base columns, scaled on training
    # rows only, and no value before the forecast start depends on the 168 horizon rows.
    rolled, n_rolled = data.marks(dict(config, variables=ROLLING))
    assert rolled.shape == (data.total, 10 + 24 + 2) and n_rolled == 34
    assert torch.equal(rolled[:, :10], marks[:, :10]) and torch.isfinite(rolled).all()
    derived = rolled[:data.val_start, 10:34]
    assert derived.mean(0).abs().max() < 1e-3 and (derived.std(0) - 1).abs().max() < 1e-3
    old24, _ = data.marks(dict(config, variables="all+roll+diff"))      # Study I's features
    assert torch.equal(old24[:, 10:16], rolled[:, 16:22]) and torch.equal(old24[:, 16:22], rolled[:, 28:34])
    other = Data()
    other.external = data.external.astype(float)
    other.external.iloc[data.n:, :6] += 5.0                             # change the horizon rows only
    assert torch.equal(other.marks(dict(config, variables=ROLLING))[0][:data.n], rolled[:data.n])
    for _, change in STAGE_3[0][3]:                                      # the feature sets of S7
        wide, n_wide = data.marks(dict(config, **change))
        assert torch.isfinite(wide).all() and torch.equal(wide[:, :34], rolled[:, :34])
        assert wide[:data.val_start, 10:n_wide].mean(0).abs().max() < 1e-3
        assert (wide[:data.val_start, 10:n_wide].std(0) - 1).abs().max() < 1e-3
        assert torch.equal(other.marks(dict(config, **change))[0][:data.n], wide[:data.n])
    assert data.marks(dict(config, variables=ROLLING + "+std24+min24+max24"))[1] == 34 + 18
    assert Runs.key(config, 0, data.n) != Runs.key(config, 0, data.n, data.n - TWO_YEARS) != Runs.key(config, 0)
    print("Later settings keep earlier keys; rolling features are scaled and backward-looking: ok")

    # The selected model: its size, and that the stored studies and files agree with it.
    selected = [build(member, data.marks(member)[0].shape[1]) for member in members(SELECTED)]
    assert [count_parameters(model) for model in selected] == [8933, 8933]
    assert SELECTED_UNITS * sum(count_parameters(model) for model in selected) == SELECTED_P
    assert SELECTED_UNITS * len(selected) * SELECTED["epochs"] == SELECTED_E
    assert all(type(module).__name__ != "MultiheadAttention" for model in selected for module in model.modules())
    names = [type(module).__name__ for module in selected[0].modules()]
    assert names.count("AutoCorrelation") == 3 and names.count("SeriesDecomposition") == 6
    if (RESULTS / "studies.json").exists():
        saved = json.loads((RESULTS / "studies.json").read_text())
        assert saved["final_config"] == identity(SELECTED), "the studies ended somewhere else"
        assert saved["deploy"]["units"] == SELECTED_UNITS and saved["deploy"]["train_rows"] == "to validation"
    if (RESULTS / "submission_info.json").exists():
        info = json.loads((RESULTS / "submission_info.json").read_text())
        assert (info["P_total_trainable_parameters"], info["E_training_epochs"]) == (SELECTED_P, SELECTED_E)
        assert info["config"] == SELECTED
        line = (RESULTS / "submission.txt").read_text().strip().split(",")
        assert len(line) == PRED_LEN and all(float(value) >= 0 for value in line)
    print("Selected model: 10 models, P = 89,330, E = 40; studies and submission files agree: ok")

    reference = baselines(data).set_index("baseline")["rmse"]
    assert abs(reference["training mean"] - 88.7) < 0.1
    assert abs(reference["same-time linear fit on optional file (raw target)"] - 72.1) < 0.1
    truth = np.array([[0., 10., 20.]])
    assert metrics(truth, truth) == {"rmse": 0.0, "mae": 0.0, "smape": 0.0}
    assert abs(metrics(np.array([[0., 30., 20.]]), truth)["smape"] - 100 * (2 * 20 / 40) / 3) < 1e-9
    print("Baselines match the probe and the three metrics behave: ok")
    print("All checks passed.")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "report":
        from report import report                 # plotting lives in its own file
        report()
    elif command in {"check", "study", "submit"}:
        {"check": check, "study": study, "submit": submit}[command]()
    else:
        raise SystemExit(__doc__)
