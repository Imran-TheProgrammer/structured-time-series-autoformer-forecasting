"""Tables (CSV + TeX) and numbered PDF figures for the Task 2 report, from the stored runs.

Called by `python task2.py report`. Nothing is trained here.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from task2 import PRED_LEN, RESULTS, Data, Runs, block_rmse, summarise

# Categorical slots 1-3 of the chart palette, checked for colour-blind separation on white.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, SECONDARY, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
ARM_COLOUR = {"past+future": BLUE, "none": ORANGE, "past": AQUA}       # colour follows the arm
ARM_LABEL = {"past+future": "optional file: past + future", "none": "no optional file",
             "past": "optional file: past only"}

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.titlelocation": "left", "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8, "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": SECONDARY, "xtick.color": AXIS, "ytick.color": AXIS,
    "xtick.labelcolor": SECONDARY, "ytick.labelcolor": SECONDARY, "text.color": INK,
    "legend.frameon": False, "lines.linewidth": 1.6, "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round", "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def save(figure, name):
    figure.savefig(RESULTS / f"{name}.pdf")
    plt.close(figure)
    print("figure:", f"{name}.pdf")


def write_table(table, name, tex=None):
    table.to_csv(RESULTS / f"{name}.csv", index=False)
    (tex if tex is not None else table).to_latex(RESULTS / f"{name}.tex", index=False, escape=False)
    print("table: ", f"{name}.csv / .tex")


def plus_minus(mean, std, digits=1):
    return [f"{m:.{digits}f} $\\pm$ {s:.{digits}f}" for m, s in zip(mean, std)]


# ----------------------------------------------------------------------------- tables
def tables(data, runs, saved):
    baselines = pd.read_csv(RESULTS / "baselines.csv").round(1)
    write_table(baselines, "q2_table_baselines",
                baselines.rename(columns={"rmse": "RMSE", "mae": "MAE", "smape": "sMAPE"}))

    for item in saved["studies"]:
        summary = summarise(runs, data, item["keys"])
        summary["chosen"] = summary["option"] == item["chosen"]
        tex = pd.DataFrame({
            "option": summary["option"], "seeds": summary["seeds"], "P": summary["P"],
            "epochs": summary["epochs"],
            "RMSE": plus_minus(summary["rmse_mean"], summary["rmse_std"]),
            "MAE": plus_minus(summary["mae_mean"], summary["mae_std"]),
            "sMAPE": plus_minus(summary["smape_mean"], summary["smape_std"]),
            "chosen": np.where(summary["chosen"], "yes", "")})
        write_table(summary.round(5), f"q2_table_{item['study']}_{item['setting']}", tex)

    for item in saved.get("checks", []):                  # runs that test something but choose nothing
        summary = summarise(runs, data, item["keys"])
        tex = pd.DataFrame({
            "option": summary["option"], "seeds": summary["seeds"],
            "RMSE": plus_minus(summary["rmse_mean"], summary["rmse_std"]),
            "MAE": plus_minus(summary["mae_mean"], summary["mae_std"]),
            "sMAPE": plus_minus(summary["smape_mean"], summary["smape_std"])})
        write_table(summary.round(5), f"q2_table_{item['study']}_{item['setting']}_check", tex)
    for item in saved.get("later", []):                   # studies S1 to S3: the lowest mean wins
        summary = summarise(runs, data, item["keys"])
        summary["chosen"] = summary["option"] == item["chosen"]
        tex = pd.DataFrame({
            "option": summary["option"], "seeds": summary["seeds"], "P": summary["P"],
            "epochs": summary["epochs"],
            "RMSE": plus_minus(summary["rmse_mean"], summary["rmse_std"], 2),
            "MAE": plus_minus(summary["mae_mean"], summary["mae_std"], 2),
            "sMAPE": plus_minus(summary["smape_mean"], summary["smape_std"], 2),
            "chosen": np.where(summary["chosen"], "yes", "")})
        write_table(summary.round(5), f"q2_table_{item['study']}", tex)
    if saved.get("ablation"):                             # the optional file on the selected model
        summary = summarise(runs, data, saved["ablation"]["keys"])
        tex = pd.DataFrame({
            "optional file": summary["option"], "seeds": summary["seeds"], "P": summary["P"],
            "RMSE": plus_minus(summary["rmse_mean"], summary["rmse_std"], 2),
            "MAE": plus_minus(summary["mae_mean"], summary["mae_std"], 2),
            "sMAPE": plus_minus(summary["smape_mean"], summary["smape_std"], 2)})
        write_table(summary.round(5), "q2_table_ablation", tex)
    for name, table in (("dropout_gap", "q2_table_A_dropout_gap"),
                        ("high_values", "q2_table_high_values"), ("acceptance", "q2_table_final_check"),
                        ("walk_forward_S9", "q2_table_S9"), ("ten_units_S10", "q2_table_S10")):
        if (RESULTS / f"{name}.csv").exists():
            write_table(pd.read_csv(RESULTS / f"{name}.csv").round(3), table)

    if (RESULTS / "importance.csv").exists():
        importance = pd.read_csv(RESULTS / "importance.csv")
        tex = pd.DataFrame({"variable shuffled": importance["variable"],
                            "RMSE increase": plus_minus(importance["mean"], importance["std"])})
        write_table(importance.round(3), "q2_table_importance", tex)

    final = json.loads((RESULTS / "final_summary.json").read_text())
    single, average, blocks = final["single_model"], final["average_of_seeds"], final["block_rmse_average"]
    rows = [{"quantity": "single model, mean over seeds", "RMSE": single["rmse"][0],
             "MAE": single["mae"][0], "sMAPE": single["smape"][0]},
            {"quantity": "single model, spread over seeds", "RMSE": single["rmse"][1],
             "MAE": single["mae"][1], "sMAPE": single["smape"][1]},
            {"quantity": "average of the seeds' forecasts", "RMSE": average["rmse"],
             "MAE": average["mae"], "sMAPE": average["smape"]}]
    write_table(pd.DataFrame(rows).round(2), "q2_table_final")
    spread = pd.DataFrame([{"blocks": 176, "mean": blocks["mean"], "std": blocks["std"],
                            "min": blocks["min"], "max": blocks["max"]}]).round(1)
    write_table(spread, "q2_table_block_spread")


# ----------------------------------------------------------------------------- figures
def autocorrelation(values, lags):
    values = values - values.mean()
    scale = np.dot(values, values)
    return np.array([np.dot(values[:-lag], values[lag:]) / scale for lag in lags])


def figure_periodicity(data):
    """Short memory, and a daily cycle that shows once the slow part is removed."""
    history = data.y[:data.val_start]                      # training rows only
    kernel = 25
    padded = np.pad(history, kernel // 2, mode="edge")
    remainder = history - np.convolve(padded, np.ones(kernel) / kernel, mode="valid")
    lags = np.arange(1, 169)
    raw, rest = autocorrelation(history, lags), autocorrelation(remainder, lags)
    profile = pd.Series(history).groupby(np.arange(len(history)) % 24).mean().to_numpy()
    pd.DataFrame({"lag": lags, "target": raw, "remainder": rest}).to_csv(
        RESULTS / "q2_fig1_periodicity.csv", index=False)

    figure, axes = plt.subplots(1, 3, figsize=(7.4, 2.5), layout="constrained")
    for axis, series, title in ((axes[0], raw, "Autocorrelation of the target"),
                                (axes[1], rest, "After removing a 25-step average")):
        axis.axhline(0, color=AXIS, linewidth=0.8)
        axis.plot(lags, series, color=BLUE)
        axis.set(title=title, xlabel="lag (steps)", xticks=np.arange(0, 169, 24), xlim=(0, 168))
    axes[0].set_ylabel("autocorrelation")
    axes[2].plot(np.arange(24), profile, color=BLUE, marker="o", markersize=4.5,
                 markeredgecolor="white", markeredgewidth=1)
    axes[2].set(title="Mean by position in a 24-step cycle", xlabel="time_idx mod 24",
                ylabel="mean target", xticks=np.arange(0, 24, 6))
    save(figure, "q2_fig1_periodicity")


def figure_lead_time(data):
    lead = pd.read_csv(RESULTS / "lead_time.csv")
    reference = pd.read_csv(RESULTS / "baselines.csv").set_index("baseline")["rmse"]["training mean"]
    figure, axis = plt.subplots(figsize=(7.4, 3.0), layout="constrained")
    axis.axhline(reference, color=MUTED, linewidth=0.9, label="always predicting the training mean")
    for arm in ("none", "past", "past+future"):
        table = lead[lead["external"] == arm].pivot(index="lead", columns="seed", values="rmse")
        if table.empty:
            continue
        mean = table.mean(axis=1)
        axis.fill_between(table.index, table.min(axis=1), table.max(axis=1),
                          color=ARM_COLOUR[arm], alpha=0.12, linewidth=0)
        axis.plot(table.index, mean, color=ARM_COLOUR[arm], label=ARM_LABEL[arm],
                  linestyle=(0, (4, 2)) if arm == "past" else "-")
        if arm == "past+future":                           # label only the line that matters
            axis.plot(PRED_LEN, mean.iloc[-1], "o", color=BLUE, markersize=5,
                      markeredgecolor="white", markeredgewidth=1, clip_on=False)
            axis.annotate(f"{mean.iloc[-1]:.0f}", (PRED_LEN, mean.iloc[-1]), xytext=(6, -3),
                          textcoords="offset points", color=INK, annotation_clip=False)
    axis.set(title="Validation RMSE at each lead time (line: mean over seeds, band: range over seeds)",
             xlabel="lead time (steps ahead)", ylabel="RMSE", xticks=np.arange(0, 169, 24),
             xlim=(1, PRED_LEN), ylim=(0, None))
    axis.legend(loc="lower right", ncol=2, fontsize=7.5)
    save(figure, "q2_fig2_lead_time")


def figure_examples(data, runs, saved):
    """Three validation blocks: an easy one, a typical one and a hard one for the final model."""
    with_file = np.mean([runs.unit_prediction(unit) for unit in saved["final_units"]], axis=0)
    arms = (saved.get("ablation") or {}).get("keys", {})
    without = (np.mean([runs.unit_prediction(unit) for unit in arms["none"]], axis=0)
               if "none" in arms else None)
    errors = block_rmse(with_file, data.truth)
    order = np.argsort(errors)
    picks = [(order[int(q * (len(order) - 1))], word) for q, word in
             ((0.10, "an easy block"), (0.50, "a typical block"), (0.90, "a hard block"))]
    figure, axes = plt.subplots(3, 1, figsize=(7.4, 6.2), layout="constrained", sharex=True)
    for axis, (block, word) in zip(axes, picks):
        origin = data.val_origins[block]
        axis.axvline(0, color=AXIS, linewidth=0.8)
        axis.plot(np.arange(-PRED_LEN, 0), data.y[origin - PRED_LEN:origin], color=MUTED,
                  linewidth=1.1, label="history")
        axis.plot(np.arange(PRED_LEN), data.truth[block], color=INK, linewidth=1.2, label="what happened")
        if without is not None:
            axis.plot(np.arange(PRED_LEN), without[block], color=ORANGE, label=ARM_LABEL["none"])
        axis.plot(np.arange(PRED_LEN), with_file[block], color=BLUE, label="submitted model")
        axis.set(title=f"Forecast from time_idx {origin + 1}: block RMSE {errors[block]:.0f} ({word})",
                 ylabel="target", xticks=np.arange(-168, 169, 24), xlim=(-PRED_LEN, PRED_LEN))
    axes[-1].set_xlabel("steps from the forecast start")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=4, fontsize=7.5)
    save(figure, "q2_fig3_examples")


def figure_block_rmse():
    blocks = pd.read_csv(RESULTS / "block_rmse.csv")["average"].to_numpy()
    edges = np.arange(0, np.ceil(blocks.max() / 10) * 10 + 10, 10)
    figure, axis = plt.subplots(figsize=(7.4, 2.8), layout="constrained")
    axis.hist(blocks, bins=edges, color=BLUE, edgecolor="white", linewidth=1.2)
    axis.axvline(blocks.mean(), color=INK, linewidth=1.0)
    axis.annotate(f"mean {blocks.mean():.0f}", (blocks.mean(), axis.get_ylim()[1]), xytext=(4, -9),
                  textcoords="offset points", color=INK)
    axis.set(title=f"RMSE of each of the {len(blocks)} validation blocks of 168 steps (submitted model)",
             xlabel="RMSE of one block", ylabel="number of blocks", xlim=(0, edges[-1]))
    axis.grid(axis="x", visible=False)
    save(figure, "q2_fig4_block_rmse")


def figure_submission(data):
    path = RESULTS / "submission_forecast.csv"
    if not path.exists():
        print("figure: q2_fig5_submission.pdf skipped (run `python task2.py submit` first)")
        return
    forecast = pd.read_csv(path)
    past = np.arange(data.n - 2 * PRED_LEN, data.n)
    figure, axis = plt.subplots(figsize=(7.4, 2.8), layout="constrained")
    axis.axvline(data.n + 0.5, color=AXIS, linewidth=0.8)
    axis.plot(past + 1, data.y[past], color=INK, linewidth=1.2, label="observed history")
    axis.plot(forecast["time_idx"], forecast["value"], color=BLUE, label="submitted forecast")
    axis.set(title="Forecast for time_idx 43657 to 43824", xlabel="time_idx", ylabel="target",
             ylim=(0, None))
    figure.legend(loc="outside upper right", ncol=2, fontsize=7.5)
    save(figure, "q2_fig5_submission")


def figure_progress():
    """Validation RMSE of the reference points and of the submitted average after each stage."""
    reference = pd.read_csv(RESULTS / "baselines.csv").set_index("baseline")["rmse"]
    checks = pd.read_csv(RESULTS / "acceptance.csv")
    kept = checks[checks["configuration"] == "new"].set_index("stage")["average_rmse"]
    first = checks[checks["configuration"] == "current"]["average_rmse"].iloc[0]
    bars = [("Always the training mean", reference["training mean"]),
            ("Linear fit on the optional file", reference["same-time linear fit on optional file (raw target)"]),
            ("Autoformer after the first studies (A to I)", first),
            ("+ two-layer feature network, raw and log targets (S1 to S3)", kept["S1-S3"]),
            ("+ means and changes over 6 and 24 steps (S4 to S6): submitted", kept["S4-S6"])]
    pd.DataFrame(bars, columns=["forecast", "validation_rmse"]).to_csv(RESULTS / "q2_fig6_progress.csv", index=False)
    figure, axis = plt.subplots(figsize=(7.4, 2.4), layout="constrained")
    positions = np.arange(len(bars))[::-1]
    axis.barh(positions, [value for _, value in bars], height=0.5, color=[MUTED, MUTED, BLUE, BLUE, BLUE])
    for position, (_, value) in zip(positions, bars):
        axis.annotate(f"{value:.1f}", (value, position), xytext=(4, 0), textcoords="offset points",
                      va="center", color=INK)
    axis.set(title="Validation RMSE: reference points (grey) and the five-seed average after each stage (blue)",
             xlabel="validation RMSE", yticks=positions, yticklabels=[label for label, _ in bars],
             xlim=(0, 100))
    axis.tick_params(axis="y", labelcolor=INK)
    axis.grid(axis="y", visible=False)
    save(figure, "q2_fig6_progress")


def report():
    data, runs = Data(), Runs()
    saved = json.loads((RESULTS / "studies.json").read_text())
    tables(data, runs, saved)
    figure_periodicity(data)
    if (RESULTS / "lead_time.csv").exists():
        figure_lead_time(data)
    figure_examples(data, runs, saved)
    figure_block_rmse()
    figure_submission(data)
    figure_progress()
