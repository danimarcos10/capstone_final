"""
Generate thesis-ready methodology visuals.

Outputs:
  1. Pipeline diagram         -> reports/figures/methodology_pipeline.png
  2. CONSORT sample flow      -> reports/figures/methodology_sample_flow.png
  3. Feature-set table        -> reports/tables/methodology_feature_sets.csv
                                 reports/figures/methodology_feature_sets.png
  4. Model performance table  -> reports/figures/model_performance_summary.png
                                 reports/tables/model_performance_summary.tex

Usage:  python tools/make_methodology_figures.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
FIGURES_DIR = REPO_ROOT / "reports" / "figures"
TABLES_DIR = REPO_ROOT / "reports" / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# === 1. Methodology pipeline diagram ===

PIPELINE_STEPS = [
    ("raw",      "Raw Pew ATP W119\nSPSS (.sav)"),
    ("audit",    "Load + Audit\n(30/30 validated)"),
    ("vars",     "Variable Construction\nTarget + 4 Feature Sets"),
    ("preproc",  "Preprocessing\nEncoding + Skip Patterns"),
    ("missing",  "Missingness:\nImpute+Indicators;\nNot sure = Own category"),
    ("split",    "Train/Test Split\n(80/20, stratified, seed=42)"),
    ("models",   "Models\nWeighted Logistic Reg.\n+ HistGradientBoosting"),
    ("eval",     "Evaluation: AUC, PR-AUC,\nBrier, ECE; Thresholds"),
    ("subgroup", "Subgroups:\nTPR/FPR + Calibration"),
    ("robust",   "Robustness: Recoding,\nNot sure, Weights,\nSeeds, Skip-patterns"),
    ("latent",   "Latent Attitude\nOrdinal Factor Analysis\n+ Matched Comparison"),
    ("outputs",  "Outputs: Reports, Tables,\nFigures, Model Cards,\nMetadata"),
]


def build_pipeline_graph():
    """Return a graphviz.Digraph for the methodology pipeline."""
    import graphviz

    g = graphviz.Digraph("methodology_pipeline", format="png", engine="dot")

    g.attr(
        rankdir="LR",
        bgcolor="white",
        fontname="Helvetica",
        dpi="300",
        size="18,4!",
        ratio="compress",
        nodesep="0.3",
        ranksep="0.45",
        margin="0.2",
    )
    g.attr(
        "node",
        shape="box",
        style="rounded,filled",
        fillcolor="#EAF2FA",
        color="#4A7FB5",
        fontname="Helvetica",
        fontsize="10",
        penwidth="1.5",
        margin="0.15,0.10",
        width="0",
        height="0",
    )
    g.attr("edge", color="#4A7FB5", arrowsize="0.7", penwidth="1.2")

    first_id = PIPELINE_STEPS[0][0]
    last_id = PIPELINE_STEPS[-1][0]

    for node_id, label in PIPELINE_STEPS:
        extra = {}
        if node_id == first_id:
            extra = dict(fillcolor="#D6E9F8", color="#2B5D8A", fontcolor="#1A3A5C")
        elif node_id == last_id:
            extra = dict(fillcolor="#D5EDDB", color="#3A7D44", fontcolor="#1F4D27")
        g.node(node_id, label=label, **extra)

    for i in range(len(PIPELINE_STEPS) - 1):
        g.edge(PIPELINE_STEPS[i][0], PIPELINE_STEPS[i + 1][0])

    return g


# === 2. CONSORT-style sample flow diagram ===

def build_sample_flow_graph():
    """Return a graphviz.Digraph for the CONSORT sample flow."""
    import graphviz

    g = graphviz.Digraph("methodology_sample_flow", format="png", engine="dot")

    g.attr(
        rankdir="TB",
        bgcolor="white",
        fontname="Helvetica",
        dpi="300",
        size="6,10!",
        margin="0.3",
        nodesep="0.35",
        ranksep="0.50",
    )
    g.attr(
        "node",
        shape="box",
        style="rounded",
        fontname="Helvetica",
        fontsize="11",
        penwidth="1.2",
        color="black",
        fillcolor="white",
        width="2.8",
        height="0",
        margin="0.18,0.12",
    )
    g.attr("edge", color="black", arrowsize="0.8", penwidth="1.0")

    # Main flow: N counts from pipeline
    g.node("loaded",   "Loaded sample\nN = 11,004")
    g.node("valid",    "Valid target\nN = 10,771")
    g.node("latent",   "Eligible (latent analysis)\nN = 8,714")
    g.node("train",    "Training set\nN = 6,971  (80%)")
    g.node("test",     "Test set\nN = 1,743  (20%)")

    # Exclusion side-boxes
    g.node(
        "exc_target",
        "Excluded:\nmissing target\nn = 233",
        shape="box", style="rounded,dashed",
        fontsize="10", color="grey40", fontcolor="grey30",
        width="2.0",
    )
    g.node(
        "exc_latent",
        "Excluded:\nmissing latent items\nn = 2,057",
        shape="box", style="rounded,dashed",
        fontsize="10", color="grey40", fontcolor="grey30",
        width="2.0",
    )

    g.edge("loaded", "valid")
    g.edge("valid",  "latent")
    g.edge("valid",  "train")
    g.edge("valid",  "test")

    g.edge("loaded", "exc_target", style="dashed", color="grey50")
    g.edge("valid",  "exc_latent", style="dashed", color="grey50")

    # Align exclusion boxes to the right of their flow counterparts
    with g.subgraph() as s:
        s.attr(rank="same")
        s.node("loaded")
        s.node("exc_target")

    with g.subgraph() as s:
        s.attr(rank="same")
        s.node("latent")
        s.node("exc_latent")

    with g.subgraph() as s:
        s.attr(rank="same")
        s.node("train")
        s.node("test")

    return g


# === 3. Feature-set summary table ===

FEATURE_TABLE = [
    {
        "Feature set": "Core Attitudes",
        "# Raw": "9",
        "# Encoded": "18",
        "What it measures": "Views on AI regulation, risk, benefit",
        "Examples": "AIMOSTBENEFIT_W119, AISCALE_a-e_W119",
    },
    {
        "Feature set": "Knowledge & AI Orientation",
        "# Raw": "4",
        "# Encoded": "7",
        "What it measures": "Self-assessed AI familiarity, usage",
        "Examples": "AIAWARE_W119, CHATGPT_W119, AIUSE_W119",
    },
    {
        "Feature set": "Demographics",
        "# Raw": "6",
        "# Encoded": "18",
        "What it measures": "Socio-demographics & political affiliation",
        "Examples": "F_AGECAT, F_EDUCCAT2, F_PARTY_FINAL",
    },
    {
        "Feature set": "Employment Context",
        "# Raw": "3",
        "# Encoded": "varies",
        "What it measures": "Work status, exposure to AI at work",
        "Examples": "F_EMPLOY, AIJOBUSE_W119",
    },
    {
        "Feature set": "Full",
        "# Raw": "22",
        "# Encoded": "58",
        "What it measures": "All feature sets combined",
        "Examples": "(union of above)",
    },
]

FEATURE_TABLE_FOOTNOTE = (
    "Encoded features reflect one-hot expansion + missing indicators "
    "(impute+indicator regime)."
)


def write_feature_csv(path: Path) -> None:
    """Write feature-set table to CSV."""
    cols = list(FEATURE_TABLE[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(FEATURE_TABLE)


def render_feature_table_png(path: Path) -> None:
    """Render feature-set table as a PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = list(FEATURE_TABLE[0].keys())
    cell_text = [[row[c] for c in cols] for row in FEATURE_TABLE]

    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_text,
        colLabels=cols,
        cellLoc="left",
        colLoc="left",
        loc="upper center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.55)

    for col_idx in range(len(cols)):
        cell = tbl[0, col_idx]
        cell.set_facecolor("#3C3C3C")
        cell.set_text_props(color="white", fontweight="bold", fontsize=9.5)
        cell.set_edgecolor("white")

    # Alternating row shading
    for row_idx in range(1, len(cell_text) + 1):
        for col_idx in range(len(cols)):
            cell = tbl[row_idx, col_idx]
            cell.set_edgecolor("#CCCCCC")
            if row_idx % 2 == 0:
                cell.set_facecolor("#F5F5F5")
            else:
                cell.set_facecolor("white")

    # Narrower for counts, wider for text columns
    col_widths = [0.15, 0.06, 0.08, 0.32, 0.30]
    for col_idx, w in enumerate(col_widths):
        for row_idx in range(len(cell_text) + 1):
            tbl[row_idx, col_idx].set_width(w)

    fig.text(
        0.05, 0.02,
        f"Note: {FEATURE_TABLE_FOOTNOTE}",
        fontsize=7.5, fontstyle="italic", color="grey",
        ha="left", va="bottom",
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(str(path), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# === 4. Model performance summary table ===

MODEL_PERF_CAPTION = (
    "Weighted test-set performance comparison across model classes "
    "and feature specifications."
)

# Falls back to generic serif if Times New Roman unavailable
_SERIF_FONT = "Times New Roman"


def _load_model_comparison(tables_dir: Path) -> list[dict]:
    """Read both CSVs and return a unified list of row dicts.

    Columns (all strings, pre-rounded):
        Model, Feature Set, ROC-AUC, PR-AUC, Brier Score, ECE,
        Balanced Accuracy, Test N
    """
    import csv as _csv

    rows: list[dict] = []

    # --- main model comparison ---
    main_csv = tables_dir / "model_comparison.csv"
    with open(main_csv, newline="", encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            rows.append({
                "Model": r["model"],
                "Feature Set": "Full (22 raw / 58 enc.)",
                "ROC-AUC": f'{float(r["roc_auc_weighted"]):.3f}',
                "PR-AUC": f'{float(r["pr_auc_weighted"]):.3f}',
                "Brier Score": f'{float(r["brier_weighted"]):.3f}',
                "ECE": f'{float(r["ece_weighted"]):.3f}',
                "Balanced Acc.": f'{float(r["balanced_acc_weighted"]):.3f}',
                "Test N": str(int(r["n_test"])),
            })

    # --- phase 7.1 matched comparison ---
    matched_csv = tables_dir / "phase7_1_model_comparison_matched.csv"
    pick = {
        "GBM: Matched Raw (9 items)": "Core Attitudes (9 raw)",
        "LR: Latent + Know + Demo + NotSure": "Latent + Know. + Demo.",
    }
    with open(matched_csv, newline="", encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            name = r["model_name"]
            if name not in pick:
                continue
            display = "Best Matched Raw (GBM)" if "GBM" in name else "Best Latent (LR)"
            rows.append({
                "Model": display,
                "Feature Set": pick[name],
                "ROC-AUC": f'{float(r["roc_auc_weighted"]):.3f}',
                "PR-AUC": f'{float(r["pr_auc_weighted"]):.3f}',
                "Brier Score": f'{float(r["brier_weighted"]):.3f}',
                "ECE": f'{float(r["ece_weighted"]):.3f}',
                "Balanced Acc.": f'{float(r["balanced_acc_weighted"]):.3f}',
                "Test N": "1743",
            })

    return rows


def render_model_performance_png(rows: list[dict], path: Path) -> None:
    """Render a black-and-white academic table as a 300-DPI PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    available = {f.name for f in fm.fontManager.ttflist}
    serif = _SERIF_FONT if _SERIF_FONT in available else "serif"

    cols = list(rows[0].keys())
    cell_text = [[r[c] for c in cols] for r in rows]

    fig, ax = plt.subplots(figsize=(13.5, 3.0))
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_text,
        colLabels=cols,
        cellLoc="center",
        colLoc="center",
        loc="upper center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.6)

    n_rows = len(cell_text)
    n_cols = len(cols)

    for row_idx in range(n_rows + 1):          # row 0 = header
        for col_idx in range(n_cols):
            cell = tbl[row_idx, col_idx]
            cell.set_edgecolor("black")
            cell.set_linewidth(0.5)
            cell.set_facecolor("white")
            cell.set_text_props(fontfamily=serif, fontsize=9)

            if row_idx == 0:
                cell.set_text_props(
                    fontfamily=serif, fontsize=9.5,
                    fontweight="bold", color="black",
                )
                cell.set_facecolor("white")
                cell.set_linewidth(0.8)

    # Col widths: Model | Feature Set | AUC | PR | Brier | ECE | BalAcc | N
    col_widths = [0.165, 0.185, 0.085, 0.085, 0.085, 0.075, 0.1, 0.07]
    for col_idx, w in enumerate(col_widths):
        for row_idx in range(n_rows + 1):
            tbl[row_idx, col_idx].set_width(w)

    # Left-align text columns; centre metrics
    for row_idx in range(n_rows + 1):
        tbl[row_idx, 0].set_text_props(ha="left")
        tbl[row_idx, 0]._loc = "left"
        tbl[row_idx, 1].set_text_props(ha="left")
        tbl[row_idx, 1]._loc = "left"

    # Booktabs-style horizontal rules
    renderer = fig.canvas.get_renderer()
    tbl.auto_set_column_width(list(range(n_cols)))

    fig.text(
        0.05, 0.01,
        f"Table: {MODEL_PERF_CAPTION}",
        fontsize=8, fontstyle="italic", color="black",
        ha="left", va="bottom", fontfamily=serif,
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(str(path), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_model_performance_tex(rows: list[dict], path: Path) -> None:
    """Write a LaTeX booktabs table for thesis inclusion."""
    cols = list(rows[0].keys())
    col_spec = "ll" + "r" * (len(cols) - 2)

    lines: list[str] = []
    lines.append("\\begin{table}[htbp]")
    lines.append("  \\centering")
    lines.append("  \\caption{" + MODEL_PERF_CAPTION + "}")
    lines.append("  \\label{tab:model-performance}")
    lines.append("  \\small")
    lines.append("  \\begin{tabular}{" + col_spec + "}")
    lines.append("    \\toprule")

    header = " & ".join(f"\\textbf{{{c}}}" for c in cols)
    lines.append(f"    {header} \\\\")
    lines.append("    \\midrule")

    # Midrule before the phase 7.1 matched/latent rows
    for i, row in enumerate(rows):
        if i == 4:
            lines.append("    \\midrule")
        vals = " & ".join(row[c] for c in cols)
        lines.append(f"    {vals} \\\\")

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# === Render helper ===

def _render_graphviz(g, stem: str, out_dir: Path) -> None:
    """Write .dot source and attempt PNG render."""
    import graphviz  # noqa: F401

    dot_path = out_dir / f"{stem}.dot"
    dot_path.write_text(g.source, encoding="utf-8")
    print(f"[OK] DOT source written to {dot_path}")

    try:
        rendered = g.render(
            filename=stem,
            directory=str(out_dir),
            cleanup=True,
        )
        print(f"[OK] PNG rendered  to {rendered}")
    except graphviz.backend.execute.ExecutableNotFound:
        png_path = out_dir / f"{stem}.png"
        print()
        print("=" * 65)
        print("[WARNING] Graphviz system binary (dot) not found.")
        print("  .dot source saved. To render manually:")
        print(f"    dot -Tpng -Gdpi=300 {dot_path} -o {png_path}")
        print("=" * 65)


# === Main ===

def main() -> None:
    try:
        import graphviz  # noqa: F401
    except ImportError:
        print("[ERROR] Python graphviz package not installed.")
        print("  Install with:  pip install graphviz")
        sys.exit(1)

    try:
        import matplotlib  # noqa: F401
        import pandas  # noqa: F401
    except ImportError:
        print("[ERROR] matplotlib and/or pandas not installed.")
        print("  Install with:  pip install matplotlib pandas")
        sys.exit(1)

    print("\n-- 1. Methodology pipeline diagram --")
    g_pipe = build_pipeline_graph()
    _render_graphviz(g_pipe, "methodology_pipeline", FIGURES_DIR)

    print("\n-- 2. CONSORT-style sample flow --")
    g_flow = build_sample_flow_graph()
    _render_graphviz(g_flow, "methodology_sample_flow", FIGURES_DIR)

    print("\n-- 3. Feature-set summary table --")
    csv_path = TABLES_DIR / "methodology_feature_sets.csv"
    write_feature_csv(csv_path)
    print(f"[OK] CSV  written  to {csv_path}")

    png_path = FIGURES_DIR / "methodology_feature_sets.png"
    render_feature_table_png(png_path)
    print(f"[OK] PNG  rendered to {png_path}")

    print("\n-- 4. Model performance summary table --")
    perf_rows = _load_model_comparison(TABLES_DIR)

    perf_png = FIGURES_DIR / "model_performance_summary.png"
    render_model_performance_png(perf_rows, perf_png)
    print(f"[OK] PNG  rendered to {perf_png}")

    perf_tex = TABLES_DIR / "model_performance_summary.tex"
    write_model_performance_tex(perf_rows, perf_tex)
    print(f"[OK] TEX  written  to {perf_tex}")

    print("\nAll methodology visuals generated successfully.")


if __name__ == "__main__":
    main()
