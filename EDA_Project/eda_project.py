"""
=====================================================================
  EXPLORATORY DATA ANALYSIS (EDA) PROJECT  —  Task 3
  Internship @ Thiranex  |  Due: 22 Jun 2026
  Author: <Your Name>
=====================================================================

What this script does (end-to-end):
  1. Loads & inspects the dataset
  2. Cleans the data  (missing values, duplicates, dtypes)
  3. Statistical summary  (describe, skewness, kurtosis)
  4. Univariate analysis  (distributions for every column)
  5. Bivariate / correlation analysis  (heatmap + pairplot)
  6. Key influencing factors  (top correlators to a target)
  7. Outlier detection  (IQR box-plots)
  8. Exports a structured PDF report automatically

Usage:
  python eda_project.py                      # uses built-in demo dataset
  python eda_project.py --csv your_file.csv  # use your own CSV
  python eda_project.py --csv your_file.csv --target ColumnName
"""

# ── std-lib ──────────────────────────────────────────────────────────
import argparse
import os
import sys
import warnings
import textwrap
from datetime import datetime

# ── third-party ──────────────────────────────────────────────────────
try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")           # non-interactive backend (works everywhere)
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.backends.backend_pdf import PdfPages
    import seaborn as sns
    from scipy import stats
except ImportError as e:
    sys.exit(
        f"\n[ERROR] Missing library: {e}\n"
        "Run:  pip install numpy pandas matplotlib seaborn scipy\n"
    )

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# ═══════════════════════════════════════════════════════════════════════
#  CONFIG  — edit these if you need to
# ═══════════════════════════════════════════════════════════════════════
REPORT_FILE = "EDA_Report.pdf"
MAX_CATS    = 15          # max unique values to treat a column as categorical
FIG_DPI     = 120


# ═══════════════════════════════════════════════════════════════════════
#  DEMO DATASET  (used when no CSV is supplied)
# ═══════════════════════════════════════════════════════════════════════
def make_demo_dataset() -> pd.DataFrame:
    """Synthetic employee dataset — realistic enough for a proper EDA."""
    np.random.seed(42)
    n = 500

    departments = np.random.choice(
        ["Engineering", "Sales", "HR", "Marketing", "Finance"], n,
        p=[0.35, 0.25, 0.15, 0.15, 0.10]
    )
    experience  = np.random.exponential(scale=5, size=n).clip(0, 30).round(1)
    base_salary = (
        35_000
        + experience * 2_800
        + np.where(departments == "Engineering", 15_000, 0)
        + np.where(departments == "Finance",     10_000, 0)
        + np.random.normal(0, 8_000, n)
    ).round(-2)

    df = pd.DataFrame({
        "Age":             np.random.randint(22, 60, n),
        "Gender":          np.random.choice(["Male", "Female", "Other"], n, p=[0.52, 0.45, 0.03]),
        "Department":      departments,
        "Experience_yrs":  experience,
        "Education":       np.random.choice(
                               ["Bachelor's", "Master's", "PhD", "Diploma"], n,
                               p=[0.50, 0.30, 0.12, 0.08]
                           ),
        "Salary_INR":      base_salary.clip(20_000, 200_000),
        "Projects_Done":   np.random.poisson(lam=4, size=n),
        "Satisfaction":    np.random.randint(1, 6, n),          # 1–5
        "Attrition":       np.random.choice([0, 1], n, p=[0.83, 0.17]),
    })

    # Inject some mess so cleaning step is meaningful
    mask = np.random.choice([True, False], n, p=[0.04, 0.96])
    df.loc[mask, "Salary_INR"] = np.nan
    df.loc[np.random.choice(n, 10, replace=False), "Age"] = np.nan
    df = pd.concat([df, df.sample(15, random_state=7)], ignore_index=True)  # duplicates
    return df


# ═══════════════════════════════════════════════════════════════════════
#  HELPER UTILITIES
# ═══════════════════════════════════════════════════════════════════════
def section(title: str) -> None:
    bar = "─" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


def save_fig(pdf: PdfPages, fig: plt.Figure, title: str = "") -> None:
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def title_page(pdf: PdfPages, dataset_name: str, shape: tuple) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#1a1a2e")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#1a1a2e")
    ax.axis("off")

    ax.text(0.5, 0.72, "Exploratory Data Analysis", ha="center", va="center",
            fontsize=36, fontweight="bold", color="#e2e8f0", transform=ax.transAxes)
    ax.text(0.5, 0.60, f"Dataset: {dataset_name}", ha="center", va="center",
            fontsize=18, color="#94a3b8", transform=ax.transAxes)
    ax.text(0.5, 0.52, f"Rows: {shape[0]:,}   |   Columns: {shape[1]}",
            ha="center", va="center", fontsize=15, color="#64748b", transform=ax.transAxes)
    ax.text(0.5, 0.40, "Thiranex Data Science Internship — Task 3",
            ha="center", va="center", fontsize=14, color="#7c3aed", transform=ax.transAxes)
    ax.text(0.5, 0.32, datetime.now().strftime("%d %B %Y"),
            ha="center", va="center", fontsize=13, color="#64748b", transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
#  STEP 1 — LOAD
# ═══════════════════════════════════════════════════════════════════════
def load_data(csv_path: str | None) -> tuple[pd.DataFrame, str]:
    if csv_path:
        if not os.path.exists(csv_path):
            sys.exit(f"[ERROR] File not found: {csv_path}")
        df = pd.read_csv(csv_path)
        name = os.path.basename(csv_path)
    else:
        print("[INFO] No CSV supplied — using built-in demo dataset.")
        df   = make_demo_dataset()
        name = "Employee_Demo_Dataset"
    return df, name


# ═══════════════════════════════════════════════════════════════════════
#  STEP 2 — INSPECT + CLEAN
# ═══════════════════════════════════════════════════════════════════════
def inspect_and_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    section("DATASET INSPECTION")
    print(df.head())
    print(f"\nShape : {df.shape}")
    print(f"\nDtypes:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nDuplicate rows: {df.duplicated().sum()}")

    log = {}

    # ── duplicates ───────────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates()
    log["duplicates_removed"] = before - len(df)
    print(f"\n[CLEAN] Removed {log['duplicates_removed']} duplicate rows.")

    # ── missing values ───────────────────────────────────────────────
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    missing_num  = df[num_cols].isnull().sum()
    missing_cat  = df[cat_cols].isnull().sum()

    for col in num_cols:
        if missing_num[col] > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"[CLEAN] '{col}': {missing_num[col]} NaNs → filled with median ({median_val:.2f})")

    for col in cat_cols:
        if missing_cat[col] > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"[CLEAN] '{col}': {missing_cat[col]} NaNs → filled with mode ('{mode_val}')")

    log["shape_after_clean"] = df.shape
    print(f"\n[CLEAN] Final shape: {df.shape}")
    return df, log


# ═══════════════════════════════════════════════════════════════════════
#  STEP 3 — STATISTICAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════
def statistical_summary(df: pd.DataFrame, pdf: PdfPages) -> pd.DataFrame:
    section("STATISTICAL SUMMARY")
    num_df = df.select_dtypes(include="number")

    desc = num_df.describe().T
    desc["skewness"]  = num_df.skew()
    desc["kurtosis"]  = num_df.kurtosis()
    desc["cv_%"]      = (num_df.std() / num_df.mean() * 100).round(2)

    print(desc.to_string())

    # ── render as a table page in the PDF ────────────────────────────
    fig, ax = plt.subplots(figsize=(14, max(3, len(desc) * 0.6 + 1.5)))
    ax.axis("off")

    cols_to_show = ["mean", "std", "min", "50%", "max", "skewness", "kurtosis", "cv_%"]
    tbl_data     = desc[cols_to_show].round(2)

    tbl = ax.table(
        cellText   = tbl_data.values,
        colLabels  = cols_to_show,
        rowLabels  = tbl_data.index,
        cellLoc    = "center",
        loc        = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)

    # colour header
    for j in range(len(cols_to_show)):
        tbl[0, j].set_facecolor("#1e3a5f")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    save_fig(pdf, fig, "Statistical Summary — Numerical Features")
    return desc


# ═══════════════════════════════════════════════════════════════════════
#  STEP 4 — UNIVARIATE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
def univariate_analysis(df: pd.DataFrame, pdf: PdfPages) -> None:
    section("UNIVARIATE ANALYSIS")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df.select_dtypes(include="object").columns
                if df[c].nunique() <= MAX_CATS]

    # ── numerical distributions ──────────────────────────────────────
    if num_cols:
        ncols = 3
        nrows = -(-len(num_cols) // ncols)      # ceiling div
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = axes.flatten()

        for i, col in enumerate(num_cols):
            ax = axes[i]
            sns.histplot(df[col].dropna(), kde=True, ax=ax, color="#4f86c6", edgecolor="white")
            ax.set_title(col, fontweight="bold")
            ax.set_xlabel("")
            skew = df[col].skew()
            ax.annotate(f"skew={skew:.2f}", xy=(0.97, 0.93), xycoords="axes fraction",
                        ha="right", fontsize=8, color="#555")

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.tight_layout()
        save_fig(pdf, fig, "Univariate Analysis — Numerical Distributions")

    # ── categorical distributions ────────────────────────────────────
    if cat_cols:
        ncols = min(3, len(cat_cols))
        nrows = -(-len(cat_cols) // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.array(axes).flatten()

        for i, col in enumerate(cat_cols):
            ax = axes[i]
            order = df[col].value_counts().index
            sns.countplot(data=df, y=col, order=order, ax=ax, palette="Blues_d")
            ax.set_title(col, fontweight="bold")
            ax.set_xlabel("Count")
            ax.set_ylabel("")

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.tight_layout()
        save_fig(pdf, fig, "Univariate Analysis — Categorical Distributions")


# ═══════════════════════════════════════════════════════════════════════
#  STEP 5 — CORRELATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
def correlation_analysis(df: pd.DataFrame, pdf: PdfPages) -> pd.DataFrame:
    section("CORRELATION ANALYSIS")

    num_df = df.select_dtypes(include="number")
    corr   = num_df.corr()

    print(corr.to_string())

    # ── heatmap ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(8, len(corr) * 1.2), max(6, len(corr) * 1.0)))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", linewidths=0.5,
        cmap="coolwarm", center=0, ax=ax,
        annot_kws={"size": 9}
    )
    ax.set_title("Correlation Heatmap", fontweight="bold")
    save_fig(pdf, fig)

    # ── pairplot (only if ≤ 6 numeric cols to keep it readable) ─────
    if 2 <= len(num_df.columns) <= 6:
        g = sns.pairplot(num_df.dropna(), corner=True, plot_kws={"alpha": 0.4})
        g.figure.suptitle("Pair Plot — Numerical Features", y=1.01, fontweight="bold")
        pdf.savefig(g.figure, bbox_inches="tight")
        plt.close(g.figure)

    return corr


# ═══════════════════════════════════════════════════════════════════════
#  STEP 6 — KEY INFLUENCING FACTORS
# ═══════════════════════════════════════════════════════════════════════
def key_influencers(
    df: pd.DataFrame,
    corr: pd.DataFrame,
    target: str | None,
    pdf: PdfPages
) -> None:
    section("KEY INFLUENCING FACTORS")

    num_cols = df.select_dtypes(include="number").columns.tolist()

    # auto-pick target if not provided
    if target is None or target not in num_cols:
        if target and target not in df.columns:
            print(f"[WARN] Target '{target}' not found. Auto-selecting.")
        target = num_cols[-1]           # use last numeric col as default
        print(f"[INFO] Target column: '{target}'")

    if len(num_cols) < 2:
        print("[SKIP] Not enough numeric columns for influencer analysis.")
        return

    top = (
        corr[target]
        .drop(target, errors="ignore")
        .abs()
        .sort_values(ascending=False)
        .head(8)
    )
    print(f"\nTop correlators with '{target}':\n{top.to_string()}")

    # bar chart
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#e74c3c" if corr.loc[f, target] < 0 else "#2ecc71" for f in top.index]
    bars   = ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1], edgecolor="white")
    ax.set_xlabel("|Pearson r|", fontweight="bold")
    ax.set_title(f"Top Influencing Factors → '{target}'", fontweight="bold")
    ax.axvline(0.3, ls="--", color="#555", lw=1, label="Moderate threshold (0.3)")
    ax.legend(fontsize=9)

    for bar, val in zip(bars, top.values[::-1]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    fig.tight_layout()
    save_fig(pdf, fig, f"Key Influencing Factors for '{target}'")

    # boxplots grouped by strongest categorical predictor
    cat_cols = [c for c in df.select_dtypes(include="object").columns
                if df[c].nunique() <= MAX_CATS]
    if cat_cols and target in df.columns:
        best_cat = max(cat_cols, key=lambda c: df.groupby(c)[target].mean().std())
        fig, ax  = plt.subplots(figsize=(10, 5))
        order    = df.groupby(best_cat)[target].median().sort_values(ascending=False).index
        sns.boxplot(data=df, x=best_cat, y=target, order=order, ax=ax, palette="Set2")
        ax.set_title(f"'{target}' distribution by '{best_cat}'", fontweight="bold")
        ax.set_xlabel(best_cat)
        fig.tight_layout()
        save_fig(pdf, fig)


# ═══════════════════════════════════════════════════════════════════════
#  STEP 7 — OUTLIER DETECTION
# ═══════════════════════════════════════════════════════════════════════
def outlier_detection(df: pd.DataFrame, pdf: PdfPages) -> None:
    section("OUTLIER DETECTION (IQR Method)")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    summary  = {}

    for col in num_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr     = q3 - q1
        lower   = q1 - 1.5 * iqr
        upper   = q3 + 1.5 * iqr
        n_out   = ((df[col] < lower) | (df[col] > upper)).sum()
        summary[col] = {"Q1": q1, "Q3": q3, "IQR": iqr,
                        "Lower_fence": lower, "Upper_fence": upper,
                        "Outliers": n_out,
                        "Outlier_%": round(n_out / len(df) * 100, 2)}

    out_df = pd.DataFrame(summary).T
    print(out_df[["IQR", "Lower_fence", "Upper_fence", "Outliers", "Outlier_%"]].to_string())

    # box plots
    ncols = 3
    nrows = -(-len(num_cols) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, col in enumerate(num_cols):
        ax = axes[i]
        sns.boxplot(data=df, y=col, ax=ax, color="#4f86c6", flierprops={"marker": "o",
                    "markerfacecolor": "#e74c3c", "markersize": 4})
        n = summary[col]["Outliers"]
        ax.set_title(f"{col}\n({n} outliers)", fontweight="bold", fontsize=10)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    save_fig(pdf, fig, "Outlier Detection — Box Plots (IQR)")


# ═══════════════════════════════════════════════════════════════════════
#  STEP 8 — INSIGHTS SUMMARY PAGE
# ═══════════════════════════════════════════════════════════════════════
def insights_page(
    df: pd.DataFrame,
    corr: pd.DataFrame,
    clean_log: dict,
    target: str | None,
    pdf: PdfPages
) -> None:
    num_df   = df.select_dtypes(include="number")
    num_cols = num_df.columns.tolist()

    # auto key insights
    insights = []
    insights.append(f"Dataset shape after cleaning: {clean_log['shape_after_clean']} "
                    f"({clean_log['duplicates_removed']} duplicates removed)")

    for col in num_cols:
        sk = df[col].skew()
        if abs(sk) > 1:
            direction = "right (positive)" if sk > 0 else "left (negative)"
            insights.append(f"'{col}' is highly {direction}-skewed (skew={sk:.2f}) — "
                            "consider log-transformation for modelling.")

    if target and target in num_cols and len(num_cols) > 1:
        top2 = corr[target].drop(target, errors="ignore").abs().nlargest(2)
        for feat, val in top2.items():
            direction = "positively" if corr.loc[feat, target] > 0 else "negatively"
            insights.append(f"'{feat}' is {direction} correlated with '{target}' "
                            f"(r={corr.loc[feat, target]:.2f}), making it a strong predictor.")

    # outlier insight
    for col in num_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr    = q3 - q1
        n_out  = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
        pct    = n_out / len(df) * 100
        if pct > 5:
            insights.append(f"'{col}' has {n_out} outliers ({pct:.1f}% of data) — "
                            "worth investigating or capping.")

    # categorical observation
    cat_cols = [c for c in df.select_dtypes(include="object").columns
                if df[c].nunique() <= MAX_CATS]
    for col in cat_cols:
        top_cat = df[col].value_counts().idxmax()
        top_pct = df[col].value_counts(normalize=True).iloc[0] * 100
        if top_pct > 50:
            insights.append(f"'{col}' is dominated by '{top_cat}' ({top_pct:.0f}%) — "
                            "class imbalance may affect modelling.")

    # ── render as a styled text page ─────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#f8fafc")
    ax  = fig.add_axes([0.05, 0.05, 0.90, 0.90])
    ax.set_facecolor("#f8fafc")
    ax.axis("off")

    ax.text(0.5, 0.97, "Key Insights & Analytical Findings",
            ha="center", va="top", fontsize=16, fontweight="bold",
            color="#1e3a5f", transform=ax.transAxes)

    y = 0.88
    for idx, insight in enumerate(insights, 1):
        wrapped = textwrap.fill(f"  {idx}.  {insight}", width=90)
        ax.text(0.02, y, wrapped, va="top", fontsize=10, color="#334155",
                transform=ax.transAxes, linespacing=1.5)
        y -= 0.10 * (1 + wrapped.count("\n") * 0.5)
        if y < 0.05:
            break

    ax.text(0.5, 0.03,
            "Generated by EDA Pipeline — Thiranex Data Science Internship",
            ha="center", va="bottom", fontsize=8, color="#94a3b8",
            transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(description="EDA Project — Task 3")
    parser.add_argument("--csv",    default=None, help="Path to your CSV file")
    parser.add_argument("--target", default=None, help="Target column for influencer analysis")
    args = parser.parse_args()

    # ── load ─────────────────────────────────────────────────────────
    df, dataset_name = load_data(args.csv)

    # ── open PDF report ──────────────────────────────────────────────
    with PdfPages(REPORT_FILE) as pdf:

        # meta
        pdf_info = pdf.infodict()
        pdf_info["Title"]   = "EDA Report — Task 3"
        pdf_info["Author"]  = "Thiranex Internship"
        pdf_info["Subject"] = "Exploratory Data Analysis"

        # ── 0. title ─────────────────────────────────────────────────
        title_page(pdf, dataset_name, df.shape)

        # ── 1 & 2. inspect + clean ───────────────────────────────────
        df, clean_log = inspect_and_clean(df)

        # ── 3. stats ─────────────────────────────────────────────────
        statistical_summary(df, pdf)

        # ── 4. univariate ────────────────────────────────────────────
        univariate_analysis(df, pdf)

        # ── 5. correlation ───────────────────────────────────────────
        corr = correlation_analysis(df, pdf)

        # resolve target
        num_cols = df.select_dtypes(include="number").columns.tolist()
        target   = args.target if args.target in num_cols else (num_cols[-1] if num_cols else None)

        # ── 6. key influencers ───────────────────────────────────────
        key_influencers(df, corr, target, pdf)

        # ── 7. outliers ──────────────────────────────────────────────
        outlier_detection(df, pdf)

        # ── 8. insights summary ──────────────────────────────────────
        insights_page(df, corr, clean_log, target, pdf)

    section("DONE")
    print(f"\n✅  PDF report saved → {os.path.abspath(REPORT_FILE)}")
    print("    Submit 'EDA_Report.pdf' along with this script.\n")


if __name__ == "__main__":
    main()
