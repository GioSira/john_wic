import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from scipy.stats import gaussian_kde
import seaborn as sns
import numpy as np


# ── Shared palette ────────────────────────────────────────────────────

COLOR_MAIN = "#1f78b4"  # blue  — KDE fill
COLOR_THRESHOLD = "#e31a1c"  # red   — threshold line + shading
COLOR_ANNOT = "#e31a1c"
COLOR_MEAN = "#e31a1c"  # red   — mean line + CI
COLOR_THRESH = "#e31a1c"  # red   — H=0.8 threshold line
COLOR_GRID = "gray"
COLOR_SPINE = "#bbbbbb"
COLOR_TICK = "#bbbbbb"

COLOR_VIOLIN = "#1f78b4"  # blue  — violin fill
COLOR_MAX = "#e31a1c"  # red   — max noise (worst case)
COLOR_ELBOW = "#333333"  # dark  — elbow vertical line

THRESHOLD_LIM = 40  # x-axis range for elbow analysis
THRESHOLD = 10  # group senses >= threshold as "10+"
ENTROPY_THRESH = 0.8  # high-entropy cutoff


def load_and_prep_data(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # Calcoliamo il numero di sensi unici per ogni parola
    df["num_senses"] = df["senses"].apply(lambda x: len(x.keys()))

    # Filtriamo eventuali anomalie o parole con frequenza 0
    df = df[df["instance_count"] > 0].copy()

    return df


def analyze_noise_ratio_journal(df, out_prefix="noise_ratio_elbow"):
    """
    Compute entropy/frequency noise ratio, detect the elbow point,
    and produce a publication-ready figure.

    Parameters
    ----------
    df         : DataFrame with columns 'instance_count' and 'entropy_level'
    out_prefix : filename stem for .pdf / .png output

    Returns
    -------
    df_enriched : df with 'noise_ratio' column added
    elbow_point : int — instance count where noise stabilises
    """

    FS_TITLE = 18
    FS_AXIS = 16
    FS_TICK = 14
    FS_ANNOT = 16

    # ── Compute noise ratio ───────────────────────────────────────────
    df = df[df["instance_count"] > 0].copy()
    df["noise_ratio"] = df["entropy_level"] / df["instance_count"]

    # ── Aggregate trend up to threshold_limit ────────────────────────
    df_trend = df[df["instance_count"] <= THRESHOLD_LIM].copy()
    trend_stats = (
        df_trend.groupby("instance_count")["noise_ratio"]
        .agg(["mean", "max"])
        .reset_index()
    )

    # ── Elbow detection (smoothed second derivative) ──────────────────
    trend_stats["max_smooth"] = (
        trend_stats["max"].rolling(window=3, center=True).mean().ffill().bfill()
    )
    trend_stats["diff2"] = trend_stats["max_smooth"].diff().diff().abs()
    trend_stats["is_stable"] = trend_stats["diff2"] < 0.01
    stable_run = trend_stats["is_stable"].rolling(window=3).sum()
    elbow_point = trend_stats.loc[stable_run == 3, "instance_count"].min()
    print(f"  Noise stabilises at N = {elbow_point}")

    # ── Figure ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f9f9f9")

    # Max noise line
    sns.lineplot(
        data=trend_stats,
        x="instance_count",
        y="max",
        marker="o",
        markersize=6,
        color=COLOR_MAX,
        linewidth=2.2,
        markerfacecolor=COLOR_MAX,
        markeredgecolor="white",
        markeredgewidth=0.8,
        label="Max noise ratio (worst case)",
        ax=ax,
        zorder=4,
    )

    # Mean noise line  (dashed, open markers — same convention as decoders)
    sns.lineplot(
        data=trend_stats,
        x="instance_count",
        y="mean",
        marker="s",
        markersize=5.5,
        color=COLOR_MEAN,
        linewidth=1.9,
        linestyle="--",
        markerfacecolor="white",
        markeredgecolor=COLOR_MEAN,
        markeredgewidth=1.6,
        label="Mean noise ratio",
        ax=ax,
        zorder=4,
    )

    # ── Elbow annotation ─────────────────────────────────────────────
    if pd.notna(elbow_point):
        ax.axvline(
            elbow_point, color=COLOR_ELBOW, lw=1.4, ls=(0, (5, 3)), alpha=0.75, zorder=3
        )

        y_arrow = trend_stats.loc[
            trend_stats["instance_count"] == elbow_point, "max"
        ].values
        y_arrow = float(y_arrow[0]) if len(y_arrow) else ax.get_ylim()[1] * 0.4

        ax.annotate(
            f"Noise stabilises\nat $N = {int(elbow_point)}$",
            xy=(elbow_point, y_arrow),
            xytext=(elbow_point + 4, y_arrow + ax.get_ylim()[1] * 0.15),
            fontsize=FS_ANNOT,
            color=COLOR_ELBOW,
            fontweight="bold",
            va="center",
            ha="left",
            arrowprops=dict(
                arrowstyle="-|>",
                color=COLOR_ELBOW,
                lw=1.2,
                mutation_scale=10,
                connectionstyle="arc3,rad=-0.15",
            ),
            zorder=5,
        )

    # ── Axes & grid ───────────────────────────────────────────────────
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    ax.grid(axis="y", which="major", lw=0.7, alpha=0.45, color=COLOR_GRID, zorder=0)
    ax.grid(axis="y", which="minor", lw=0.3, alpha=0.20, color=COLOR_GRID, zorder=0)
    ax.grid(axis="x", which="major", lw=0.4, alpha=0.20, color=COLOR_GRID, zorder=0)

    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(COLOR_SPINE)
    ax.tick_params(axis="both", labelsize=FS_TICK, color=COLOR_TICK)

    ax.set_xlabel("Instance Count ($N$)", fontsize=FS_AXIS, labelpad=8)
    ax.set_ylabel(r"Noise Ratio ($H \,/\, N$)", fontsize=FS_AXIS, labelpad=8)

    # ── Legend ────────────────────────────────────────────────────────
    ax.legend(
        fontsize=FS_ANNOT,
        framealpha=0.95,
        edgecolor="#cccccc",
        handlelength=2.4,
        handletextpad=0.6,
        labelspacing=0.45,
        borderpad=0.75,
        loc="upper right",
    )

    # ── Title ─────────────────────────────────────────────────────────
    ax.set_title(
        "Entropy-to-Frequency Ratio: Noise Threshold",
        fontsize=FS_TITLE,
        fontweight="bold",
        pad=11,
        color="#1a1a1a",
    )

    # ── Export ────────────────────────────────────────────────────────
    plt.tight_layout()
    plt.savefig(f"{out_prefix}.png", bbox_inches="tight", dpi=300, facecolor="white")
    plt.close()


def plot_entropy_threshold_analysis(
    df, instance_count=0, out_prefix="entropy_threshold_analysis"
):
    """
    Parameters
    ----------
    df             : DataFrame with columns 'entropy_level' and 'instance_count'
    instance_count : minimum instance count filter (default 0 = no filter)
    out_prefix     : filename stem for .pdf / .png output
    """

    FS_TITLE = 18
    FS_AXIS = 16
    FS_TICK = 14
    FS_ANNOT = 16

    df_filtered = df[df["instance_count"] >= instance_count].copy()

    # ── Compute y_top from KDE directly ──────────────────────────────
    # With the Agg backend, ax.get_ylim() is unreliable before plt.show().
    # We compute the KDE peak ourselves so y_top is available immediately
    # for fill_betweenx, axvline, and annotation positioning.
    kde = gaussian_kde(df_filtered["entropy_level"], bw_method="scott")
    x_eval = np.linspace(0.0, 1.0, 500)
    y_top = kde(x_eval).max() * 1.12  # 12% headroom above the KDE peak

    # ── Figure ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f9f9f9")

    # ── KDE plot ──────────────────────────────────────────────────────
    sns.kdeplot(
        df_filtered["entropy_level"],
        fill=True,
        color=COLOR_MAIN,
        alpha=0.25,
        linewidth=2.2,
        clip=(0.0, 1.0),
        ax=ax,
        zorder=3,
    )

    # ── Freeze y limits before drawing anything else ─────────────────
    ax.set_ylim(0, y_top)

    # ── Threshold vertical line ───────────────────────────────────────
    ax.axvline(THRESHOLD, color=COLOR_THRESHOLD, lw=1.8, ls=(0, (5, 3)), zorder=4)

    # ── Red shading above threshold ───────────────────────────────────
    ax.fill_betweenx(
        [0, y_top], THRESHOLD, 1.0, color=COLOR_THRESHOLD, alpha=0.08, zorder=2
    )

    # ── Annotation ────────────────────────────────────────────────────
    perc_above = (df_filtered["entropy_level"] > THRESHOLD).mean() * 100

    ax.annotate(
        f"{perc_above:.2f}% of lemmas\nexhibit maximal ambiguity",
        # Arrow tip: on the threshold line, mid-height
        xy=(THRESHOLD, y_top * 0.55),
        xycoords="data",
        # Text: top-left corner of the axes — always in the low-KDE zone
        xytext=(0.05, 0.92),
        textcoords="axes fraction",
        fontsize=FS_ANNOT,
        color=COLOR_ANNOT,
        fontweight="bold",
        va="top",
        ha="left",
        arrowprops=dict(
            arrowstyle="-|>",
            color=COLOR_ANNOT,
            lw=1.5,
            mutation_scale=14,
            connectionstyle="arc3,rad=-0.2",
        ),
        zorder=6,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor=COLOR_ANNOT,
            alpha=0.85,
            lw=1.2,
        ),
    )

    # ── Axes & grid ───────────────────────────────────────────────────
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0, y_top)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    ax.grid(axis="y", which="major", lw=0.7, alpha=0.45, color=COLOR_GRID, zorder=0)
    ax.grid(axis="y", which="minor", lw=0.3, alpha=0.20, color=COLOR_GRID, zorder=0)
    ax.grid(axis="x", which="major", lw=0.4, alpha=0.20, color=COLOR_GRID, zorder=0)

    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(COLOR_SPINE)
    ax.tick_params(axis="both", labelsize=FS_TICK, color=COLOR_TICK)

    ax.set_xlabel("Normalized Entropy", fontsize=FS_AXIS, labelpad=8)
    ax.set_ylabel("Density", fontsize=FS_AXIS, labelpad=8)

    # ── Title ─────────────────────────────────────────────────────────
    ax.set_title(
        "Identification of High-Entropy Lemmas",
        fontsize=FS_TITLE,
        fontweight="bold",
        pad=11,
        color="#1a1a1a",
    )

    # ── Export ────────────────────────────────────────────────────────
    plt.tight_layout()
    plt.savefig(f"{out_prefix}.png", bbox_inches="tight", dpi=300, facecolor="white")
    plt.close()


def plot_final_journal_entropy(df, instance_count=0):

    # --- 1. PRE-PROCESSING SCIENTIFICO ---
    # Filtro: Rimuoviamo i lemmi dove OGNI senso ha frequenza 1 (rumore statistico)
    df_clean = df[df["senses"].apply(lambda x: any(v > 1 for v in x.values()))].copy()
    df_clean = df_clean[df_clean["instance_count"] >= instance_count]

    # Calcolo numero di sensi e raggruppamento coda lunga (8+)
    df_clean["n_senses"] = df_clean["senses"].apply(len)
    limit = 8
    df_clean["group"] = df_clean["n_senses"].apply(
        lambda x: str(x) if x < limit else f"{limit}+"
    )
    order = [str(i) for i in range(2, limit)] + [f"{limit}+"]

    # --- 2. CONFIGURAZIONE ESTETICA ---
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.labelweight": "bold",
        }
    )

    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

    # --- 3. IL GRAFICO (Layered Architecture) ---

    # Layer 1: Strip Plot (I "Punti" - mostriamo la densità reale con trasparenza)
    sns.stripplot(
        data=df_clean,
        x="group",
        y="entropy_level",
        order=order,
        palette="flare",
        alpha=0.15,
        jitter=0.3,
        size=3,
        ax=ax,
    )

    # Layer 2: Violin Plot (La "Forma" - mostriamo la distribuzione solo su un lato)
    # Usiamo un trucco per renderlo più elegante: metà violino
    sns.violinplot(
        data=df_clean,
        x="group",
        y="entropy_level",
        order=order,
        palette="flare",
        bw_adjust=0.5,
        cut=0,
        inner=None,
        alpha=0.3,
        ax=ax,
    )

    # Layer 3: Point Plot (Il "Trend" - la verità statistica)
    sns.pointplot(
        data=df_clean,
        x="group",
        y="entropy_level",
        order=order,
        color="black",
        scale=0.9,
        markers="D",
        capsize=0.15,
        linestyles="--",
        errwidth=1.5,
        ax=ax,
        label="Mean Entropy (CI 95%)",
    )

    # --- 4. SOGLIA CRITICA 0.9 ---
    ax.axhline(0.9, color="#d73027", linestyle=":", linewidth=2, alpha=0.8)
    # ax.text(len(order)-1.5, 0.92, 'Threshold: High Ambiguity ($H>0.9$)',
    #        color='#d73027', fontweight='bold', fontsize=10, ha='center')

    # --- 5. RIFINITURE E LABELS ---
    ax.set_title(
        "Lexical Ambiguity Analysis: Entropy vs. Polysemy", fontsize=18, pad=25
    )
    ax.set_xlabel("Degree of Polysemy (Number of Senses)", fontsize=14, labelpad=15)
    ax.set_ylabel("Normalized Semantic Entropy ($H$)", fontsize=14, labelpad=15)

    # Pulizia assi (Tufte Style)
    sns.despine(offset=10, trim=True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)

    # Annotazione dati filtrati
    # n_removed = len(df) - len(df_clean)
    # plt.annotate(f"Filter active: excluded {n_removed} sparse entries (single-instance senses).",
    #             xy=(0.02, 0.02), xycoords='axes fraction', fontsize=9, style='italic', alpha=0.7)

    plt.tight_layout()
    plt.savefig("./final_entropy_analysis.png", format="png", dpi=300)


def plot_refined_research_visual_journal(
    df, instance_count=100, out_prefix="entropy_violin"
):

    # ── Shared palette ────────────────────────────────────────────────────
    COLOR_VIOLIN = "#1f78b4"  # blue  — violin fill
    COLOR_MEAN = "#e31a1c"  # red   — mean line + CI
    COLOR_THRESH = "#e31a1c"  # red   — H=0.8 threshold line
    COLOR_GRID = "gray"
    COLOR_SPINE = "#bbbbbb"
    COLOR_TICK = "#bbbbbb"

    # ── 1. Filtering ─────────────────────────────────────────────────
    df_f = df[df["instance_count"] >= instance_count]
    df_f["num_senses"] = df_f["senses"].apply(len)
    df_f["senses_label"] = df_f["num_senses"].apply(
        lambda x: f"{x}" if x < THRESHOLD else f"{THRESHOLD}+"
    )

    order = [str(i) for i in range(2, THRESHOLD)] + [f"{THRESHOLD}+"]
    # Keep only categories that actually exist in the data
    order = [o for o in order if o in df_f["senses_label"].values]

    # ── 2. Figure ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f9f9f9")

    # ── 3. Violin plot ────────────────────────────────────────────────
    sns.violinplot(
        data=df_f,
        x="senses_label",
        y="entropy_level",
        order=order,
        color=COLOR_VIOLIN,
        alpha=0.30,
        linewidth=0,
        inner=None,
        ax=ax,
        zorder=2,
        saturation=0.9,
    )
    # Re-apply alpha (seaborn violinplot ignores alpha on facecolor in some versions)
    for poly in ax.collections:
        poly.set_alpha(0.28)
        poly.set_facecolor(COLOR_VIOLIN)
        poly.set_edgecolor("none")

    # ── 4. Point plot (mean + 95% CI) ─────────────────────────────────
    sns.pointplot(
        data=df_f,
        x="senses_label",
        y="entropy_level",
        order=order,
        color=COLOR_MEAN,
        markers="o",
        capsize=0.08,
        label="Mean entropy (95% CI)",
        ax=ax,
    )
    # Make pointplot markers solid
    for artist in ax.lines:
        artist.set_zorder(5)

    # ── 5. High-entropy threshold line ───────────────────────────────
    ax.axhline(
        ENTROPY_THRESH, color=COLOR_THRESH, lw=1.6, ls=(0, (5, 3)), alpha=0.80, zorder=4
    )
    ax.text(
        len(order) - 0.55,
        ENTROPY_THRESH + 0.018,
        f"$H = {ENTROPY_THRESH}$ (high-ambiguity threshold)",
        fontsize=14,
        color=COLOR_THRESH,
        va="bottom",
        ha="right",
        style="italic",
    )

    # ── 6. Axes & grid ────────────────────────────────────────────────
    ax.set_ylim(0.0, 1.05)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.grid(axis="y", which="major", lw=0.7, alpha=0.45, color=COLOR_GRID, zorder=0)
    ax.grid(axis="y", which="minor", lw=0.3, alpha=0.20, color=COLOR_GRID, zorder=0)
    ax.grid(axis="x", which="major", lw=0.4, alpha=0.15, color=COLOR_GRID, zorder=0)

    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(COLOR_SPINE)
    ax.tick_params(axis="both", labelsize=16, color=COLOR_TICK)

    ax.set_xlabel("Number of senses per lemma", fontsize=16, labelpad=10)
    ax.set_ylabel("Entropy level ($H$)", fontsize=16, labelpad=10)

    # ── 7. Legend ─────────────────────────────────────────────────────
    violin_patch = Line2D(
        [0],
        [0],
        marker="s",
        color="none",
        markerfacecolor=COLOR_VIOLIN,
        markerfacecoloralt=COLOR_VIOLIN,
        markersize=11,
        alpha=0.5,
        label="Entropy distribution (KDE)",
    )
    mean_handle = Line2D(
        [0],
        [0],
        color=COLOR_MEAN,
        lw=2.0,
        marker="o",
        markersize=6,
        markerfacecolor=COLOR_MEAN,
        label="Mean entropy (95% CI)",
    )
    thresh_handle = Line2D(
        [0],
        [0],
        color=COLOR_THRESH,
        lw=1.6,
        ls=(0, (5, 3)),
        label=f"$H = {ENTROPY_THRESH}$ threshold",
    )

    ax.legend(
        handles=[violin_patch, mean_handle, thresh_handle],
        fontsize=20,
        framealpha=0.95,
        edgecolor="#cccccc",
        handlelength=2.2,
        handletextpad=0.6,
        labelspacing=0.5,
        borderpad=0.8,
        loc="lower right",
    )

    # ── 8. Title ──────────────────────────────────────────────────────
    ax.set_title(
        "Semantic Entropy vs. Polysemy Degree",
        fontsize=18,
        fontweight="bold",
        pad=12,
        color="#1a1a1a",
    )

    # ── 9. Export ─────────────────────────────────────────────────────
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(f"{out_prefix}.png", bbox_inches="tight", dpi=300, facecolor="white")
    plt.close()


if __name__ == "__main__":

    JSON_FILE = "/xl-wsd-data/es/es_entropy_scores.json"

    df = load_and_prep_data(JSON_FILE)

    analyze_noise_ratio_journal(df)

    plot_refined_research_visual_journal(df, instance_count=10)

    plot_entropy_threshold_analysis(df, instance_count=10)
