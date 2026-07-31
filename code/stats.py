import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from scipy.stats import gaussian_kde
import seaborn as sns
import numpy as np


plt.rcParams.update(
    {
        "font.family": "serif",  # I font serif (es. Times) sono standard nei paper
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,  # Alta risoluzione per la stampa
        "savefig.bbox": "tight",  # Evita che i bordi vengano tagliati
    }
)

sns.set_theme(style="ticks", context="paper")
# Palette colorblind-friendly (viridis o palette Seaborn)
palette = sns.color_palette("colorblind")

# ── Palette — identical to plot_polysemy.py ───────────────────────────
COLOR_MAIN = "#1f78b4"  # blue  — KDE fill
COLOR_THRESHOLD = "#e31a1c"  # red   — threshold line + shading
COLOR_ANNOT = "#e31a1c"
COLOR_GRID = "gray"
COLOR_SPINE = "#bbbbbb"
COLOR_TICK = "#bbbbbb"
THRESHOLD = 0.8


def load_and_prep_data(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # Calcoliamo il numero di sensi unici per ogni parola
    df["num_senses"] = df["senses"].apply(lambda x: len(x.keys()))

    # Filtriamo eventuali anomalie o parole con frequenza 0
    df = df[df["instance_count"] > 0].copy()
    return df


def plot_entropy_distribution(df, instance_count=0):
    """
    Fig 1: Distribuzione dell'entropia semantica nel dataset.
    Mostra se la maggior parte delle parole ha un solo senso (entropia ~0) o è molto ambigua.
    """

    filtered_df = df[df["instance_count"] >= instance_count]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    sns.histplot(
        filtered_df["entropy_level"],
        bins=50,
        kde=True,
        color=palette[0],
        ax=ax,
        edgecolor="black",
        alpha=0.6,
    )

    ax.set_title("Distribution of Semantic Entropy Across Vocabulary", pad=15)
    ax.set_xlabel("Entropy Level")
    ax.set_ylabel("Number of Words (Frequency)")

    # Rimuoviamo i bordi superiore e destro per un look più pulito (Tufte style)
    sns.despine(trim=True)

    plt.tight_layout()
    plt.savefig("./fig1_entropy_distribution.png", format="png")


def plot_frequency_vs_entropy(df, instance_count=0):
    """
    Fig 2: Relazione tra frequenza della parola (Log) ed Entropia.
    Risponde alla domanda: le parole più usate sono anche le più polisemiche/ambigue?
    """

    filtered_df = df[df["instance_count"] >= instance_count]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # Usiamo un jointplot o uno scatterplot con hexbin se i dati sono molto densi.
    # Qui usiamo uno scatter plot con alpha per gestire l'overplotting e una linea di trend.
    sns.regplot(
        x=np.log10(filtered_df["instance_count"]),
        y=filtered_df["entropy_level"],
        scatter_kws={"alpha": 0.3, "color": palette[1], "s": 20, "edgecolor": "none"},
        line_kws={"color": palette[3], "linewidth": 2},
        ax=ax,
    )

    ax.set_title("Word Frequency vs. Semantic Entropy", pad=15)
    ax.set_xlabel("Instance Count (Log$_{10}$ Scale)")
    ax.set_ylabel("Entropy Level")

    sns.despine(trim=True)

    plt.tight_layout()
    plt.savefig("./fig2_freq_vs_entropy.png", format="png")


def plot_top_words_complexity(df, top_n=15):
    """
    Fig 3: I lemmi più frequenti e la loro complessità (Entropia vs Numero di Sensi).
    Un grafico a bolle/scatter categorico per illustrare casi specifici.
    """
    # Prendiamo le top N parole più frequenti
    top_df = df.nlargest(top_n, "instance_count").copy()

    # Rimuoviamo il tag POS per rendere le etichette più leggibili (es. 'be_VERB' -> 'be')
    top_df["clean_word"] = top_df["key"].apply(lambda x: x.split("_")[0])

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(
        data=top_df,
        x="num_senses",
        y="entropy_level",
        size="instance_count",
        hue="entropy_level",
        sizes=(100, 1000),
        palette="viridis",
        legend=False,
        ax=ax,
        alpha=0.8,
        edgecolor="black",
    )

    # Aggiungiamo le etichette delle parole accanto ai punti
    for _, row in top_df.iterrows():
        ax.text(
            row["num_senses"] + 0.3,
            row["entropy_level"],
            row["clean_word"],
            fontsize=11,
            ha="left",
            va="center",
        )

    ax.set_title(f"Semantic Complexity of Top {top_n} Most Frequent Words", pad=15)
    ax.set_xlabel("Number of Distinct Senses")
    ax.set_ylabel("Entropy Level")

    # Estendiamo un po' l'asse X per fare spazio alle etichette di testo
    ax.set_xlim(
        left=top_df["num_senses"].min() - 1, right=top_df["num_senses"].max() + 3
    )

    sns.despine()

    plt.tight_layout()
    plt.savefig("./fig3_top_words_complexity.png", format="png")


def plot_senses_vs_entropy_trend(df, instance_count=0):
    """
    Fig 4: Correlazione globale tra Numero di Sensi ed Entropia.
    Mostra come l'entropia media cresce all'aumentare del numero di significati,
    includendo gli intervalli di confidenza.
    """

    filtered_df = df[df["instance_count"] >= instance_count]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Raggruppiamo i dati rari (es. parole con > 15 sensi) per evitare rumore statistico alla fine del grafico.
    # Calcoliamo il 99° percentile per tagliare gli outlier estremi sull'asse X se necessario,
    # ma per ora tracciamo tutto in modo pulito.

    # Usiamo un pointplot: unisce i punti medi con una linea e mostra le barre di errore (CI 95%)
    sns.pointplot(
        data=filtered_df,
        x="num_senses",
        y="entropy_level",
        color=palette[2],  # Usa un colore della palette definita all'inizio
        capsize=0.2,  # Trattino orizzontale sulle barre di errore
        markers="o",
        linestyles="-",
        ax=ax,
    )

    ax.set_title("Correlation between Number of Senses and Average Entropy", pad=15)
    ax.set_xlabel("Number of Distinct Senses (Ascending Order)")
    ax.set_ylabel("Entropy Level (Mean with 95% CI)")

    # Per evitare che l'asse X diventi illeggibile se ci sono parole con 30+ sensi,
    # sfoltiamo le etichette sull'asse X mostrandone una su due o mantenendo un font piccolo
    plt.xticks(rotation=45)

    sns.despine(trim=True)

    plt.tight_layout()
    plt.savefig("./fig4_senses_vs_entropy_trend.png", format="png")


def plot_sophisticated_entropy_analysis(df, instance_count=0):
    """
    Fig 4: Analisi avanzata dell'Entropia per Numero di Sensi.
    Utilizza un Boxenplot per un look moderno e leggibile.
    """
    # 1. Preparazione: Raggruppiamo la 'coda lunga' (es. tutti i sensi > 10)
    # Questo evita che il grafico diventi troppo largo e vuoto a destra.
    plot_df = df.copy()
    threshold = 10
    plot_df["senses_grouped"] = plot_df["num_senses"].apply(
        lambda x: f"{x}" if x < threshold else f"{threshold}+"
    )

    # Ordiniamo le categorie correttamente
    order = [f"{i}" for i in range(1, threshold)] + [f"{threshold}+"]

    # 2. Creazione del Grafico
    fig, ax = plt.subplots(figsize=(12, 7))

    # Il Boxenplot è perfetto per dataset grandi: mostra i decili della distribuzione
    sns.boxenplot(
        data=plot_df,
        x="senses_grouped",
        y="entropy_level",
        order=order,
        palette="crest",  # Una sfumatura elegante dal blu al verde
        ax=ax,
        showfliers=False,  # Nascondiamo i singoli punti outlier per pulizia visiva
    )

    # Aggiungiamo una linea che unisce le MEDIE per guidare l'occhio nel trend
    means = plot_df.groupby("senses_grouped")["entropy_level"].mean().reindex(order)
    plt.plot(
        range(len(order)),
        means,
        color="red",
        marker="o",
        linewidth=2,
        markersize=8,
        label="Mean Entropy",
        alpha=0.7,
    )

    # 3. Estetica da Journal
    ax.set_title(
        "Semantic Ambiguity vs. Lexical Polysemy", fontsize=16, pad=20, weight="bold"
    )
    ax.set_xlabel("Number of Senses (Polysemy Count)", fontsize=13)
    ax.set_ylabel("Entropy Level (Uncertainty)", fontsize=13)

    # Aggiungiamo una griglia leggera solo sull'asse Y
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.grid(False)

    # Annotazione per spiegare il grafico
    ax.text(
        0.5,
        0.05,
        "The width of the boxes indicates the distribution density.",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )

    sns.despine(left=True, bottom=True)

    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("./fig4_sophisticated_entropy.png", format="png")


def plot_refined_research_visual(df, instance_count=100):
    # --- 1. FILTRAGGIO AVANZATO ---
    # Teniamo solo i lemmi dove ALMENO UN senso ha più di una istanza.
    # Se tutti i sensi hanno frequenza = 1, scartiamo (evita il rumore della distribuzione uniforme spuria).
    df_filtered = df[
        df["senses"].apply(lambda x: any(count > 1 for count in x.values()))
    ].copy()
    df_filtered = df_filtered[df_filtered["instance_count"] >= instance_count]

    # Calcoliamo il numero di sensi
    df_filtered["num_senses"] = df_filtered["senses"].apply(lambda x: len(x))

    # --- 2. RAGGRUPPAMENTO PER LEGGIBILITÀ ---
    threshold = 10  # Sopra gli 8 sensi i dati sono pochi, li raggruppiamo
    df_filtered["senses_label"] = df_filtered["num_senses"].apply(
        lambda x: f"{x}" if x < threshold else f"{threshold}+"
    )
    order = [f"{i}" for i in range(2, threshold)] + [f"{threshold}+"]
    # Partiamo da 2 sensi, perché con 1 senso l'entropia è sempre 0.

    # --- 3. CREAZIONE DEL GRAFICO ---
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    sns.set_style(
        "whitegrid", {"axes.grid": False}
    )  # Background pulito senza griglia pesante

    # Violin Plot: mostra la distribuzione della densità (la "pancia" dei dati)
    sns.violinplot(
        data=df_filtered,
        x="senses_label",
        y="entropy_level",
        order=order,
        palette="Blues",
        inner=None,  # Rimuoviamo il box interno per pulizia
        alpha=0.4,  # Rendiamo i violini trasparenti
        linewidth=0,  # Rimuoviamo il bordo dei violini
    )

    # Point Plot: aggiunge la linea della media con gli intervalli di confidenza
    sns.pointplot(
        data=df_filtered,
        x="senses_label",
        y="entropy_level",
        order=order,
        color="#d73027",  # Un rosso accademico per contrasto
        scale=0.8,
        join=True,
        capsize=0.1,
        label="Mean Entropy",
    )

    # --- 4. DETTAGLI DI DESIGN ---
    plt.title(
        "Semantic Entropy vs. Polysemy Degree",
        fontsize=16,
        pad=20,
        loc="center",
        weight="bold",
    )
    plt.xlabel("Number of Senses per Lemma", fontsize=12)
    plt.ylabel("Entropy Level ($H$)", fontsize=12)

    ax.axhline(0.8, color="#d73027", linestyle=":", linewidth=2, alpha=0.8)

    # Annotazione statistica
    # total_removed = len(df) - len(df_filtered)
    # plt.text(0.98, 0.02, f"Note: Excluded {total_removed} sparse entries (all senses frequency = 1)",
    #         transform=plt.gca().transAxes, fontsize=9, style='italic',
    #         ha='right', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    sns.despine(trim=True)  # Stile Tufte: rimuove gli assi inutili

    plt.tight_layout()
    plt.savefig("./journal_entropy_violin.png", format="png", dpi=300)



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
    THRESHOLD = 10  # group senses >= threshold as "10+"
    ENTROPY_THRESH = 0.8  # high-entropy cutoff

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


def check_thresholds(df):
    for n_threshold in [10, 30, 50, 100]:
        # Filtro frequenza
        temp_df = df[df["instance_count"] >= n_threshold]
        # Filtro entropia alta (Hard Cases)
        hard_cases = temp_df[temp_df["entropy_level"] >= 0.8]

        print(f"Soglia n={n_threshold}:")
        print(f"  - Lemmi totali: {len(temp_df)}")
        print(f"  - Hard Cases (H >= 0.8): {len(hard_cases)}")
        print("-" * 30)


# ── Shared palette — identical to plot_polysemy.py ────────────────────
COLOR_MAX = "#e31a1c"  # red   — max noise (worst case)
COLOR_MEAN = "#1f78b4"  # blue  — mean noise
COLOR_ELBOW = "#333333"  # dark  — elbow vertical line
COLOR_GRID = "gray"
COLOR_SPINE = "#bbbbbb"
COLOR_TICK = "#bbbbbb"
THRESHOLD_LIM = 40  # x-axis range for elbow analysis


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


def analyze_noise_ratio(df):

    # Rimuoviamo frequenze 0 per evitare divisioni per zero
    df = df[df["instance_count"] > 0].copy()
    # df = df[df['senses'].apply(lambda x: any(v > 1 for v in x.values()))].copy()

    # 2. Calcolo della metrica di rumore
    df["noise_ratio"] = df["entropy_level"] / df["instance_count"]

    # 3. Analisi del decadimento per trovare il "Gomito"
    # Raggruppiamo per instance_count e calcoliamo il noise_ratio medio e massimo
    # (Ci concentriamo sui count fino a 50, dove avviene la transizione)
    threshold_limit = 40
    df_trend = df[df["instance_count"] <= threshold_limit].copy()

    trend_stats = (
        df_trend.groupby("instance_count")["noise_ratio"]
        .agg(["mean", "max"])
        .reset_index()
    )

    # 4. Visualizzazione da Journal (Elbow Method)
    # plt.rcParams.update({"font.family": "serif", "font.size": 12})
    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Plottiamo il valore MASSIMO del rumore per ogni frequenza (il worst-case scenario)
    sns.lineplot(
        data=trend_stats,
        x="instance_count",
        y="max",
        marker="o",
        color="#d73027",
        linewidth=2,
        label="Max Noise Ratio (Worst Case)",
        ax=ax,
    )

    # Plottiamo la MEDIA del rumore
    sns.lineplot(
        data=trend_stats,
        x="instance_count",
        y="mean",
        marker="s",
        color="#4575b4",
        linewidth=2,
        label="Mean Noise Ratio",
        ax=ax,
    )

    # Troviamo approssimativamente il gomito calcolando la derivata prima discreta (differenza)
    # Quando la differenza tra un punto e il successivo diventa trascurabile (< 0.01), il rumore si è stabilizzato
    # Verifichiamo dove la derivata resta sotto soglia per un po'

    # 1. Smussiamo la curva per evitare che il rumore casuale mandi fuori giri la derivata seconda
    trend_stats["max_smothed"] = (
        trend_stats["max"]
        .rolling(window=3, center=True)
        .mean()
        .fillna(method="bfill")
        .fillna(method="ffill")
    )

    # 2. Calcoliamo la derivata seconda (accelerazione della curva)
    trend_stats["diff2"] = trend_stats["max_smothed"].diff().diff().abs()

    # 3. Definiamo la stabilità: la derivata seconda deve essere trascurabile
    # 0.01 è un buon valore, ma verifica che non sia troppo stringente per i tuoi dati
    trend_stats["is_stable"] = trend_stats["diff2"] < 0.01

    # 4. Cerchiamo la persistenza (3 punti di fila)
    stable_points = trend_stats["is_stable"].rolling(window=3).sum()
    elbow_point = trend_stats[stable_points == 3]["instance_count"].min()

    print(f"Il segnale si stabilizza a N = {elbow_point}")

    if pd.notna(elbow_point):
        ax.axvline(elbow_point, color="black", linestyle="--", alpha=0.7)
        ax.annotate(
            "Mathematical Elbow (Noise stabilizes)",
            xy=(elbow_point, 0.1),
            xytext=(elbow_point + 5, 0.2),
            arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6),
        )

    # ax.set_title("Entropy-to-Frequency Ratio: Identifying the Noise Threshold", fontsize=12, pad=15, weight='bold')
    ax.set_xlabel("Instance Count ($N$)", fontsize=12)
    ax.set_ylabel(r"Noise Ratio ($\frac{H}{N}$)", fontsize=12)

    sns.despine(trim=True)
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig("./noise_ratio_elbow.png", format="png", dpi=300)

    return df, elbow_point


def plot_semantic_stability_analysis(df, min_instances=0):
    """
    Analizza la stabilità dell'entropia semantica in base alla polisemia.
    Identifica il punto in cui l'incertezza statistica (rumore) decade
    per lasciare spazio al segnale semantico.
    """

    # --- 1. PRE-PROCESSING E NORMALIZZAZIONE ---
    # Filtro: Rimuoviamo i casi con troppe poche istanze per essere significativi
    df_clean = df.copy()

    # Calcolo numero di sensi (k)
    df_clean["n_senses"] = df_clean["senses"].apply(len)

    # IMPORTANTE: L'entropia deve essere normalizzata sul numero di sensi (log2(k))
    # Se la tua 'entropy_level' non lo è già, abilita questa riga:
    # df_clean['h_norm'] = df_clean['entropy_level'] / np.log2(df_clean['n_senses'])
    df_clean["h_norm"] = df_clean["entropy_level"]

    # Applichiamo il filtro sulle istanze basato sul tuo "gomito"
    df_clean = df_clean[df_clean["instance_count"] >= min_instances]

    # Raggruppamento coda lunga (8+)
    limit = 8
    df_clean["group"] = df_clean["n_senses"].apply(
        lambda x: str(x) if x < limit else f"{limit}+"
    )
    order = [str(i) for i in range(2, limit)] + [f"{limit}+"]

    # --- 2. CONFIGURAZIONE ESTETICA (Journal Quality) ---
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.labelweight": "bold",
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # --- 3. LAYERED ARCHITECTURE ---

    # Palette scientifica
    palette = "viridis"

    # Layer 1: Strip Plot (Densità dei dati grezzi)
    sns.stripplot(
        data=df_clean,
        x="group",
        y="h_norm",
        order=order,
        palette=palette,
        alpha=0.1,
        jitter=0.3,
        size=2,
        ax=ax,
        zorder=1,
    )

    # Layer 2: Violin Plot (Distribuzione semantica - metà violino per pulizia)
    sns.violinplot(
        data=df_clean,
        x="group",
        y="h_norm",
        order=order,
        palette=palette,
        bw_adjust=0.5,
        cut=0,
        inner=None,
        alpha=0.4,
        ax=ax,
        zorder=2,
    )

    # Layer 3: Point Plot (Trend della Media con Confidenza 95%)
    sns.pointplot(
        data=df_clean,
        x="group",
        y="h_norm",
        order=order,
        color="#2c3e50",
        scale=0.8,
        markers="D",
        capsize=0.1,
        linestyles="--",
        errwidth=2,
        ax=ax,
        label="Mean $H_{norm}$",
    )

    # --- 4. LINEE DI RIFERIMENTO ---
    # Soglia critica di alta ambiguità (0.9)
    ax.axhline(0.9, color="#d73027", linestyle="--", linewidth=1.5, alpha=0.6)

    # --- 5. RIFINITURE E LABELS ---
    ax.set_title(
        f"Semantic Entropy Stability (Min Instances: {min_instances})",
        fontsize=16,
        pad=20,
    )
    ax.set_xlabel("Polysemy Degree (Number of Senses)", fontsize=13, labelpad=12)
    ax.set_ylabel("Normalized Shannon Entropy ($H_{norm}$)", fontsize=13, labelpad=12)

    # Limiti asse Y (l'entropia normalizzata è 0-1)
    ax.set_ylim(-0.05, 1.05)

    # Pulizia assi (Tufte Style)
    sns.despine(offset=10, trim=True)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)

    # Annotazione tecnica sul filtraggio
    plt.annotate(
        f"Confidence Threshold: N ≥ {min_instances}\nSignal-to-Noise: Optimized",
        xy=(0.02, 0.93),
        xycoords="axes fraction",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig("./semantic_stability_analysis.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    JSON_FILE = "semcor_en_MERGE_test-en_MERGE_wngt_glosses_en.data_ENTROPY_LEVEL.json"

    df = load_and_prep_data(JSON_FILE)

    # plot_entropy_distribution(df, instance_count=10)

    # plot_frequency_vs_entropy(df, instance_count=10)

    # plot_top_words_complexity(df)

    # plot_senses_vs_entropy_trend(df)

    # plot_sophisticated_entropy_analysis(df)

    plot_refined_research_visual_journal(df, instance_count=10)

    plot_entropy_threshold_analysis(df, instance_count=10)

    # plot_final_journal_entropy(df)

    # check_thresholds(df)

    analyze_noise_ratio_journal(df)

    # plot_semantic_stability_analysis(df)
