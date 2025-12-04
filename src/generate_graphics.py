import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import squarify

sns.set_theme(style="whitegrid")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "../figures/gerais")
PILOT_DIR = os.path.join(FIG_DIR, "../pilotos")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(PILOT_DIR, exist_ok=True)

df = pd.read_csv("data/processed/driver_emotions.csv")


emotion_groups = {
    "positive": ["admiration","joy","gratitude","approval","caring","optimism","pride","love"],
    "negative": ["annoyance","anger","fear","disapproval","disgust","remorse","sadness","disappointment","embarrassment"],
    "informational": ["neutral","curiosity","confusion","surprise"]
}

def group_emotion(emotion):
    for group, items in emotion_groups.items():
        if emotion in items:
            return group
    return "other"

df["emotion_group"] = df["emotion"].apply(group_emotion)
df_no_neutral = df[df["emotion"] != "neutral"]


plt.figure(figsize=(12,8))
df["emotion"].value_counts().sort_values().plot(kind="barh", color="skyblue")
plt.title("Distribuição das emoções (todas)")
plt.xlabel("Quantidade")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "01_distribuicao_emocoes_barplot.png"), dpi=300)
plt.close()

plt.figure(figsize=(8,5))
df["emotion_group"].value_counts().plot(kind="bar", color=["#5DADE2","#E74C3C","#58D68D"])
plt.title("Distribuição de emoções agrupadas")
plt.xlabel("Grupo emocional")
plt.ylabel("Quantidade")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "02_emocao_agrupada_barplot.png"), dpi=300)
plt.close()

counts = df["emotion"].value_counts()
plt.figure(figsize=(12,7))
squarify.plot(sizes=counts.values, label=counts.index, alpha=0.8)
plt.title("Treemap das emoções")
plt.axis("off")
plt.savefig(os.path.join(FIG_DIR, "03_treemap_emocoes.png"), dpi=300)
plt.close()

emotion_counts = df["emotion"].value_counts()
valid_emotions = emotion_counts[emotion_counts >= 20].index
df_reduced = df[df["emotion"].isin(valid_emotions)]

pivot = df_reduced.pivot_table(
    index="driver_name",
    columns="emotion",
    values="score",
    aggfunc="mean"
)

plt.figure(figsize=(14,7))
sns.heatmap(pivot, annot=True, cmap="Blues")
plt.title("Heatmap (somente emoções frequentes)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "04_heatmap_emocoes_frequentes.png"), dpi=300)
plt.close()

plt.figure(figsize=(12,6))
sns.stripplot(data=df_no_neutral, x="emotion", y="score", jitter=True, alpha=0.5)
plt.xticks(rotation=45)
plt.title("Distribuição de score por emoção (sem neutral)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "05_stripplot_score_emocao.png"), dpi=300)
plt.close()

g = sns.catplot(
    data=df_no_neutral,
    x="emotion_group",
    kind="count",
    col="driver_name",
    col_wrap=4,
    height=3,
    sharex=False,
    sharey=False
)
g.set_xticklabels(rotation=45)
plt.savefig(os.path.join(FIG_DIR, "06_facetgrid_emocoes_por_piloto.png"), dpi=300)
plt.close()

pivot_groups = df_no_neutral.pivot_table(
    index="driver_name",
    columns="emotion_group",
    values="score",
    aggfunc="count",
    fill_value=0
)

pivot_groups.plot(kind="bar", stacked=True, figsize=(12,6),
                  color=["#5DADE2","#E74C3C","#58D68D"])
plt.title("Emoções agrupadas por piloto (stacked)")
plt.xlabel("Piloto")
plt.ylabel("Quantidade de falas")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "07_stackplot_piloto_emocao.png"), dpi=300)
plt.close()


pilotos = df["driver_name"].unique()

for piloto in pilotos:
    subdir = os.path.join(PILOT_DIR, piloto.replace(" ", "_"))
    os.makedirs(subdir, exist_ok=True)

    df_p = df[df["driver_name"] == piloto]

    plt.figure(figsize=(10,5))
    df_p["emotion"].value_counts().plot(kind="bar", color="skyblue")
    plt.title(f"Distribuição das emoções - {piloto}")
    plt.xlabel("Emoção")
    plt.ylabel("Quantidade")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(subdir, f"{piloto}_distribuicao_emocoes.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8,5))
    df_p["emotion_group"].value_counts().plot(kind="bar", color=["#5DADE2","#E74C3C","#58D68D"])
    plt.title(f"Grupos de emoções - {piloto}")
    plt.xlabel("Grupo emocional")
    plt.ylabel("Quantidade")
    plt.tight_layout()
    plt.savefig(os.path.join(subdir, f"{piloto}_emocoes_agrupadas.png"), dpi=300)
    plt.close()


    plt.figure(figsize=(10,5))
    sns.boxplot(data=df_p, x="emotion", y="score")
    plt.xticks(rotation=45)
    plt.title(f"Distribuição de score das emoções - {piloto}")
    plt.tight_layout()
    plt.savefig(os.path.join(subdir, f"{piloto}_scores_emocoes.png"), dpi=300)
    plt.close()

