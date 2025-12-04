import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")

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

stress = ["annoyance", "confusion", "fear"]
df["is_stress"] = df["emotion"].isin(stress)

OUT = "figures/emotions"
os.makedirs(OUT, exist_ok=True)

neg_count = (df["emotion_group"] == "negative").groupby(df["driver_name"]).mean().sort_values()

plt.figure(figsize=(10,6))
neg_count.plot(kind="barh", color="red")
plt.xlabel("Proporção de falas negativas")
plt.title("Ranking: Pilotos que mais reclamam")
plt.tight_layout()
plt.savefig(f"{OUT}/01_pilotos_que_reclamam.png", dpi=300)
plt.close()

pos_count = (df["emotion_group"] == "positive").groupby(df["driver_name"]).mean().sort_values()

plt.figure(figsize=(10,6))
pos_count.plot(kind="barh", color="green")
plt.xlabel("Proporção de falas positivas")
plt.title("Ranking: Pilotos mais positivos")
plt.tight_layout()
plt.savefig(f"{OUT}/02_pilotos_positivos.png", dpi=300)
plt.close()

stress_count = df.groupby("driver_name")["is_stress"].mean().sort_values()

plt.figure(figsize=(10,6))
stress_count.plot(kind="barh", color="orange")
plt.xlabel("Índice de estresse")
plt.title("Pilotos mais estressados / reativos")
plt.tight_layout()
plt.savefig(f"{OUT}/03_indice_estresse.png", dpi=300)
plt.close()

var_count = df.groupby("driver_name")["emotion"].nunique().sort_values()

plt.figure(figsize=(10,6))
var_count.plot(kind="barh", cmap="viridis")
plt.xlabel("Número de emoções diferentes detectadas")
plt.title("Diversidade emocional por piloto")
plt.tight_layout()
plt.savefig(f"{OUT}/04_diversidade_emocional.png", dpi=300)
plt.close()
