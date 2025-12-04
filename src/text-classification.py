import pandas as pd
from transformers import pipeline

df = pd.read_csv("../data/raw/transcricoes_radio.csv")

df["transcription"] = df["transcription"].fillna("").astype(str)
df["driver_name"] = df["driver_name"].fillna("Unknown")

clf = pipeline(
    "text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=None
)

def classify(text):
    preds = clf(text)[0]
    if len(preds) == 0:
        return ("neutral", 0.0)

    # emoção dominante
    best = max(preds, key=lambda x: x["score"])
    emotion = best["label"]
    score = round(float(best["score"]), 4)

    return (emotion, score)


df[["emotion", "score"]] = df["transcription"].apply(
    lambda t: pd.Series(classify(t))
)

df_out = df[["driver_name", "emotion", "score", "transcription"]]
df_out.to_csv("driver_emotions.csv", index=False)

print("Processo Finalizado.")