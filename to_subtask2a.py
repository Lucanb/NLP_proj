import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("submission_task2a_FINAL.csv")

cols = [
    "pred_dispo_change_valence",
    "pred_dispo_change_arousal"
]

print(df[cols].describe())

for c in cols:
    plt.figure()
    df[c].hist(bins=50)
    plt.title(c)
    plt.grid(True)
    plt.show()
