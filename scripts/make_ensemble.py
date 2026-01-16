import pandas as pd
import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best_valence_csv", required=True, help="CSV Valență (Chained)")
    parser.add_argument("--best_arousal_csv", required=True, help="CSV Arousal (Specialist)")
    parser.add_argument("--eval_repo", required=True, help="Repo evaluare")
    parser.add_argument("--out_csv", default="submission_GRAND_FINAL.csv")
    args = parser.parse_args()

    print("="*60)
    print("🧠  SMART ENSEMBLE MERGER (Inner Join on ID)")
    print("="*60)

    print(f"Loading Valence: {args.best_valence_csv}")
    df_v = pd.read_csv(args.best_valence_csv)
    print(f" -> Linii: {len(df_v)}")

    print(f"Loading Arousal: {args.best_arousal_csv}")
    df_a = pd.read_csv(args.best_arousal_csv)
    print(f" -> Linii: {len(df_a)}")


    if "text_id" not in df_v.columns or "text_id" not in df_a.columns:
        print(" EROARE: Nu pot face Smart Merge fara coloana 'text_id' în ambele fisiere!")
        sys.exit(1)


    df_a_clean = df_a[["text_id", "pred_delta_arousal"]].copy()
    df_a_clean = df_a_clean.rename(columns={"pred_delta_arousal": "arousal_specialist"})


    print("\nExecutare Inner Join pe 'text_id'...")
    df_final = pd.merge(df_v, df_a_clean, on="text_id", how="inner")
    
    print(f" -> Linii comune gqsite: {len(df_final)}")
    
    if len(df_final) == 0:
        print(" EROARE: Nu există niciun ID comun între cele două fiwiere! Verifica datele.")
        sys.exit(1)


    df_final["pred_delta_arousal"] = df_final["arousal_specialist"]

    df_final = df_final.drop(columns=["arousal_specialist"])


    df_final.to_csv(args.out_csv, index=False)
    print(f"✅ Fișier salvat: {args.out_csv}")

    sys.path.append(args.eval_repo)
    try:
        import eval as official_eval
        print("\n" + "="*40)
        print("🏆 REZULTATE FINALE (PE INTERSECȚIE)")
        print("="*40)
        
        ids = df_final["text_id"].astype(str).tolist()

        res_v = official_eval.task2_correlation(
            ids, 
            df_final["pred_delta_valence"].tolist(), 
            df_final["gold_delta_valence"].tolist()
        )
        print(f"Valence: {res_v['r']:.4f}")

        res_a = official_eval.task2_correlation(
            ids, 
            df_final["pred_delta_arousal"].tolist(), 
            df_final["gold_delta_arousal"].tolist()
        )
        print(f"Arousal: {res_a['r']:.4f}")
        
        mean_r = (res_v['r'] + res_a['r']) / 2
        print("-" * 40)
        print(f" MEDIA FINALA: {mean_r:.4f}")
        print("-" * 40)
        
    except ImportError:
        print("Nu am putut rula evaluarea.")

if __name__ == "__main__":
    main()