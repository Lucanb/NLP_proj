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

    # 1. Încărcare
    print(f"Loading Valence: {args.best_valence_csv}")
    df_v = pd.read_csv(args.best_valence_csv)
    print(f" -> Linii: {len(df_v)}")

    print(f"Loading Arousal: {args.best_arousal_csv}")
    df_a = pd.read_csv(args.best_arousal_csv)
    print(f" -> Linii: {len(df_a)}")

    # 2. Pregătire pentru Merge
    # Ne asigurăm că avem text_id (cheia comună)
    if "text_id" not in df_v.columns or "text_id" not in df_a.columns:
        print("❌ EROARE: Nu pot face Smart Merge fără coloana 'text_id' în ambele fișiere!")
        sys.exit(1)

    # Păstrăm din df_a doar ce ne trebuie: ID-ul și Predicția de Arousal
    # Redenumim coloana ca să nu se bată cap în cap la merge
    df_a_clean = df_a[["text_id", "pred_delta_arousal"]].copy()
    df_a_clean = df_a_clean.rename(columns={"pred_delta_arousal": "arousal_specialist"})

    # 3. MERGE (Intersecția)
    # Păstrăm doar rândurile care există în AMBELE seturi de validare
    print("\nExecutare Inner Join pe 'text_id'...")
    df_final = pd.merge(df_v, df_a_clean, on="text_id", how="inner")
    
    print(f" -> Linii comune găsite: {len(df_final)}")
    
    if len(df_final) == 0:
        print("❌ EROARE: Nu există niciun ID comun între cele două fișiere! Verifică datele.")
        sys.exit(1)

    # 4. Construcția Coloanelor Finale
    # Valence vine din df_v (deja existent ca pred_delta_valence)
    # Arousal vine din coloana redenumită 'arousal_specialist'
    df_final["pred_delta_arousal"] = df_final["arousal_specialist"]
    
    # Curățăm coloana ajutătoare
    df_final = df_final.drop(columns=["arousal_specialist"])

    # 5. Salvare
    df_final.to_csv(args.out_csv, index=False)
    print(f"✅ Fișier salvat: {args.out_csv}")

    # 6. Evaluare
    sys.path.append(args.eval_repo)
    try:
        import eval as official_eval
        print("\n" + "="*40)
        print("🏆 REZULTATE FINALE (PE INTERSECȚIE)")
        print("="*40)
        
        ids = df_final["text_id"].astype(str).tolist()
        
        # Valence (Sursa: Chained SOTA)
        res_v = official_eval.task2_correlation(
            ids, 
            df_final["pred_delta_valence"].tolist(), 
            df_final["gold_delta_valence"].tolist()
        )
        print(f"Valence: {res_v['r']:.4f}")

        # Arousal (Sursa: Specialist)
        res_a = official_eval.task2_correlation(
            ids, 
            df_final["pred_delta_arousal"].tolist(), 
            df_final["gold_delta_arousal"].tolist()
        )
        print(f"Arousal: {res_a['r']:.4f}")
        
        mean_r = (res_v['r'] + res_a['r']) / 2
        print("-" * 40)
        print(f"⭐ MEDIA FINALĂ: {mean_r:.4f}")
        print("-" * 40)
        
    except ImportError:
        print("Nu am putut rula evaluarea.")

if __name__ == "__main__":
    main()