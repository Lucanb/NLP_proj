import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_data", required=True, help="Path to original train_subtask1.csv")
    parser.add_argument("--pred_data", required=True, help="Path to output from Task 1 (val_predictions...)")
    parser.add_argument("--output", required=True, help="Path to save the fixed CSV")
    args = parser.parse_args()

    print(f"Reading original data: {args.original_data}")
    df_orig = pd.read_csv(args.original_data)
    
    # Asigură-te că avem text_id (dacă nu există, îl creăm cum am făcut la antrenare)
    if "text_id" not in df_orig.columns:
        if "id" in df_orig.columns:
            df_orig["text_id"] = df_orig["id"]
        else:
            # Reconstruim ID-urile generate (item_0, item_1...)
            # ATENȚIE: Asta merge doar dacă ordinea e identică. 
            # Mai sigur e să ne bazăm pe un ID existent sau text.
            print("⚠️ ATENȚIE: Nu găsesc text_id în original. Încerc join pe text (poate dura).")
    
    print(f"Reading predictions: {args.pred_data}")
    df_pred = pd.read_csv(args.pred_data)

    # 1. Redenumim coloanele din predicții pentru a fi compatibile cu Task 2
    # Task 1 output: id, true_valence, pred_valence...
    # Task 2 input expected: user_id, timestamp, valence, arousal, pred_valence...
    
    rename_map = {
        "id": "text_id",
        "true_valence": "valence", 
        "true_arousal": "arousal"
    }
    df_pred = df_pred.rename(columns=rename_map)

    # 2. Facem MERGE pentru a aduce User_ID și Timestamp din original
    # Folosim 'text_id' ca cheie comună
    
    # Selectăm din original doar ce ne lipsește
    cols_to_add = ["text_id", "user_id", "timestamp"]
    if "collection_phase" in df_orig.columns:
        cols_to_add.append("collection_phase")
    if "is_words" in df_orig.columns:
        cols_to_add.append("is_words")

    # Verificăm dacă df_orig are text_id, altfel îl creăm din index dacă e consistent
    if "text_id" not in df_orig.columns:
         # Dacă originalul nu are ID, e riscant. Presupunem că join-ul pe text e mai sigur.
         cols_to_add = ["text", "user_id", "timestamp"]
         merge_on = "text"
    else:
         merge_on = "text_id"

    print(f"Merging tables based on '{merge_on}'...")
    
    # Păstrăm doar rândurile care au predicții (INNER JOIN)
    df_final = pd.merge(
        df_pred, 
        df_orig[cols_to_add], 
        on=merge_on, 
        how="inner"
    )

    # Verificare
    print(f"Rows before: {len(df_pred)}, Rows after merge: {len(df_final)}")
    
    if len(df_final) == 0:
        print("❌ EROARE CRITICĂ: Merge-ul a rezultat în 0 rânduri. ID-urile nu se potrivesc.")
        return

    # Salvare
    df_final.to_csv(args.output, index=False)
    print(f"✅ CSV reparat salvat în: {args.output}")
    print("Coloane disponibile:", list(df_final.columns))

if __name__ == "__main__":
    main()