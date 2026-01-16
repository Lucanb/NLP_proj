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
    

    if "text_id" not in df_orig.columns:
        if "id" in df_orig.columns:
            df_orig["text_id"] = df_orig["id"]
        else:

            print(" ATENTIE: Nu gasesc text_id în original. Incerc join pe text (poate dura).")
    
    print(f"Reading predictions: {args.pred_data}")
    df_pred = pd.read_csv(args.pred_data)

    
    rename_map = {
        "id": "text_id",
        "true_valence": "valence", 
        "true_arousal": "arousal"
    }
    df_pred = df_pred.rename(columns=rename_map)


    cols_to_add = ["text_id", "user_id", "timestamp"]
    if "collection_phase" in df_orig.columns:
        cols_to_add.append("collection_phase")
    if "is_words" in df_orig.columns:
        cols_to_add.append("is_words")

    if "text_id" not in df_orig.columns:
         cols_to_add = ["text", "user_id", "timestamp"]
         merge_on = "text"
    else:
         merge_on = "text_id"

    print(f"Merging tables based on '{merge_on}'...")

    df_final = pd.merge(
        df_pred, 
        df_orig[cols_to_add], 
        on=merge_on, 
        how="inner"
    )

    print(f"Rows before: {len(df_pred)}, Rows after merge: {len(df_final)}")
    
    if len(df_final) == 0:
        print("EROARE CRITICA: Merge-ul a rezultat în 0 randuri. ID-urile nu se potrivesc.")
        return

    df_final.to_csv(args.output, index=False)
    print(f"✅ CSV reparat salvat în: {args.output}")
    print("Coloane disponibile:", list(df_final.columns))

if __name__ == "__main__":
    main()