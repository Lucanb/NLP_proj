import pandas as pd
import json
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Fișierul CSV complet (val_predictions_FULL.csv)")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    
    # Luăm toți userii unici
    users = sorted(df["user_id"].unique().tolist())
    
    # Amestecăm o singură dată, cu seed fix
    np.random.seed(42)
    np.random.shuffle(users)
    
    # 80% Train, 20% Validation
    cut = int(0.8 * len(users))
    train_users = users[:cut]
    val_users = users[cut:]
    
    split_data = {
        "train": train_users,
        "val": val_users
    }
    
    with open("fixed_split.json", "w") as f:
        json.dump(split_data, f)
        
    print(f"✅ Split salvat în fixed_split.json")
    print(f"Train Users: {len(train_users)}")
    print(f"Val Users: {len(val_users)}")

if __name__ == "__main__":
    main()