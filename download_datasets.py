import openml
import pandas as pd
import os

os.makedirs("datasets", exist_ok=True)

dataset_configs = [
    (45551, "df_atlas"),
    (46888, "df_sepsis"),
    (46860, "df_support"),
    (46882, "df_jigsaw"),
    (46359, "df_fraud"),
    (41147, "df_albert"),
    (45553, "df_fico"),
    (43582, "df_diabetes"),
    (40498, "df_wine"),
    (1462, "df_banknote")
]


for dataset_id, name in dataset_configs:
    print(f"Downloading {name} (id={dataset_id})...")
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    df = pd.concat([X, y], axis=1)
    df.to_csv(f"datasets/{name}.csv", index=False)
    print(f"  Saved: datasets/{name}.csv  shape={df.shape}")

print("Done.")
