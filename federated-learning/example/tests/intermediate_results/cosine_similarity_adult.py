import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from typing import Dict
from functools import reduce

sys.path.append(os.path.expanduser("~/federated-learning"))

from pathlib import Path
from federated.feddata import ACSDataStatesBySize, ACSDataStatesCode
from folktables import ACSDataSource, ACSIncome
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity



class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
        self.freq_maps = {}

    def fit(self, X, y=None):
        for col in self.columns:
            freqs = X[col].value_counts(normalize=True)
            self.freq_maps[col] = freqs
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            X_copy[col] = X_copy[col].map(self.freq_maps[col]).fillna(0)
        return X_copy


def clean_categorical_column(col):
    return col.apply(lambda c: str(int(float(c))) if pd.notnull(c) else c)


def load_adult(list_states: list[str] = None, year: str = '2021', horizon: str = '1-Year', split: bool = True):

    """Load the ACS dataset."""
    # Download if necessary ACS data source for each state
    data_source = ACSDataSource(survey_year=year, horizon=horizon, survey='person')

    if list_states is not None:
        acs_data = data_source.get_data(states=list_states, download=False)
    else:
        acs_data = data_source.get_data(download=False)
    
    acs_data.rename(columns={"RELSHIPP": "RELP"}, inplace=True)

    # Preprocessing of ACS data source for the task "income > 50k"
    seed = 42
    local_acs_data = {}
    x_generalization_test = pd.DataFrame()
    y_generalization_test = pd.DataFrame()
    ACSIncome._features.append('PINCP')
    for state_code in list_states:
        state_data = acs_data.loc[acs_data["ST"] == ACSDataStatesCode[state_code].value]

        df_features, df_target, _ = ACSIncome.df_to_pandas(state_data)
        #state_income = df_features[['PINCP']]
        #df_features.drop('PINCP', axis=1, inplace=True)

        if split:
            x_train, x_global_test, y_train, y_global_test = train_test_split(df_features, df_target, test_size=0.1,
                                                                              random_state=seed)
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=seed)
            x_generalization_test = pd.concat([x_generalization_test, x_global_test])
            y_generalization_test = pd.concat([y_generalization_test, y_global_test])
            local_acs_data[state_code] = {"x_train": x_train, "x_val": x_val, "y_train": y_train, "y_val": y_val}
        else:
            local_acs_data[state_code] = pd.concat([df_features, df_target, state_income])


    return local_acs_data


clients_number = 50
clients_name = [state.name for state in ACSDataStatesBySize][:clients_number]
data = load_adult(list_states=clients_name)

# === 1. Feature specification
#categorical_features = ['COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'SEX', 'RAC1P']

ordinal_features = ['SCHL']
frequency_features = ['POBP', 'OCCP']
one_hot_features = ['MAR', 'RAC1P', 'COW', 'RELP']
numerical_features = ['AGEP', 'WKHP', 'PINCP']
other_features = ['SEX']

#data[other_features] = data[other_features].astype(int)

all_features = numerical_features + ordinal_features + frequency_features + one_hot_features + other_features #categorical_features

# === 2. Results path
results_dir = "/home/dudasilva/federated-learning/example/tests/results/tests_complete_adult/fl_without_groups/descriptive_analysis"
os.makedirs(results_dir, exist_ok=True)

# === 3. Dictionary to store results
state_similarity_scores = {}
all_similarities_df = []


"""
for v in data.values():
    for col in ordinal_features + frequency_features + one_hot_features:
        v["x_train"][col] = clean_categorical_column(v["x_train"][col])

        if col == "SEX":
           v["x_train"][col] = v["x_train"][col].map({'1': '0', '2': '1'})

        else:
        v["x_train"][col] = v["x_train"][col].astype(str)
"""
        


# === 4. Iterate over each state
for state_name, state_data in data.items():

    print(f"Processing {state_name}...")

    # 4.1 Current state's data
    X_state = state_data["x_train"][all_features].copy()

    X_state = X_state.loc[:, ~X_state.columns.duplicated()]

    # 4.2 Remaining states' data
    X_rest = pd.concat([
        v["x_train"][all_features]
        for k, v in data.items() if k != state_name
    ])

    X_rest = X_rest.loc[:, ~X_rest.columns.duplicated()]

    n_na_state = X_state.isna().sum().sum()

    # Remove rows with missing values from both state and rest
    X_state = X_state.dropna()
    X_rest = X_rest.dropna()

    """
    # 4.3 Preprocessing: scale numerical and one-hot encode categorical
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ])
    """

    # 4.3 Alter Preprocessing: scale numerical, one-hot, label and ordinal encode for categorical
    preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("ord", OrdinalEncoder(), ordinal_features),
    ("frequency", FrequencyEncoder(columns=frequency_features), frequency_features),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output = False), one_hot_features),
    ("dummy_sex", "passthrough", other_features),
    ])

    preprocessor.fit(X_rest)
    X_state_trans = preprocessor.transform(X_state)
    X_rest_trans = preprocessor.transform(X_rest)


    # 4.4 Apply PCA to reduce dimensionality (keep 95% variance)
    #pca = PCA(n_components=0.95)
    X_rest_pca = X_rest_trans #pca.fit_transform(X_rest_trans.toarray())
    X_state_pca = X_state_trans #pca.transform(X_state_trans.toarray())

    # 4.5 Compute mean vector of other states
    X_rest_mean = X_rest_pca.mean(axis=0).reshape(1, -1)

    print(f"Cosine similarity calculation {state_name}...")

    # 4.6 Compute cosine similarity for each row of current state vs rest mean
    similarities = cosine_similarity(X_state_pca, X_rest_mean).flatten()

    """
    # 4.7 Store aggregate results
    state_similarity_scores[state_name] = {
        "min": np.min(similarities),
        "q1": np.percentile(similarities, 25),
        "median": np.median(similarities),
        "q3": np.percentile(similarities, 75),
        "max": np.max(similarities),
        "mean": np.mean(similarities),
        "std": np.std(similarities),
        "n_NA_state": n_na_state,
        "n_obs": X_state.shape[0]
    }
    """

    # 4.8 Save individual similarity scores for boxplot
    all_similarities_df.append(
        pd.DataFrame({"similarity": similarities, "state": state_name, 'n_obs': X_state.shape[0]})
    )

# === 5. Combine similarity data
similarity_df = pd.concat(all_similarities_df, ignore_index=True)


"""
# === 5.2. Inicializar dicionários para cada limiar
subset_states_011 = {}
subset_states_08 = {}

# === 5.3. Calcular percentagens por estado
for state in similarity_df["state"].unique():
    state_df = similarity_df[similarity_df["state"] == state]
    
    pct_011 = (state_df["similarity"] > 0.11).mean()
    pct_08 = (state_df["similarity"] > 0.8).mean()
    
    subset_states_011[state] = {"x_pct": pct_011}
    subset_states_08[state] = {"x_pct": pct_08}

total_diff = sum(
    abs(subset_states_011[st]["x_pct"] - subset_states_08[st]["x_pct"])
    for st in subset_states_011.keys()
)

# === 5.4. Guardar ficheiros
subset_criteria_logs = "subset_criteria_logs"
subset_path = Path("/home/dudasilva/federated-learning/example/tests") / subset_criteria_logs 
subset_path.mkdir(parents=True, exist_ok=True)

filename_011 = "subset_states_cosine_similarity_v2.txt"
filename_08 = "subset_states_cosine_similarity_thr0_8_v2.txt"
"""

"""
with open(subset_path / filename_011, "w") as f:
    for state, info in subset_states_011.items():
        f.write(f"{state}: x_pct={info['x_pct']:.3f}\n")

with open(subset_path / filename_08, "w") as f:
    for state, info in subset_states_08.items():
        f.write(f"{state}: x_pct={info['x_pct']:.3f}\n")
"""


# Contagem de valores de similarity por estado

DATA_PATH_MINE = "/home/dudasilva/federated-learning/example/tests"
state_sizes = os.path.join(DATA_PATH_MINE, "results/tests_complete_adult/fl_without_groups/descriptive_analysis/state_sizes.txt")
data_sizes_dict = {}

with open(state_sizes, 'r') as f:
    for line in f:
        if ':' in line:
            state, rest = line.strip().split(':')
            rest = int(round(float(rest)))
            data_sizes_dict[state.strip()] = rest

ordered_states = sorted(data_sizes_dict, key=data_sizes_dict.get, reverse=True) #by datasize do maior para o menor


# === 6. Boxplot per state for bottom-k cosine similarities (most heterogeneous)

k = 10
bottom_k_per_state = []

for state_name, group in similarity_df.groupby("state"):
    bottom_k = group.nsmallest(k, "similarity")
    bottom_k_per_state.append(bottom_k)

bottom_k_df = pd.concat(bottom_k_per_state)


plt.figure(figsize=(16, 8))
bottom_k_df.boxplot(column="similarity", by="state", rot=90)
plt.suptitle("")
plt.title("")
plt.xlabel("State")
plt.ylabel("Cosine similarity")
plt.tight_layout()
#plt.savefig(os.path.join(results_dir, f"boxplot_bottom_{k}_cosine_similarity_per_state_frequency_encoding_v2.pdf"))
plt.close()



# === 7. Boxplot per state for bottom 1% cosine similarities

#k_pct = max(1, int(0.01 * len(similarity_df)))
#bottom_pct = similarity_df.nsmallest(k_pct, "similarity")

bottom_pct_per_state = []
percentiles = [1, 2.5]

for p in percentiles:
    for state_name, group in similarity_df.groupby("state"):
        k_pct = max(1, int((p/100) * len(group)))  # 1% por estado
        bottom_k_pct = group.nsmallest(k_pct, "similarity")
        bottom_pct_per_state.append(bottom_k_pct)
    
    bottom_pct = pd.concat(bottom_pct_per_state)
    bottom_pct["state"] = pd.Categorical(bottom_pct["state"], categories=ordered_states, ordered=True)
    
    plt.figure(figsize=(30, 8))
    bottom_pct.boxplot(column="similarity", by="state", rot=90)
    label = f"{p}pct"
    label_clean = label.replace(".", "_").replace("pct", "pctile")
    plt.suptitle("")
    plt.title("")
    plt.xlabel("State")
    plt.ylabel(f"Cosine similarity (bottom {p}% per state)")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"boxplot_bottom_{label_clean}_cosine_similarity_per_state_frequency_encoding_v2.pdf"))
    plt.close()


# === 8. Boxplots per state for most heterogeneous samples (cosine similarity below thresholds)

#thresholds = [0, -0.2, -0.5, -0.7, -0.75, -0.8]

similarities = similarity_df["similarity"]
similarities_per_state = similarity_df.groupby("state")["similarity"]

thresholds_info = [
    (np.percentile(similarities, p), f"{p} pct") for p in [1, 2.5]
]

thresholds_info_per_state = {state: [(np.percentile(group, p), f"{p} pct") for p in [1, 2.5]] for state, group in similarities_per_state}

"""
# === 3. Iterar sobre cada limiar e calcular proporções por estado
for threshold, label in thresholds_info:
    threshold = np.round(threshold, 5)
    label_clean = label.replace(".", "_").replace("pct", "pctile")
    threshold_str = f"{threshold:.5f}".replace(".", "_")  # ex: 0.002 → "0_002"
    filename = f"subset_states_cosine_similarity_{label_clean}_thr_{threshold_str}.txt"
    
    subset_states = {}
    for state in similarity_df["state"].unique():
        state_df = similarity_df[similarity_df["state"] == state]
        pct_x = (state_df["similarity"] > threshold).mean()
        
        # Procurar limiar local por estado

        state_thresholds = thresholds_info_per_state.get(state, [])
        
        state_thresholds = thresholds_info_per_state.get(state, [])
        st_thrs = next((val for val, lbl in state_thresholds if lbl == label), None)

        subset_states[state] = {
            "x_pct": pct_x,
            "state_threshold": round(st_thrs, 5) if st_thrs is not None else None
        }
    
    
    # === 4. Guardar ficheiro
    with open(subset_path / filename, "w") as f:
         for state, info in subset_states.items():
             f.write(f"{state}: x_pct={info['x_pct']:.3f}, state_threshold={info['state_threshold']:.5f}\n")
"""

# Limite inferior do boxplot
q1 = np.percentile(similarities, 25)
q3 = np.percentile(similarities, 75)
iqr = q3 - q1
lower_limit = q1 - 1.5 * iqr
#thresholds_info.append((lower_limit, "lower_iqr"))

# Total points per state (for percentage calculation)
total_counts = similarity_df["state"].value_counts().sort_index().reset_index()
total_counts.columns = ["state", "total"]
summary_list = []

for threshold, label in thresholds_info:
    threshold = np.round(threshold, 5)
    filtered_df = similarity_df[similarity_df["similarity"] < threshold]
    filtered_df["state"] = pd.Categorical(filtered_df["state"], categories=ordered_states, ordered=True)

    if not filtered_df.empty:
        # Boxplot by state
        plt.figure(figsize=(30, 8))
        filtered_df.boxplot(column="similarity", by="state", rot=90)
        plt.suptitle("")
        plt.title("")
        plt.xlabel("State")
        label_percentile = label.replace(" pct", " percentile")
        plt.ylabel(f"Cosine similarity below global {label_percentile}")
        plt.tight_layout()
        label_clean = label.replace("th percentile", " pct")
        threshold_str = f"{threshold:.5f}".replace(".", "_")

      
        plt.savefig(os.path.join(results_dir, f"boxplot_cosine_similarity_below_{label_clean}_per_state_frequency_encoding_v2.pdf"))
        plt.close()

        # Count and percentage per state
        counts = filtered_df["state"].value_counts().sort_index().reset_index()
        counts.columns = ["state", f"n_below_{label}"]
        merged = pd.merge(total_counts, counts, on="state", how="left").fillna(0)
    

        
        col_n = f"n_below_{label_clean}_thr_{threshold_str}"
        col_pct = f"pct_below_{label_clean}_thr_{threshold_str}"
        merged[col_n] = merged[f"n_below_{label}"].astype(int)
        merged[col_pct] = 100 * merged[f"n_below_{label}"] / merged["total"]
        summary_list.append(merged[["state", col_n, col_pct]])
        

        # Scatterplot: % below threshold vs. dataset size
        plt.figure(figsize=(8, 6))
        plt.scatter(merged["total"], merged[col_pct])
        plt.xlabel("Dataset size")
        plt.ylabel(f"Percentage below global {label_percentile}")
        plt.title("")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"scatter_pct_below_{label_clean}_vs_data_size_frequency_encoding_v2.pdf"))
        plt.close()

# === 9. Save summary CSV with count and percentage below each cosine similarity threshold

"""
summary_bool = True
if summary_bool:
    summary_df = reduce(lambda left, right: pd.merge(left, right, on="state", how="outer"), summary_list)
    summary_df = pd.merge(total_counts, summary_df, on="state", how="left")
    summary_df = summary_df.fillna(0)
    for col in summary_df.columns:
        if col.startswith("n_below_"):
            summary_df[col] = summary_df[col].astype(int)
        elif col.startswith("pct_below_"):
            summary_df[col] = summary_df[col].round(2)
    summary_df.to_csv(os.path.join(results_dir, "cosine_similarity_below_threshold_summary_frequency_encoding_v3.csv"), index=False)
"""

"""
# === 6. Save summary statistics to CSV
summary_df = pd.DataFrame.from_dict(state_similarity_scores, orient='index')
summary_df.index.name = "state"
summary_df.reset_index(inplace=True)
summary_df.to_csv(os.path.join(results_dir, "cosine_similarity_summary.csv"), index=False)
"""

"""
# === 7. Boxplot per state
plt.figure(figsize=(16, 6))
similarity_df.boxplot(column='similarity', by='state', rot=90)
plt.suptitle("")
plt.title("")
plt.xlabel("State")
plt.ylabel("Cosine similarity")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "boxplot_cosine_similarity_by_state.png"))
plt.close()
"""

# === 8. Global cumulative similarity plot (all states combined)
plt.figure(figsize=(10, 6))

# Sort all similarity values from all states
all_similarities = np.sort(similarity_df["similarity"].values)
cumulative_proportion = np.arange(1, len(all_similarities) + 1) / len(all_similarities)
cumulative_proportion = cumulative_proportion * 100


# Plot single cumulative curve
plt.plot(all_similarities, cumulative_proportion, color="blue", linewidth=2)

plt.xlabel("Cosine similarity")
plt.ylabel("Cumulative percentage")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
#plt.savefig(os.path.join(results_dir, "global_cumulative_cosine_similarity_frequency_encoding_v2.pdf"))
plt.close()

"""

# === 9. Cumulative similarity plot
plt.figure(figsize=(10, 6))
for state, group in similarity_df.groupby("state"):
    sorted_vals = np.sort(group["similarity"].values)
    cum_vals = np.arange(1, len(sorted_vals)+1) / len(sorted_vals)
    plt.plot(sorted_vals, cum_vals, label=state, alpha=0.3)

plt.title("Cumulative Cosine Similarity per State")
plt.xlabel("Cosine Similarity")
plt.ylabel("Cumulative Proportion")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "cumulative_cosine_similarity_by_state.png"))
plt.close()
"""
