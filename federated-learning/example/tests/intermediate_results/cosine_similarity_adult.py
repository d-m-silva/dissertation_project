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



def load_adult_dummy(list_states: list[str] = None, year: str = '2021', horizon: str = '1-Year', split: bool = True):

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
data = load_adult_dummy(list_states=clients_name)

# === 1. Feature specification

ordinal_features = ['SCHL']
frequency_features = ['POBP', 'OCCP']
one_hot_features = ['MAR', 'RAC1P', 'COW', 'RELP']
numerical_features = ['AGEP', 'WKHP', 'PINCP']
binary_features = ['SEX']


all_features = numerical_features + ordinal_features + frequency_features + one_hot_features + binary_features

# === 2. Results path
RESULTS_DIR = "dissertation_project/federated-learning/example/tests/intermediate_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# === 3. Dictionary to store results
state_similarity_scores = {}
all_similarities_df = []


# === 4. Iterate over each state
for state_name, state_data in data.items():

    print(f"Processing {state_name}...")

    # 4.1 Current state's data
    X_state = state_data["x_train"][all_features].copy()

    X_state = X_state.loc[:, ~X_state.columns.duplicated()]

    X_state[binary_features] = X_state[binary_features].map({'1': 1, '2': 2})

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



    # 4.3 Alter Preprocessing: scale numerical, one-hot, label and ordinal encode for categorical
    preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("ord", OrdinalEncoder(), ordinal_features),
    ("frequency", FrequencyEncoder(columns=frequency_features), frequency_features),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output = False), one_hot_features),
    ("dummy_sex", "passthrough", binary_features),
    ])

    preprocessor.fit(X_rest)
    X_state_trans = preprocessor.transform(X_state)
    X_rest_trans = preprocessor.transform(X_rest)

    # 4.5 Compute mean vector of other states
    X_rest_mean = X_rest_trans.mean(axis=0).reshape(1, -1)

    print(f"Cosine similarity calculation {state_name}...")

    # 4.6 Compute cosine similarity for each row of current state vs rest mean
    similarities = cosine_similarity(X_state_pca, X_rest_mean).flatten()

    # 4.8 Save individual similarity scores for boxplot
    all_similarities_df.append(
        pd.DataFrame({"similarity": similarities, "state": state_name, 'n_obs': X_state.shape[0]})
    )

# === 5. Combine similarity data
similarity_df = pd.concat(all_similarities_df, ignore_index=True)

# Getting datasize to then order states by datasize, from biggest to smallest

DATA_PATH = "dissertation_project/federated-learning/example/tests"
state_sizes = os.path.join(DATA_PATH, "tests/intermediate_results/state_sizes.txt")
data_sizes_dict = {}

with open(state_sizes, 'r') as f:
    for line in f:
        if ':' in line:
            state, rest = line.strip().split(':')
            rest = int(round(float(rest)))
            data_sizes_dict[state.strip()] = rest

ordered_states = sorted(data_sizes_dict, key=data_sizes_dict.get, reverse=True)

# === 6. Boxplot per state for bottom 1% and 2.5% cosine similarities

bottom_pct_per_state = []
percentiles = [1, 2.5]

for p in percentiles:
    for state_name, group in similarity_df.groupby("state"):
        k_pct = max(1, int((p/100) * len(group))) 
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
    plt.savefig(os.path.join(RESULTS_DIR, f"boxplot_bottom_{label_clean}_cosine_similarity_per_state_frequency_encoding.pdf"))
    plt.close()


# === 7. Boxplots per state for most heterogeneous samples (cosine similarity below thresholds)

similarities = similarity_df["similarity"]

thresholds_info = [
    (np.percentile(similarities, p), f"{p} pct") for p in [1, 2.5]
]

# Limite inferior do boxplot
q1 = np.percentile(similarities, 25)
q3 = np.percentile(similarities, 75)
iqr = q3 - q1
lower_limit = q1 - 1.5 * iqr


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

      
        plt.savefig(os.path.join(RESULTS_DIR, f"boxplot_cosine_similarity_below_{label_clean}_per_state_frequency_encoding.pdf"))
        plt.close()


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
plt.savefig(os.path.join(RESULTS_DIR, "global_cumulative_cosine_similarity_frequency_encoding.pdf"))
plt.close()
