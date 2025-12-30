from sklearn.model_selection import train_test_split
from pathlib import Path
from folktables import ACSDataSource, ACSIncome
from ..config import FederatedData
from .acs_data_states import ACSDataStatesCode
from typing import List

import numpy as np
import random
import pandas as pd


def load_acs_local_data(list_states: list[str] = None, year: str = '2021', horizon: str = '1-Year', split: bool = True,
                         use_subset_criteria: str = "avg_+-_std", subset_dir_name: str = "subset_criteria_logs",
                         num_samples: int = 30):
    """Load the ACS dataset for federated learning, returning both FL subset and full data."""

    data_source = ACSDataSource(survey_year=year, horizon=horizon, survey='person')

    if list_states is not None:
        acs_data = data_source.get_data(states=list_states, download=False)
    else:
        acs_data = data_source.get_data(download=True)
        list_states = [state.name for state in ACSDataStatesCode]

    acs_data.rename(columns={"RELSHIPP": "RELP"}, inplace=True)

    ACSIncome._features.append('PINCP')
    df_features_all, _, _ = ACSIncome.df_to_pandas(acs_data)

    df_features_all = df_features_all.loc[:, ~df_features_all.columns.duplicated()]

    # Calculate global statistics
    global_avg = df_features_all['PINCP'].mean()
    global_std = df_features_all['PINCP'].std()
    n_global = len(df_features_all['PINCP'])
    q1 = df_features_all['PINCP'].quantile(0.25)
    q3 = df_features_all['PINCP'].quantile(0.75)
    iqr = q3 - q1
    percentile_5 = df_features_all['PINCP'].quantile(0.05)
    percentile_95 = df_features_all['PINCP'].quantile(0.95)

    local_acs_data = {}
    x_generalization_test = pd.DataFrame()
    y_generalization_test = pd.DataFrame()
    subset_states = {}

    for state_code in list_states:
        state_data = acs_data.loc[acs_data["ST"] == ACSDataStatesCode[state_code].value]
        df_features, df_target, _ = ACSIncome.df_to_pandas(state_data)

        # Calculate state statistics
        state_avg = df_features['PINCP'].mean()
        state_std = df_features['PINCP'].std()
        n_state = len(df_features['PINCP'])
        q1_state = df_features['PINCP'].quantile(0.25)
        q3_state = df_features['PINCP'].quantile(0.75)
        iqr_state = q3_state - q1_state
        percentile_5_state = df_features['PINCP'].quantile(0.05)
        percentile_95_state = df_features['PINCP'].quantile(0.95)


        x_all = df_features.copy().reset_index(drop=True)
        y_all = df_target.copy().reset_index(drop=True)

        x_all = x_all.loc[:, ~x_all.columns.duplicated()]
        y_all = y_all.loc[:, ~y_all.columns.duplicated()]

        # Determine bounds based on criteria
        if use_subset_criteria == "avg_+-_std":
            lower, upper = global_avg - global_std, global_avg + global_std
        elif use_subset_criteria == "avg_ic_95":
            margin = norm.ppf(0.975) * (global_std / np.sqrt(n_global))
            lower, upper = global_avg - margin, global_avg + margin
        elif use_subset_criteria == "percentiles_5_95":
            lower, upper = percentile_5, percentile_95
        elif use_subset_criteria == "iqr":
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        elif use_subset_criteria == "avg_+-_std_k3":
            lower, upper = global_avg - 3 * global_std, global_avg + 3 * global_std
        elif use_subset_criteria == "random":
            p = len(x_all) / len(df_features_all)
            mask = np.random.rand(len(x_all)) < p
            x_subset = x_all[mask].drop(columns=["PINCP"])
            y_subset = y_all[mask]
        elif use_subset_criteria == "density_clipping":
             state_income = x_all["PINCP"]
             others_data = acs_data.loc[acs_data["ST"] != ACSDataStatesCode[state_code].value]
             others_feature, _ , _ = ACSIncome.df_to_pandas(others_data)
             others_PINCP = others_feature['PINCP']
             clipped_income, _ = clip_density(state_income, others_PINCP, state_code)
             mask = x_all["PINCP"].isin(clipped_income)
             x_subset = x_all[mask].drop(columns=["PINCP"])
             y_subset = y_all[mask]

        elif use_subset_criteria == "avg_+-_std_client":
           lower, upper = state_avg - state_std, state_avg + state_std
        elif use_subset_criteria == "avg_ic_95_client":
            margin = norm.ppf(0.975) * (state_std / np.sqrt(n_state))
            lower, upper = state_avg - margin, state_avg + margin
        elif use_subset_criteria == "percentiles_5_95_client":
            lower, upper = percentile_5_state, percentile_95_state
        elif use_subset_criteria == "iqr_client":
            lower, upper = q1_state - 1.5 * iqr_state, q3_state  + 1.5 * iqr_state
        elif use_subset_criteria == "avg_+-_std_k3_client":
            lower, upper = state_avg - 3 * state_std, state_avg + 3 * state_std

        elif "cosine_similarity_1_pct":
             # Filter out other states' data (excluding current state) and drop rows with missing values
             others_data = acs_data.loc[acs_data["ST"] != ACSDataStatesCode[state_code].value]
             others_feature, _ , _ = ACSIncome.df_to_pandas(others_data)

             ordinal_features = ['SCHL']
             frequency_features = ['POBP', 'OCCP']
             one_hot_features = ['MAR', 'RAC1P', 'COW', 'RELP']
             numerical_features = ['AGEP', 'WKHP', 'PINCP']
             binary_features = ['SEX']

             others_feature[binary_features] = others_feature[binary_features].map({'1': 1, '2': 2})
            
             all_features = numerical_features + ordinal_features + frequency_features + one_hot_features + other_features

             available_columns = [col for col in all_features if col in others_feature.columns]
             others_feature = others_feature[available_columns].dropna()
             x_all = x_all[available_columns].dropna()

             x_all = x_all.loc[:, ~x_all.columns.duplicated()]
             others_feature = others_feature.loc[:, ~others_feature.columns.duplicated()]

             preprocessor = ColumnTransformer([
             ("num", StandardScaler(), numerical_features),
             ("ord", OrdinalEncoder(), ordinal_features),
             ("frequency", FrequencyEncoder(columns=frequency_features), frequency_features),
             ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output = False), one_hot_features),
             ("dummy_sex", "passthrough", other_features),
             ])
             preprocessor.fit(others_feature)

             x_all_trans = preprocessor.transform(x_all)
             x_rest_trans = preprocessor.transform(others_feature)

             x_all_trans = np.asarray(x_all_trans.toarray()) if hasattr(x_all_trans, "toarray") else np.asarray(x_all_trans)
             x_rest_trans = np.asarray(x_rest_trans.toarray()) if hasattr(x_rest_trans, "toarray") else np.asarray(x_rest_trans)


             # Compute cosine similarity between each row in current state and the mean vector of other states
             x_rest_mean = x_rest_trans.mean(axis=0).reshape(1, -1)
             similarities = cosine_similarity(x_all_trans, x_rest_mean).flatten()

             # Keep only rows with cosine similarity above a fixed threshold (1% percentile)
             mask = similarities > np.percentile(similarities, 1)

             x_subset = x_all[mask].drop(columns=["PINCP"]).reset_index(drop=True)
             y_subset = y_all[mask].reset_index(drop=True)

        elif "cosine_similarity_thr_2_5_pct":
             # Filter out other states' data (excluding current state) and drop rows with missing values ("cosine_similarity_thr0_8" before)
             others_data = acs_data.loc[acs_data["ST"] != ACSDataStatesCode[state_code].value]
             others_feature, _ , _ = ACSIncome.df_to_pandas(others_data)

             ordinal_features = ['SCHL']
             frequency_features = ['POBP', 'OCCP']
             one_hot_features = ['MAR', 'RAC1P', 'COW', 'RELP']
             numerical_features = ['AGEP', 'WKHP', 'PINCP']
             binary_features = ['SEX']

             others_feature[binary_features] = others_feature[binary_features].map({'1': 1, '2': 2})
            
             all_features = numerical_features + ordinal_features + frequency_features + one_hot_features + binary_features

             available_columns = [col for col in all_features if col in others_feature.columns]
             others_feature = others_feature[available_columns].dropna()

             x_all = x_all[available_columns].dropna()

             x_all = x_all.loc[:, ~x_all.columns.duplicated()]
             others_feature = others_feature.loc[:, ~others_feature.columns.duplicated()]

             preprocessor = ColumnTransformer([
             ("num", StandardScaler(), numerical_features),
             ("ord", OrdinalEncoder(), ordinal_features),
             ("frequency", FrequencyEncoder(columns=frequency_features), frequency_features),
             ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output = False), one_hot_features),
             ("dummy_sex", "passthrough", other_features),
             ])
             preprocessor.fit(others_feature)

             x_all_trans = preprocessor.transform(x_all)
             x_rest_trans = preprocessor.transform(others_feature)

             x_all_trans = np.asarray(x_all_trans.toarray()) if hasattr(x_all_trans, "toarray") else np.asarray(x_all_trans)
             x_rest_trans = np.asarray(x_rest_trans.toarray()) if hasattr(x_rest_trans, "toarray") else np.asarray(x_rest_trans)


             # Compute cosine similarity between each row in current state and the mean vector of other states
             x_rest_mean = x_rest_trans.mean(axis=0).reshape(1, -1)
             similarities = cosine_similarity(x_all_trans, x_rest_mean).flatten()

             # Keep only rows with cosine similarity above a fixed threshold (2.5% percentile)
             mask = similarities > np.percentile(similarities, 2.5)

             x_subset = x_all[mask].drop(columns=["PINCP"]).reset_index(drop=True)
             y_subset = y_all[mask].reset_index(drop=True)


        else:
            raise ValueError(f"Unknown subset criteria: {use_subset_criteria}")

        if use_subset_criteria != "random" and use_subset_criteria != "density_clipping" and use_subset_criteria != "cosine_similarity_thr_1_pct" and use_subset_criteria != "cosine_similarity_thr_2_5_pct":
            filtered_idx = (x_all["PINCP"] >= lower) & (x_all["PINCP"] <= upper)
            x_subset = x_all[filtered_idx].drop(columns=["PINCP"]).reset_index(drop=True)
            y_subset = y_all[filtered_idx].reset_index(drop=True)

        pct_x = len(x_subset) / len(x_all) if len(x_all) > 0 else 0
        subset_states[state_code] = {"x_pct": pct_x}

        if split and len(x_subset) >= num_samples:
            x_train, x_global_test, y_train, y_global_test = train_test_split(
                x_subset, y_subset, test_size=0.1, random_state=42)
            x_train, x_val, y_train, y_val = train_test_split(
                x_train, y_train, test_size=0.1, random_state=42)

            x_generalization_test = pd.concat([x_generalization_test, x_global_test])
            y_generalization_test = pd.concat([y_generalization_test, y_global_test])

            local_acs_data[state_code] = {
                "fl": {
                    "x_train": x_train,
                    "x_val": x_val,
                    "y_train": y_train,
                    "y_val": y_val
                },
                "all": {
                    "x": x_all,
                    "y": y_all
                }
            }
        else:
            local_acs_data[state_code] = {
                "all": {
                    "x": df_features,
                    "y": df_target
                }
            }

    # Save subset states information
    subset_path = Path("/home/dudasilva/federated-learning/example/tests") / subset_dir_name
    subset_path.mkdir(parents=True, exist_ok=True)
    filename = f"subset_states_{use_subset_criteria.replace(' ', '').replace('+', 'plus').replace('-', 'minus')}.txt"
    with open(subset_path / filename, "w") as f:
        for state, pct in subset_states.items():
            f.write(f"{state}: x_pct={pct['x_pct']:.3f}\n")

    return FederatedData(
        clients_data=local_acs_data,
        x_gen_test=x_generalization_test,
        y_gen_test=y_generalization_test
    )




def load_acs_local_data_old(list_states: List[str] = None, year: str = '2021', horizon: str = '1-Year', split: bool = True, use_subset_criteria: bool = False):
    # Load the ACS dataset for federated learning, returning both FL subset and full data.

    data_source = ACSDataSource(
        survey_year=year, horizon=horizon, survey='person'
    )

    if list_states is not None:
        acs_data = data_source.get_data(states=list_states, download=False)
    else:
        acs_data = data_source.get_data(download=False)
        list_states = [state.name for state in ACSDataStatesCode]

    acs_data.rename(columns={"RELSHIPP": "RELP"}, inplace=True)
    ACSIncome._features.append('PINCP')

    df_features_all, _, _ = ACSIncome.df_to_pandas(acs_data)
    global_std = np.std(df_features_all['PINCP'])
    global_avg = np.mean(df_features_all['PINCP'])

    seed = 42
    local_acs_data = {}
    x_generalization_test = pd.DataFrame()
    y_generalization_test = pd.DataFrame()
    subset_states = []

    for state_code in list_states:
        state_data = acs_data.loc[acs_data["ST"] == ACSDataStatesCode[state_code].value]
        df_features, df_target, _ = ACSIncome.df_to_pandas(state_data)

        x_all = df_features.copy()
        y_all = df_target.copy()

        if use_subset_criteria:
            filtered_idx = (
                (x_all["PINCP"] >= global_avg - global_std) &
                (x_all["PINCP"] <= global_avg + global_std)
            )
            x_subset = x_all[filtered_idx].drop(columns=["PINCP"])
            y_subset = y_all[filtered_idx]
        else:
            x_subset = x_all.drop(columns=["PINCP"])
            y_subset = y_all

        num_samples = 30  # can be another number

        df_features.drop('PINCP', axis=1, inplace=True)

        if use_subset_criteria:
            if len(x_subset) >= num_samples and len(y_subset) >= num_samples:
                subset_states.append(state_code)

                x_train, x_global_test, y_train, y_global_test = train_test_split(
                    x_subset, y_subset, test_size=0.1, random_state=seed)
                x_train, x_val, y_train, y_val = train_test_split(
                    x_train, y_train, test_size=0.1, random_state=seed)

                x_generalization_test = pd.concat([x_generalization_test, x_global_test])
                y_generalization_test = pd.concat([y_generalization_test, y_global_test])

                #Data is divided in two categories: one is used for FL and the other for Finetune ("all")
                local_acs_data[state_code] = {
                    "fl": {
                        "x_train": x_train,
                        "x_val": x_val,
                        "y_train": y_train,
                        "y_val": y_val
                    },
                    "all": {
                        "x": x_all,
                        "y": y_all
                    }
                }
            else:
                local_acs_data[state_code] = {
                    "all": {
                        "x": df_features,
                        "y": df_target
                    }
                }
        else:
            x_train, x_global_test, y_train, y_global_test = train_test_split(
                df_features, df_target, test_size=0.1, random_state=seed)
            x_train, x_val, y_train, y_val = train_test_split(
                x_train, y_train, test_size=0.1, random_state=seed)

            x_generalization_test = pd.concat([x_generalization_test, x_global_test])
            y_generalization_test = pd.concat([y_generalization_test, y_global_test])


            #Data is divided in two categories: one is used for FL and the other for Finetune ("all") -> no filtering in this case!
            local_acs_data[state_code] = {
                "fl": {
                    "x_train": x_train,
                    "x_val": x_val,
                    "y_train": y_train,
                    "y_val": y_val
                },
                "all": {
                    "x": df_features,
                    "y": df_target
                }
            }

    return FederatedData(
        clients_data=local_acs_data,
        x_gen_test=x_generalization_test,
        y_gen_test=y_generalization_test
    )

