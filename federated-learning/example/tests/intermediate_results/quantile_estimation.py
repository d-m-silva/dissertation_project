import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

from sklearn.model_selection import train_test_split
from federated.feddata import ACSDataStatesBySize, ACSDataStatesCode
from folktables import ACSDataSource, ACSIncome


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

        x_train, x_global_test, y_train, y_global_test = train_test_split(df_features, df_target, test_size=0.1,
                                                                              random_state=seed)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=seed)
        x_generalization_test = pd.concat([x_generalization_test, x_global_test])
        y_generalization_test = pd.concat([y_generalization_test, y_global_test])
        local_acs_data[state_code] = {"x_train": x_train, "x_val": x_val, "y_train": y_train, "y_val": y_val}

    return local_acs_data


#### Global percentile estimation for PINCP #####

clients_number = 50
clients_name = [state.name for state in ACSDataStatesBySize][:clients_number]
data = load_adult_dummy(list_states=clients_name)

adult_data = pd.concat([v["x_train"]["PINCP"] for k, v in data.items()])

state_counts = adult_data.groupby("State")["PINCP"].count()

# Global 25% and 75% percentiles
pincp_global = adult_data["PINCP"].dropna()
Q25_global = np.percentile(pincp_global, 25)
Q75_global = np.percentile(pincp_global, 75)

# Local 25% and 75% percentiles
state_q25 = adult_data.groupby("State")["PINCP"].quantile(0.25)
state_q75 = adult_data.groupby("State")["PINCP"].quantile(0.75)

max_Q25_state = state_q25.max()
min_Q25_state = state_q25.min()

max_Q75_state = state_q75.max()
min_Q75_state = state_q75.min()

#1. Weighted estimation 

Q75_weighted = (state_q75 * state_counts).sum() / state_counts.sum()

Q25_weighted = (state_q25 * state_counts).sum() / state_counts.sum()

# Differences

diff_max_pond = (Q75_weighted - Q75_global)/Q75_global

diff_min_pond = (Q25_weighted - Q25_global) #Q25_global is zero, absolute error is needed

#2. Iterative stochastic estimation

import numpy as np

def iterative_stochastic_estimation(
        local_quantiles_dict,   
        local_sizes_dict,                                    
        max_iter=50
    ):
    """
    Estimates a global quantile using an iterative stochastic update rule.
    
    Parameters
    ----------
    local_quantiles_dict : dict
        Mapping from each group/state to its local quantile estimate.
    local_sizes_dict : dict
        Mapping from each group/state to its sample size.
    max_iter : int
        Maximum number of iterations.
    
    Returns
    -------
    float
        Estimated global quantile (Q_min).
    """

    # Prepare data
    states = list(local_quantiles_dict.keys())
    q_values = np.array(list(local_quantiles_dict.values()))
    sizes = np.array([local_sizes_dict[s] for s in states])

    total_size = sizes.sum()

    n_states = len(states)
    weights = sizes / sizes.sum()

    # Initialization
    Q_min = q_values.min()

    for _ in range(max_iter):
        prev_Q_min = Q_min

        # Weighted random state selection
        idx = np.random.choice(n_states, p=weights)
        state_i = states[idx]

        n_i = local_sizes_dict[state_i]
        q_i = local_quantiles_dict[state_i]

        # Update rule
        fi = n_i / total_size
        Fi = 1 - fi
        Q_min = q_i * Fi + fi * Q_min

        # Convergence check
        if Q_min == prev_Q_min:
            break

    return Q_min

q = 0.75
Q_75_list = [group["PINCP"].dropna().quantile(q) for state, group in adult_data.groupby("State")]

Q75_est = iterative_stochastic_estimation(
    local_quantiles_dict = Q_75_list,
    local_sizes_dict = state_counts,
    max_iter=50
)

error_iter = ((Q75_est - Q75_global)/Q75_global)*100

q = 0.25
P_25_list = [group["PINCP"].dropna().quantile(q) for state, group in adult_data.groupby("State")]

Q25_est = iterative_stochastic_estimation(
    local_quantiles_dict = Q_25_list,
    local_sizes_dict = state_counts,
    max_iter=50
)

error_iter = (Q25_est - Q25_global)

#### Global quantile estimation for cosine similarity #####

