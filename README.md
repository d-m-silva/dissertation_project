# Cross-Silo Federated Learning: An Empirical Study on Learning Gains

## Project Structure

```
dissertation_project/
│
├── LICENSE
├── README.md (this file)
├── requirements.txt #Python packages
│
└── federated-learning/
    ├── data/ 
    │   ├── README.md
    ├── example/
    │   ├── tests/ 
    │   │   ├── filtering_logs/ 
    │   │   │   └── ...
    │   │   ├── results/
    │   │   │   └── ...
    │   │   └── test_run_fl.py  
    │   ├── config_exp.py 
    │   └── model.py #Used models
    ├── federated/
    │   ├── client/
    │   │   └── fed_client.py
    │   ├── server/
    │   │   ├── fed_agg.py
    │   │   ├── fed_avg.py
    │   │   ├── fed_differencial_privacy.py
    │   │   └── fed_per.py    
    │   └── feddata/
    │       ├── acs_data_states.py
    │       ├── adult_fl_loader.py 
    │       └── sent140_loader_silo_exp.py 
    │
    ├── config.py
    └── fed_learning_exp.py
```

   
        
        
