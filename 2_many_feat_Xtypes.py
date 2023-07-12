# %% 
from julearn import run_cross_validation, PipelineCreator
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np
import time 

def get_Xtypes(column_names):
    counter = 0  
    out = {f"fish_{i}":[] for i in range(150)}
    for col in column_names:
        out[f"fish_{counter}"].append(col)
        counter += 1 
        if counter==150:
            counter=0
    return out




# %% 
df_timings = pd.DataFrame()
for n_feat in range(1_000, 10_001, 2_000):
    X, y = make_classification(n_samples=100, n_features=n_feat)
    
    df = pd.DataFrame(X, columns=[str(i) + "_feat" for i in range(len(X[0]))])
    X = list(df.columns)
    X_types = get_Xtypes(X)
    df["y"] = y
    y="y"
    pipe = (PipelineCreator("classification", apply_to="fish_1")
     .add("filter_columns", apply_to="*", keep="fish_1")
     .add("dummy")
    )

    start = time.time()

    run_cross_validation(
        X=X, y=y, data=df,X_types=X_types,
        model=pipe,
        verbose=8001, cv=2
    )

    timing_list = time.time() - start
    print(f"List Feet{n_feat}")
    print(timing_list)

    pipe = (PipelineCreator("classification", apply_to="fish_1")
     .add("filter_columns", apply_to="*", keep="fish_1")
     .add("dummy")
    )
    start = time.time()
    run_cross_validation(
        X=[".*feat"], y=y, data=df,X_types=X_types,
        model=pipe,
        verbose=8001, cv=2
    )

    timing_regex = time.time() - start
    print(f"Regex Feet{n_feat}")
    print(timing_regex)
    _df = pd.DataFrame(dict(list=[timing_list], regex=[timing_regex]))
    df_timings = pd.concat([df_timings, _df])
    print(df_timings)

df_timings.to_csv("save_that_timing_of_Xtypes.csv")
print(df_timings)



