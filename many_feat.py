# %% 
from julearn import run_cross_validation
from sklearn.datasets import make_classification
import pandas as pd
import time 



# %% 
df_timings = pd.DataFrame()
for n_feat in range(1_000, 10_001, 2_000):
    X, y = make_classification(n_samples=100, n_features=n_feat)
    df = pd.DataFrame(X, columns=[str(i) + "feat" for i in range(len(X[0]))])
    X = list(df.columns)
    df["y"] = y
    y="y"

    start = time.time()
    run_cross_validation(
        X=X, y=y, data=df,
        model="dummy", problem_type="classification",
        verbose=8001, cv=2
    )

    timing_list = time.time() - start
    print(f"List Feet{n_feat}")
    print(timing_list)

    start = time.time()
    run_cross_validation(
        X=[".*feat"], y=y, data=df,
        model="dummy", problem_type="classification",
        verbose=8001, cv=2
    )

    timing_regex = time.time() - start
    print(f"Regex Feet{n_feat}")
    print(timing_regex)
    _df = pd.DataFrame(dict(list=[timing_list], regex=[timing_regex]))
    df_timings = pd.concat([df_timings, _df])
    print(df_timings)

df_timings.to_csv("save_that_timing.csv")
print(df_timings)



