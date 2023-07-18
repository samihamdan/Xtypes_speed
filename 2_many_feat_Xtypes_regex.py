# %%
from julearn import run_cross_validation, PipelineCreator
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np
import time

N_FEAT_GROUPS = 150


def get_col_names(n_features):
    names = []
    for i in range(n_features):
        t_group = i % N_FEAT_GROUPS
        names.append(f"fish_{t_group}_feat_{i}")
    return names


def get_Xtypes(column_names):
    out = {f"fish_{i}": [] for i in range(N_FEAT_GROUPS)}
    for icol, col in enumerate(column_names):
        t_group = icol % N_FEAT_GROUPS
        out[f"fish_{t_group}"].append(col)
    return out


# %%
df_timings = pd.DataFrame()
for n_feat in range(1_000, 10_001, 2_000):
    X, y = make_classification(n_samples=100, n_features=n_feat)
    col_names = get_col_names(n_feat)
    df = pd.DataFrame(X, columns=col_names)
    X = list(df.columns)
    X_types = get_Xtypes(X)
    X_types_regexp = {
        f"fish_{i}": f"fish_{i}_feat_.*" for i in range(N_FEAT_GROUPS)
    }
    df["y"] = y
    y = "y"
    pipe = (
        PipelineCreator("classification", apply_to="fish_1")
        .add("filter_columns", apply_to="*", keep="fish_1")
        .add("dummy")
    )

    start = time.time()

    run_cross_validation(
        X=X, y=y, data=df, X_types=X_types, model=pipe, verbose=8001, cv=2
    )

    timing_list = time.time() - start
    print(f"List Feet {n_feat}")
    print(timing_list)

    pipe = (
        PipelineCreator("classification", apply_to="fish_1")
        .add("filter_columns", apply_to="*", keep="fish_1")
        .add("dummy")
    )
    start = time.time()
    run_cross_validation(
        X=[".*feat_.*"],
        y=y,
        data=df,
        X_types=X_types,
        model=pipe,
        verbose=8001,
        cv=2,
    )

    timing_regex = time.time() - start
    print(f"Regex Feet {n_feat}")
    print(timing_regex)

    pipe = (
        PipelineCreator("classification", apply_to="fish_1")
        .add("filter_columns", apply_to="*", keep="fish_1")
        .add("dummy")
    )
    start = time.time()
    run_cross_validation(
        X=[".*feat_.*"],
        y=y,
        data=df,
        X_types=X_types_regexp,
        model=pipe,
        verbose=8001,
        cv=2,
    )

    timing_regex_xtypes = time.time() - start
    print(f"X_types regex Regex Feet {n_feat}")
    print(timing_regex)
    _df = pd.DataFrame(
        dict(
            list=[timing_list],
            regex=[timing_regex],
            xtypes_regex=[timing_regex_xtypes],
            n_feat=[n_feat],
        )
    )
    df_timings = pd.concat([df_timings, _df])
    print(df_timings)

df_timings.to_csv("save_that_timing_of_Xtypes_regex.csv")
print(df_timings)

# %%
