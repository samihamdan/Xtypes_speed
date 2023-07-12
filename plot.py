import pandas as pd
import seaborn as sns

df = pd.read_csv("./save_that_timing.csv")
df_types = pd.read_csv("./save_that_timing_of_Xtypes.csv")
df["n_feat"]= range(1000, 10_001, 2000)
df_types["n_feat"]= range(1000, 10_001, 2000)
df = pd.concat([df.assign(kind="simple"), df_types.assign(kind="Xtypes")])

df = pd.melt(df,
             value_vars=["list", "regex"], id_vars=["n_feat", "kind"], 
             value_name="Time in sec"
             )

f = sns.catplot(x="n_feat", y="Time in sec", hue="variable",col="kind", data=df, kind="point")
f.savefig("haha.png")
