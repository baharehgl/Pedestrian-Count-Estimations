import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("processed/processed.csv")   # or your main CSV
# 1) Save a small sample table as an image (easiest: screenshot from Jupyter)

print(df.head(10))  # show this in Jupyter and screenshot it

# 2) Histogram of pm_tot
plt.figure(figsize=(5,4))
plt.hist(df["pm_tot"], bins=10, edgecolor="black")
plt.xlabel("pm_tot (pedestrian count)")
plt.ylabel("Number of observations")
plt.title("Distribution of pm_tot")
plt.tight_layout()
plt.savefig("slide3_pm_tot_hist.png", dpi=200)
plt.show()