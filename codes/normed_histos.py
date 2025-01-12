import matplotlib.pyplot as plt
import numpy as np
import json

raw_metrics_baseline_path = "../metrics/baseline_1000/raw_metrics.json"
raw_metrics_normed_path = "../metrics/normed/raw_metrics.json"

with open(raw_metrics_baseline_path) as fin:
    baseline_metrics = list(json.load(fin).values())[0]

with open(raw_metrics_normed_path) as fin:
    normed_metrics = list(json.load(fin).values())[0]

num_models = 2
num_metrics = len(baseline_metrics.keys())

fig, ax = plt.subplots(1,num_metrics,figsize=(5*num_metrics,5))

for i,metric in enumerate(baseline_metrics.keys()):
    baseline_val = baseline_metrics[metric]
    normed_val = normed_metrics[metric]
    ax[i].bar([f"{metric}_baseline", f"{metric}_normed"], [baseline_val, normed_val])
    ax[i].set_title(metric)
    ax[i].set_xlabel("Model")
    ax[i].set_ylabel(metric)
    # ax[i].set_xticks(rotation=90)

plt.tight_layout()
save_path = "../figures/normed_histos.png"
plt.savefig(save_path)
plt.close()
print(f"Saved graph to {save_path}")

