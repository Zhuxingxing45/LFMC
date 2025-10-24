import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 数据表
data = {
    "BaseModel": ["LLama3-8b", "LLama3-8b", "LLama3-8b"],
    "FT Data": ["Origin", "Origin+LFUD", "Origin+LOCD"],
    "FOLIO": [58.29, 62.23, 65.85],
    "LogiQA_v2": [64.30, 65.07, 66.67],
    "ReClor": [73.93, 75.53, 75.93],
    "logiqa-zh": [50.21, 51.74, 53.56],
}

# 转成 DataFrame
df = pd.DataFrame(data)

# 设置 Seaborn 主题
sns.set_theme(style="whitegrid", font_scale=1.1)

# 按数据集绘制4个热力图
datasets = ["FOLIO", "LogiQA_v2", "ReClor", "logiqa-zh"]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for i, dataset in enumerate(datasets):
    # 只保留 BaseModel, FT Data, 当前数据集
    pivot_df = df.pivot(index="BaseModel", columns="FT Data", values=dataset)
    
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        cbar=False,
        ax=axes[i],
        linewidths=0.5,
        annot_kws={"size": 10, "weight": "bold"}
    )
    
    axes[i].set_title(dataset, fontsize=12, weight="bold")
    axes[i].set_xlabel("FT Data")
    axes[i].set_ylabel("BaseModel")

plt.tight_layout()
plt.show()
plt.savefig("heatmap.png", dpi=300, bbox_inches="tight")
