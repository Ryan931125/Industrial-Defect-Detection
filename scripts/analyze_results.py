import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.container import BarContainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_root", type=str, required=True, help="Folder containing 03/05/07/09 split outputs")
    parser.add_argument("--output_dir", type=str, default="assets/results")
    args = parser.parse_args()

    source_root = Path(args.source_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = ["03", "05", "07", "09"]
    model_files = {
        "LLaVA-1.5": "llava-hf_llava-1_5-7b-hf_predictions.csv",
        "LLaVA-1.6": "llava-hf_llava-v1_6-mistral-7b-hf_predictions.csv",
        "Qwen3-VL-8B": "Qwen_Qwen3-VL-8B-Instruct_predictions.csv",
    }

    rows = []
    cat_rows = []

    for split in splits:
        for model_name, filename in model_files.items():
            csv_path = source_root / split / filename
            if not csv_path.exists():
                continue

            df = pd.read_csv(csv_path)
            rows.append(
                {
                    "split": split,
                    "model": model_name,
                    "overall_acc": float(df["Correct"].mean()),
                    "n": len(df),
                }
            )

            grouped = df.groupby("Category")["Correct"].mean().reset_index()
            for _, row in grouped.iterrows():
                cat_rows.append(
                    {
                        "split": split,
                        "model": model_name,
                        "category": row["Category"],
                        "acc": float(row["Correct"]),
                    }
                )

    summary_by_split = pd.DataFrame(rows)
    category_by_split = pd.DataFrame(cat_rows)

    all_summary = []
    for model_name, filename in model_files.items():
        dfs = []
        for split in splits:
            csv_path = source_root / split / filename
            if csv_path.exists():
                dfs.append(pd.read_csv(csv_path))
        if dfs:
            merged = pd.concat(dfs, ignore_index=True)
            all_summary.append(
                {
                    "model": model_name,
                    "overall_acc": float(merged["Correct"].mean()),
                    "n": len(merged),
                }
            )

    summary_overall = pd.DataFrame(all_summary)
    summary_by_category = (
        category_by_split.groupby(["model", "category"])["acc"].mean().reset_index()
        if not category_by_split.empty
        else pd.DataFrame(columns=["model", "category", "acc"])
    )

    summary_by_split.to_csv(output_dir / "summary_by_split.csv", index=False)
    summary_overall.to_csv(output_dir / "summary_overall.csv", index=False)
    summary_by_category.to_csv(output_dir / "summary_by_category_all_splits.csv", index=False)

    if not summary_overall.empty:
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(7, 4.5))
        ax = sns.barplot(data=summary_overall, x="model", y="overall_acc", palette="viridis", edgecolor="black")
        ax.set_ylim(0, 1)
        ax.set_title("Overall Accuracy Across All Splits")
        ax.set_xlabel("Model")
        ax.set_ylabel("Accuracy")
        for container in ax.containers:
            if isinstance(container, BarContainer):
                ax.bar_label(container, fmt="%.3f", padding=3)
        plt.tight_layout()
        plt.savefig(output_dir / "overall_model_comparison.png", dpi=300)


if __name__ == "__main__":
    main()
