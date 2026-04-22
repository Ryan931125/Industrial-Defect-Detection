import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.container import BarContainer


def model_slug(model_name):
    return model_name.lower().replace(" ", "_").replace("-", "_").replace(".", "")


def normalize_binary_label(value):
    if pd.isna(value):
        return None

    text = str(value).strip().lower()
    if not text:
        return None

    if text in {"yes", "y", "true", "1"}:
        return "Yes"
    if text in {"no", "n", "false", "0"}:
        return "No"

    if "no defect" in text:
        return "No"
    if "normal" in text:
        return "No"
    if "defect" in text:
        return "Yes"

    if "yes" in text:
        return "Yes"
    if "no" in text:
        return "No"

    return None


def save_confusion_matrix_plot(cm, output_path, title):
    total = cm.to_numpy().sum()
    if total == 0:
        return

    row_sums = cm.sum(axis=1).replace(0, 1)
    row_pct = cm.div(row_sums, axis=0)
    annot = cm.astype(int).astype(str) + "\n" + (row_pct * 100).round(1).astype(str) + "%"

    plt.figure(figsize=(5.2, 4.5))
    ax = sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=False, linewidths=0.5, linecolor="white")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


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
    labels = ["No", "Yes"]
    defect_rows_by_model = {name: [] for name in model_files}

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

            if {"Category", "Ground Truth", "Prediction"}.issubset(df.columns):
                defect_df = df[df["Category"] == "Defect Detection"].copy()
                if not defect_df.empty:
                    defect_df["gt_label"] = defect_df["Ground Truth"].apply(normalize_binary_label)
                    defect_df["pred_label"] = defect_df["Prediction"].apply(normalize_binary_label)
                    defect_df = defect_df.dropna(subset=["gt_label", "pred_label"])

                    if not defect_df.empty:
                        cm = pd.crosstab(defect_df["gt_label"], defect_df["pred_label"])
                        cm = cm.reindex(index=labels, columns=labels, fill_value=0)

                        split_slug = split
                        model_name_slug = model_slug(model_name)
                        cm.to_csv(output_dir / f"confusion_matrix_defect_{split_slug}_{model_name_slug}.csv")
                        save_confusion_matrix_plot(
                            cm,
                            output_dir / f"confusion_matrix_defect_{split_slug}_{model_name_slug}.png",
                            f"Defect Detection Confusion Matrix\nSplit {split} - {model_name}",
                        )
                        defect_rows_by_model[model_name].append(defect_df[["gt_label", "pred_label"]])

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
        plt.close()

    if not summary_by_split.empty:
        pivot_split = summary_by_split.pivot(index="split", columns="model", values="overall_acc")
        pivot_split = pivot_split.reindex(index=splits)
        plt.figure(figsize=(7, 4.2))
        ax = sns.heatmap(pivot_split, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1, linewidths=0.5)
        ax.set_title("Accuracy Heatmap by Defect Split")
        ax.set_xlabel("Model")
        ax.set_ylabel("Split")
        plt.tight_layout()
        plt.savefig(output_dir / "heatmap_by_split.png", dpi=300)
        plt.close()

    if not summary_by_category.empty:
        pivot_cat = summary_by_category.pivot(index="category", columns="model", values="acc")
        plt.figure(figsize=(7.2, 4.8))
        ax = sns.heatmap(pivot_cat, annot=True, fmt=".3f", cmap="YlOrRd", vmin=0, vmax=1, linewidths=0.5)
        ax.set_title("Accuracy Heatmap by Question Category")
        ax.set_xlabel("Model")
        ax.set_ylabel("Category")
        plt.tight_layout()
        plt.savefig(output_dir / "heatmap_by_category.png", dpi=300)
        plt.close()

    for model_name, parts in defect_rows_by_model.items():
        if not parts:
            continue

        combined = pd.concat(parts, ignore_index=True)
        cm = pd.crosstab(combined["gt_label"], combined["pred_label"])
        cm = cm.reindex(index=labels, columns=labels, fill_value=0)

        model_name_slug = model_slug(model_name)
        cm.to_csv(output_dir / f"confusion_matrix_defect_all_{model_name_slug}.csv")
        save_confusion_matrix_plot(
            cm,
            output_dir / f"confusion_matrix_defect_all_{model_name_slug}.png",
            f"Defect Detection Confusion Matrix\nAll Splits - {model_name}",
        )


if __name__ == "__main__":
    main()
