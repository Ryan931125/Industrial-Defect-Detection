import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.container import BarContainer


def model_slug(model_name):
    return model_name.lower().replace(" ", "_").replace("-", "_").replace(".", "")


def category_slug(category_name):
    return category_name.lower().replace(" ", "_").replace("/", "_")


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


def normalize_mount_side(value):
    if pd.isna(value):
        return None

    text = str(value).strip().lower()
    if not text:
        return None

    if text in {"a", "(a)", "a)", "top"}:
        return "Top"
    if text in {"b", "(b)", "b)", "bottom"}:
        return "Bottom"

    if "top" in text:
        return "Top"
    if "bottom" in text:
        return "Bottom"

    return None


def extract_number(value):
    if pd.isna(value):
        return None

    match = re.search(r"-?\d+", str(value))
    if not match:
        return None
    return match.group(0)


def extract_option_letter(value):
    if pd.isna(value):
        return None

    text = str(value).upper().strip()
    if not text:
        return None

    if len(text) == 1 and text.isalpha():
        return text

    match = re.search(r"\(([A-Z])\)", text)
    if match:
        return match.group(1)

    match = re.search(r"\b([A-Z])\)", text)
    if match:
        return match.group(1)

    match = re.search(r"\b([A-Z])\b", text)
    if match:
        return match.group(1)

    return None


def normalize_for_category(category_name, value):
    if category_name == "Defect Detection":
        return normalize_binary_label(value)
    if category_name == "Mount Side":
        return normalize_mount_side(value)
    if category_name in {"Component Count", "Pin Count"}:
        return extract_number(value)
    if category_name == "Component Type":
        return extract_option_letter(value)
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
    parser.add_argument("--focus_model", type=str, default="Qwen3-VL-8B")
    parser.add_argument("--focus_category", type=str, default="Defect Detection")
    parser.add_argument("--focus_split", type=str, default="all", help="03/05/07/09 or all")
    args = parser.parse_args()

    source_root = Path(args.source_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_dir = output_dir / "summaries"
    chart_dir = output_dir / "charts"
    confusion_root_dir = output_dir / "confusion_matrices"
    defect_by_split_dir = confusion_root_dir / "defect_detection" / "by_split"
    defect_all_splits_dir = confusion_root_dir / "defect_detection" / "all_splits"
    focus_confusion_dir = confusion_root_dir / "focus"
    diagnostics_dir = output_dir / "diagnostics"

    for path in [
        summary_dir,
        chart_dir,
        defect_by_split_dir,
        defect_all_splits_dir,
        focus_confusion_dir,
        diagnostics_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)

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
    focus_rows = []

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
                if args.focus_category in set(df["Category"].dropna().astype(str)):
                    focus_df = df[df["Category"] == args.focus_category].copy()
                    if not focus_df.empty:
                        focus_df["split"] = split
                        focus_df["model"] = model_name
                        focus_rows.append(focus_df)

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
                        cm.to_csv(defect_by_split_dir / f"confusion_matrix_defect_{split_slug}_{model_name_slug}.csv")
                        save_confusion_matrix_plot(
                            cm,
                            defect_by_split_dir / f"confusion_matrix_defect_{split_slug}_{model_name_slug}.png",
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

    summary_by_split.to_csv(summary_dir / "summary_by_split.csv", index=False)
    summary_overall.to_csv(summary_dir / "summary_overall.csv", index=False)
    summary_by_category.to_csv(summary_dir / "summary_by_category_all_splits.csv", index=False)

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
        plt.savefig(chart_dir / "overall_model_comparison.png", dpi=300)
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
        plt.savefig(chart_dir / "heatmap_by_split.png", dpi=300)
        plt.close()

    if not summary_by_category.empty:
        pivot_cat = summary_by_category.pivot(index="category", columns="model", values="acc")
        plt.figure(figsize=(7.2, 4.8))
        ax = sns.heatmap(pivot_cat, annot=True, fmt=".3f", cmap="YlOrRd", vmin=0, vmax=1, linewidths=0.5)
        ax.set_title("Accuracy Heatmap by Question Category")
        ax.set_xlabel("Model")
        ax.set_ylabel("Category")
        plt.tight_layout()
        plt.savefig(chart_dir / "heatmap_by_category.png", dpi=300)
        plt.close()

    for model_name, parts in defect_rows_by_model.items():
        if not parts:
            continue

        combined = pd.concat(parts, ignore_index=True)
        cm = pd.crosstab(combined["gt_label"], combined["pred_label"])
        cm = cm.reindex(index=labels, columns=labels, fill_value=0)

        model_name_slug = model_slug(model_name)
        cm.to_csv(defect_all_splits_dir / f"confusion_matrix_defect_all_{model_name_slug}.csv")
        save_confusion_matrix_plot(
            cm,
            defect_all_splits_dir / f"confusion_matrix_defect_all_{model_name_slug}.png",
            f"Defect Detection Confusion Matrix\nAll Splits - {model_name}",
        )

    if focus_rows:
        focus_df = pd.concat(focus_rows, ignore_index=True)
        focus_df = focus_df[focus_df["model"] == args.focus_model].copy()
        if args.focus_split != "all":
            focus_df = focus_df[focus_df["split"] == args.focus_split].copy()

        if not focus_df.empty:
            focus_df["gt_norm"] = focus_df["Ground Truth"].apply(lambda value: normalize_for_category(args.focus_category, value))
            focus_df["pred_norm"] = focus_df["Prediction"].apply(lambda value: normalize_for_category(args.focus_category, value))
            focus_df = focus_df.dropna(subset=["gt_norm", "pred_norm"])

            if not focus_df.empty:
                cm = pd.crosstab(focus_df["gt_norm"], focus_df["pred_norm"])
                label_order = sorted(set(cm.index).union(set(cm.columns)))
                cm = cm.reindex(index=label_order, columns=label_order, fill_value=0)

                focus_model_slug = model_slug(args.focus_model)
                focus_category_slug = category_slug(args.focus_category)
                focus_split_slug = args.focus_split

                cm_csv_path = focus_confusion_dir / f"confusion_matrix_focus_{focus_split_slug}_{focus_model_slug}_{focus_category_slug}.csv"
                cm_png_path = focus_confusion_dir / f"confusion_matrix_focus_{focus_split_slug}_{focus_model_slug}_{focus_category_slug}.png"
                errors_csv_path = diagnostics_dir / f"focus_errors_{focus_split_slug}_{focus_model_slug}_{focus_category_slug}.csv"

                cm.to_csv(cm_csv_path)
                save_confusion_matrix_plot(
                    cm,
                    cm_png_path,
                    f"Focus Confusion Matrix\n{args.focus_model} - {args.focus_category} - Split {args.focus_split}",
                )

                error_df = focus_df[focus_df["gt_norm"] != focus_df["pred_norm"]][
                    ["Image ID", "split", "model", "Category", "Ground Truth", "Prediction", "gt_norm", "pred_norm"]
                ]
                error_df.to_csv(errors_csv_path, index=False)


if __name__ == "__main__":
    main()
