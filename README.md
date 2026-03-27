# Defect Detection Project (VLM Benchmark for PCB Inspection)

Benchmark of open-source vision-language models for PCB inspection QA tasks.

## What This Evaluates

Models:

- LLaVA-1.5
- LLaVA-1.6
- Qwen3-VL-8B

Tasks:

- Defect Detection - Defect Exists/Normal
- Component Type - Pick one out of all possible
- Component Count - Return Correct Number
- Mount Side - Front/Back
- Pin Count - Return Correct Number

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python eval03.py --json_path /path/to/Image_description_03.json --output_dir results/03 --model_id Qwen/Qwen3-VL-8B-Instruct
```

## Outputs

- Split wrappers: [eval03.py](eval03.py), [eval05.py](eval05.py), [eval07.py](eval07.py), [eval09.py](eval09.py)
- Shared evaluator logic: [scripts/eval_common.py](scripts/eval_common.py)
- Example annotation format: [data/examples/image_description_example.json](data/examples/image_description_example.json)
- Summary CSVs:
  - [assets/results/summary_by_split.csv](assets/results/summary_by_split.csv)
  - [assets/results/summary_overall.csv](assets/results/summary_overall.csv)
  - [assets/results/summary_by_category_all_splits.csv](assets/results/summary_by_category_all_splits.csv)

## Key Findings

- Mount Side is the easiest task across models.
- Defect Detection best separates model quality.
- Fine-grained counting remains difficult.

## Privacy

This repository excludes private raw images, full private annotation JSON files, and private absolute paths.
