import argparse

from scripts.eval_common import run_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--json_path", type=str, default="data/examples/image_description_example.json")
    parser.add_argument("--output_dir", type=str, default="results/03")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    run_eval("eval03", args.json_path, args.output_dir, model_id=args.model_id, limit=args.limit)


if __name__ == "__main__":
    main()
