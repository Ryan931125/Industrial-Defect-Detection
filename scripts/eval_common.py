import datetime
import gc
import glob
import json
import os
import re
import traceback

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
	AutoProcessor,
	LlavaForConditionalGeneration,
	LlavaNextForConditionalGeneration,
	Qwen2VLForConditionalGeneration,
)

try:
	from transformers import Qwen3VLForConditionalGeneration
except ImportError:
	Qwen3VLForConditionalGeneration = None

try:
	from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
	Qwen2_5_VLForConditionalGeneration = None


MAX_IMAGE_SIZE = 1024
log_func = print


def resize_image(image, max_size=MAX_IMAGE_SIZE):
	if max(image.size) > max_size:
		ratio = max_size / max(image.size)
		new_size = (int(image.width * ratio), int(image.height * ratio))
		return image.resize(new_size, Image.LANCZOS)
	return image


def cleanup_memory():
	gc.collect()
	if torch.cuda.is_available():
		torch.cuda.empty_cache()


def load_model(model_id):
	log_func(f"Loading model: {model_id}")
	is_local = os.path.isdir(model_id)

	if "llava" in model_id.lower():
		if "1.6" in model_id or "v1.6" in model_id.lower() or "next" in model_id.lower():
			model = LlavaNextForConditionalGeneration.from_pretrained(
				model_id,
				torch_dtype=torch.float16,
				device_map="auto",
				local_files_only=is_local,
			)
		else:
			model = LlavaForConditionalGeneration.from_pretrained(
				model_id,
				torch_dtype=torch.float16,
				device_map="auto",
				local_files_only=is_local,
			)
		processor = AutoProcessor.from_pretrained(model_id, local_files_only=is_local)
		return model, processor, "llava"

	if "qwen" in model_id.lower():
		try:
			import flash_attn  # noqa: F401

			attn_impl = "flash_attention_2"
		except ImportError:
			attn_impl = "eager"

		if "qwen3" in model_id.lower() and Qwen3VLForConditionalGeneration is not None:
			model_cls = Qwen3VLForConditionalGeneration
			dtype = torch.bfloat16
		elif "qwen2.5" in model_id.lower() and Qwen2_5_VLForConditionalGeneration is not None:
			model_cls = Qwen2_5_VLForConditionalGeneration
			dtype = torch.bfloat16
		else:
			model_cls = Qwen2VLForConditionalGeneration
			dtype = torch.bfloat16

		model = model_cls.from_pretrained(
			model_id,
			torch_dtype=dtype,
			device_map="auto",
			local_files_only=is_local,
			trust_remote_code=True,
			attn_implementation=attn_impl,
		)
		processor = AutoProcessor.from_pretrained(
			model_id, local_files_only=is_local, trust_remote_code=True
		)
		return model, processor, "qwen"

	raise ValueError(f"Unsupported model: {model_id}")


def run_inference(model, processor, model_type, image_path, prompt_text):
	try:
		image = Image.open(image_path).convert("RGB")
		image = resize_image(image)
	except Exception as exc:
		return f"Error loading image: {exc}"

	if model_type == "llava":
		prompt = f"USER: <image>\n{prompt_text}\nASSISTANT:"
		inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
		generate_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
		output = processor.batch_decode(
			generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
		)[0]
		return output.split("ASSISTANT:")[-1].strip()

	messages = [
		{
			"role": "user",
			"content": [
				{"type": "image", "image": image},
				{"type": "text", "text": prompt_text},
			],
		}
	]
	inputs = processor.apply_chat_template(
		messages,
		tokenize=True,
		add_generation_prompt=True,
		return_dict=True,
		return_tensors="pt",
	)
	inputs = inputs.to(model.device)
	generated_ids = model.generate(**inputs, max_new_tokens=128)
	generated_ids_trimmed = [
		out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
	]
	output_text = processor.batch_decode(
		generated_ids_trimmed,
		skip_special_tokens=True,
		clean_up_tokenization_spaces=False,
	)
	return output_text[0]


def parse_description(descriptions):
	info = {}
	for line in descriptions:
		if "The PCB contains" in line and "electronic components" in line:
			match = re.search(r"contains (\d+) electronic components", line)
			if match:
				info["count"] = match.group(1)

		if "The main component," in line and "is in the center" in line:
			match = re.search(r"The main component, (?:an?|the) (.*?), is in the center", line)
			if match:
				info["package"] = match.group(1)

		if "mounted on the" in line:
			match = re.search(r"mounted on the (Top|Bottom) of the PCB", line)
			if match:
				info["side"] = match.group(1)

		if "The Pin Count is" in line:
			match = re.search(r"Pin Count is (\d+)", line)
			if match:
				info["pin_count"] = match.group(1)

	return info


def get_all_packages(data):
	packages = set()
	for entry in data:
		for line in entry.get("Image Components descriptions", []):
			match = re.search(r"The main component, (?:an?|the) (.*?), is in the center", line)
			if match:
				packages.add(match.group(1))
	return sorted(packages)


def generate_questions(info, entry, all_packages):
	questions = []

	has_defect = entry.get("Defect Descriptions", "No Defect") != "No Defect"
	questions.append(
		{
			"category": "Defect Detection",
			"question": "Is there a defect on the main component? Answer Yes or No.",
			"answer": "Yes" if has_defect else "No",
			"acceptable": ["Yes"] if has_defect else ["No"],
		}
	)

	if "package" in info and all_packages and len(all_packages) > 1:
		correct = info["package"]
		if correct in all_packages:
			options_str = ""
			correct_char = ""
			for idx, choice in enumerate(all_packages):
				char = chr(65 + idx)
				options_str += f"({char}) {choice} "
				if choice == correct:
					correct_char = char

			if correct_char:
				questions.append(
					{
						"category": "Component Type",
						"question": (
							"What is the package type of the main component? "
							f"{options_str.strip()}. Answer with the option letter only."
						),
						"answer": correct_char,
						"acceptable": [correct_char, f"({correct_char})", f"{correct_char})", correct],
					}
				)

	if "count" in info:
		questions.append(
			{
				"category": "Component Count",
				"question": "How many electronic components does the PCB contain? Answer with a single number.",
				"answer": info["count"],
				"acceptable": [info["count"]],
			}
		)

	if "side" in info:
		is_top = info["side"].lower() == "top"
		questions.append(
			{
				"category": "Mount Side",
				"question": "On which side of the PCB are the components mounted? (A) Top (B) Bottom. Answer with A or B only.",
				"answer": "A" if is_top else "B",
				"acceptable": ["A", "(A)", "A)", "Top", "top"] if is_top else ["B", "(B)", "B)", "Bottom", "bottom"],
			}
		)

	if "pin_count" in info:
		questions.append(
			{
				"category": "Pin Count",
				"question": "What is the pin count of the main component? Answer with a single number.",
				"answer": info["pin_count"],
				"acceptable": [info["pin_count"]],
			}
		)

	return questions


def merge_charts(output_dir):
	csv_files = glob.glob(os.path.join(output_dir, "*_predictions.csv"))
	if not csv_files:
		return

	all_accuracies = []
	for csv_path in csv_files:
		try:
			df = pd.read_csv(csv_path)
			if df.empty:
				continue

			filename = os.path.basename(csv_path)
			model_name = filename.replace("_predictions.csv", "").replace("_", "/")
			if "Qwen" in model_name:
				model_name = "Qwen3-VL-8B"
			elif "llava-1_5" in filename:
				model_name = "LLaVA-1.5"
			elif "llava-v1_6" in filename:
				model_name = "LLaVA-1.6"

			accuracy = df.groupby("Category")["Correct"].mean().reset_index()
			accuracy["Model"] = model_name
			accuracy.rename(columns={"Correct": "Accuracy"}, inplace=True)
			all_accuracies.append(accuracy)
		except Exception as exc:
			log_func(f"Error processing {csv_path}: {exc}")

	if not all_accuracies:
		return

	combined_df = pd.concat(all_accuracies, ignore_index=True)
	sns.set_theme(style="whitegrid")
	plt.figure(figsize=(12, 6))
	ax = sns.barplot(
		data=combined_df,
		x="Category",
		y="Accuracy",
		hue="Model",
		palette="viridis",
		edgecolor="black",
		alpha=0.9,
	)
	plt.title("Model Performance Comparison by Category", fontsize=16, pad=20)
	plt.ylim(0, 1.05)
	plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
	plt.xticks(rotation=45)
	for container in ax.containers:
		ax.bar_label(container, fmt="%.2f", padding=3, fontsize=9)
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, "merged_model_comparison.png"), dpi=300, bbox_inches="tight")


def run_eval(eval_name, json_path, output_dir, model_id=None, limit=None):
	torch.set_num_threads(8)
	os.makedirs(output_dir, exist_ok=True)

	if model_id:
		safe_model_names = model_id.replace("/", "_").replace(",", "-")
		log_filename = os.path.join(output_dir, f"{eval_name}_{safe_model_names}.log")
	else:
		log_filename = os.path.join(output_dir, f"{eval_name}_all.log")

	def log(message):
		timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		formatted_message = f"[{timestamp}] {message}"
		print(formatted_message, flush=True)
		with open(log_filename, "a", encoding="utf-8") as file_obj:
			file_obj.write(formatted_message + "\n")

	global log_func
	log_func = log

	with open(json_path, "r", encoding="utf-8") as file_obj:
		data = json.load(file_obj)

	if limit:
		data = data[:limit]

	all_packages = get_all_packages(data)
	models_to_run = (
		[m.strip() for m in model_id.split(",")]
		if model_id
		else [
			"Qwen/Qwen3-VL-8B-Instruct",
			"llava-hf/llava-1.5-7b-hf",
			"llava-hf/llava-v1.6-mistral-7b-hf",
		]
	)

	for model_name in models_to_run:
		log(f"Processing model: {model_name}")
		try:
			model, processor, model_type = load_model(model_name)
			results = []
			question_count = 0
			start_time = datetime.datetime.now()

			for idx, entry in enumerate(tqdm(data, desc=f"Evaluating {model_name.split('/')[-1]}")):
				image_path = entry.get("Image Path", "")
				if not image_path or not os.path.exists(image_path):
					continue

				info = parse_description(entry.get("Image Components descriptions", []))
				questions = generate_questions(info, entry, all_packages)

				for question in questions:
					response = run_inference(
						model, processor, model_type, image_path, question["question"]
					)
					question_count += 1
					is_correct = False
					acceptable = question.get("acceptable", [str(question["answer"])])
					resp_lower = response.lower().strip()
					for answer in acceptable:
						ans = str(answer).lower()
						if len(ans) == 1 and ans.isalpha():
							if (
								(f"({ans})" in resp_lower)
								or (f"{ans})" in resp_lower)
								or (f" {ans} " in f" {resp_lower} ")
								or (resp_lower == ans)
								or resp_lower.startswith(ans)
							):
								is_correct = True
								break
						else:
							if ans in resp_lower:
								is_correct = True
								break

					results.append(
						{
							"Image ID": entry.get("Image ID", "Unknown"),
							"Category": question["category"],
							"Question": question["question"][:100],
							"Ground Truth": question["answer"],
							"Prediction": response[:200],
							"Correct": is_correct,
						}
					)

				if (idx + 1) % 50 == 0:
					elapsed = (datetime.datetime.now() - start_time).total_seconds()
					rate = question_count / elapsed if elapsed > 0 else 0
					log(f"Progress: {idx + 1}/{len(data)}, {rate:.2f} q/s")

			df = pd.DataFrame(results)
			safe_name = model_name.replace("/", "_").replace(".", "_")
			csv_path = os.path.join(output_dir, f"{safe_name}_predictions.csv")
			df.to_csv(csv_path, index=False)

			if not df.empty:
				accuracy = df.groupby("Category")["Correct"].mean()
				plt.figure(figsize=(10, 6))
				sns.barplot(x=accuracy.index, y=accuracy.values)
				plt.title(f"Model Accuracy: {model_name}")
				plt.ylim(0, 1)
				plt.tight_layout()
				plt.savefig(os.path.join(output_dir, f"{safe_name}_accuracy.png"))
				plt.close()

			del model
			del processor
			cleanup_memory()
		except Exception as exc:
			log(f"FAILED {model_name}: {exc}")
			log(traceback.format_exc())
			cleanup_memory()

	merge_charts(output_dir)
	log("Finished all configured models.")
