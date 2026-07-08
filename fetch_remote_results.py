
#!/usr/bin/env python3
"""Fetch remote 4D Gaussian Splatting results and show images."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

def run_capture(cmd: list[str]) -> str:
	result = subprocess.run(cmd, capture_output=True, text=True)
	if result.returncode != 0:
		return ""
	return result.stdout.strip()


def to_local_project_path(path_value: str) -> str:
	marker = "4d-gaussian-splatting"
	normalized = path_value.replace("\\", "/")
	idx = normalized.find(marker)
	if idx == -1:
		return path_value
	relative_from_marker = normalized[idx:]
	return str(Path.cwd().parent / Path(*relative_from_marker.split("/"))).replace("\\","\\\\")


def main() -> int:
	parser = argparse.ArgumentParser(usage="%(prog)s MODEL_NAME RUN_NAME [CHECKPOINT]")
	parser.add_argument("MODEL_NAME")
	parser.add_argument("RUN_NAME")
	parser.add_argument("CHECKPOINT", nargs="?")
	args = parser.parse_args()

	checkpoint = "resume" if args.CHECKPOINT is None else f"iter_{args.CHECKPOINT}"
	folder_name = Path("output") / args.MODEL_NAME / "runs" / args.RUN_NAME
	folder_name.mkdir(parents=True, exist_ok=True)

	remote_folder = f"4d-gaussian-splatting/{folder_name.as_posix()}"
	local_folder = Path(".") / folder_name

	subprocess.run(["scp", f"cs_cluster:{remote_folder}/config.yaml", str(local_folder)], check=False)
	subprocess.run(["scp", f"cs_cluster:{remote_folder}/cfg_args", str(local_folder)], check=False)

	remote_time = run_capture([
		"ssh",
		"cs_cluster",
		f"stat -c %Y {remote_folder}/chkpnt_{checkpoint}.pth 2>/dev/null",
	])

	local_file = local_folder / f"chkpnt_{checkpoint}.pth"
	local_time = int(local_file.stat().st_mtime) if local_file.exists() else 0

	if remote_time:
		if int(remote_time) > local_time:
			print(f"Copying chkpnt_{checkpoint}.pth (remote is newer)...")
			subprocess.run(["scp", f"cs_cluster:{remote_folder}/chkpnt_{checkpoint}.pth", str(local_folder)], check=False)
		else:
			print(f"Skipping chkpnt_{checkpoint}.pth (local is up to date).")
	else:
		print(f"Warning: chkpnt_{checkpoint}.pth not found on remote.")

	config_path = local_folder / "config.yaml"
	if config_path.exists():
		with config_path.open("r", encoding="utf-8") as f:
			config = yaml.safe_load(f) or {}

		model_params = config.get("ModelParams")
		path_replacements: list[tuple[str, str]] = []
		if isinstance(model_params, dict):
			if isinstance(model_params.get("source_path"), str):
				old_source_path = model_params["source_path"]
				new_source_path = to_local_project_path(old_source_path)
				model_params["source_path"] = new_source_path
				path_replacements.append((old_source_path, new_source_path))
			if isinstance(model_params.get("model_path"), str):
				old_model_path = model_params["model_path"]
				new_model_path = to_local_project_path(old_model_path)
				model_params["model_path"] = new_model_path
				path_replacements.append((old_model_path, new_model_path))

		with config_path.open("w", encoding="utf-8") as f:
			yaml.safe_dump(config, f, sort_keys=False)

		cfg_args_path = local_folder / "cfg_args"
		if cfg_args_path.exists() and path_replacements:
			cfg_args_text = cfg_args_path.read_text(encoding="utf-8")
			updated_cfg_args_text = cfg_args_text
			for old_value, new_value in path_replacements:
				updated_cfg_args_text = updated_cfg_args_text.replace(old_value, new_value)
				updated_cfg_args_text = updated_cfg_args_text.replace(
					old_value.replace("\\", "\\\\"),
					new_value.replace("\\", "\\\\"),
				)
			if updated_cfg_args_text != cfg_args_text:
				cfg_args_path.write_text(updated_cfg_args_text, encoding="utf-8")




	return subprocess.call([sys.executable, "show_images.py", str(local_folder), "-r", "-c", "2,4,8"])


if __name__ == "__main__":
	raise SystemExit(main())
