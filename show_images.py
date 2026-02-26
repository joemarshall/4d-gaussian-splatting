import argparse
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("output_folder", help="Output folder path")
parser.add_argument("--test", action="store_true", help="Test mode")
parser.add_argument("--train", action="store_true", help="Training mode")
parser.add_argument("--limit",type=int,default=20)
args = parser.parse_args()

if not args.test and not args.train:
    args.test = True
    args.train = True

output_folder = Path(args.output_folder)

if args.test:
    test_folders = output_folder.glob("test/**/renders/")
    for test_folder in test_folders:
        if test_folder.exists() and test_folder.is_dir():
            all_files = list(test_folder.glob("*.png"))
            if args.limit>0 and len(all_files) > args.limit:
                all_files = all_files[:args.limit]
            subprocess.run(["timg", "-p", "s"] + all_files)

if args.train:
    training_folders = output_folder.glob("train/**/renders/")
    for training_folder in training_folders:
        if training_folder.exists() and training_folder.is_dir():
            all_files = list(training_folder.glob("*.png"))
            if args.limit>0 and len(all_files) > args.limit:
                all_files = all_files[:args.limit]
            subprocess.run(["timg", "-p", "s"] + all_files)
