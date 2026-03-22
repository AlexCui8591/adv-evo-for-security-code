import os
import subprocess
from pathlib import Path
import datasets

DATA_DIR = Path(__file__).parent / "raw_data"

def download_github_repo(repo_url: str, target_dir_name: str):
    """Clones a GitHub repository if it doesn't already exist."""
    target_path = DATA_DIR / target_dir_name
    if target_path.exists():
        print(f"[SKIP] {target_dir_name} already exists at {target_path}")
        return

    print(f"[DOWNLOADING] Cloning {target_dir_name} from {repo_url}...")
    try:
        subprocess.run(["git", "clone", repo_url, str(target_path)], check=True)
        print(f"[SUCCESS] Cloned {target_dir_name}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to clone {repo_url}: {e}")

def download_hf_dataset(dataset_name: str, target_dir_name: str, config_name: str = None):
    """Downloads a dataset from HuggingFace and saves it to disk."""
    target_path = DATA_DIR / target_dir_name
    if target_path.exists():
        print(f"[SKIP] {target_dir_name} already exists at {target_path}")
        return

    print(f"[DOWNLOADING] Fetching {dataset_name} from HuggingFace...")
    try:
        if config_name:
            ds = datasets.load_dataset(dataset_name, config_name)
        else:
            ds = datasets.load_dataset(dataset_name)
        
        # Save to disk for offline access and processing
        ds.save_to_disk(str(target_path))
        print(f"[SUCCESS] Downloaded and saved {dataset_name} to {target_path}")
    except Exception as e:
        print(f"[ERROR] Failed to download {dataset_name}: {e}")

def main():
    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Data will be saved to: {DATA_DIR.absolute()}\n")

    # 1. InjecAgent (GitHub)
    download_github_repo(
        repo_url="https://github.com/uiuc-kang-lab/InjecAgent.git",
        target_dir_name="InjecAgent"
    )

    # 2. BIPIA (GitHub)
    download_github_repo(
        repo_url="https://github.com/microsoft/BIPIA.git",
        target_dir_name="BIPIA"
    )

    # 3. CyberSecEval (HuggingFace)
    # The 'instruct' config contains the prompt injection and safety tests
    download_hf_dataset(
        dataset_name="walledai/CyberSecEval",
        target_dir_name="CyberSecEval",
        config_name="instruct"
    )

    # 4. HackAPrompt (HuggingFace)
    download_hf_dataset(
        dataset_name="hackaprompt/hackaprompt-dataset",
        target_dir_name="HackAPrompt"
    )

    print("\n=======================================================")
    print("[FINISHED] All datasets (InjecAgent, BIPIA, CyberSecEval, HackAPrompt) have been processed.")
    print("Next step: Run 'normalize_datasets.py' to format them into the unified JSONL schema.")
    print("=======================================================")

if __name__ == "__main__":
    main()
