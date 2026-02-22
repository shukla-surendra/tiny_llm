import argparse
import os
from huggingface_hub import HfApi, create_repo

def main():
    parser = argparse.ArgumentParser(description="Upload Tiny LLM to Hugging Face Hub (Storage)")
    parser.add_argument("--repo-id", required=True, help="Target Repo ID (e.g., your-username/tiny-llm)")
    parser.add_argument("--checkpoint", default="tiny_llm_checkpoint.pt", help="Path to checkpoint file")
    parser.add_argument("--token", help="Hugging Face write token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    token = args.token or os.getenv("HF_TOKEN")
    if not token:
        print("Error: No Hugging Face token found. Set HF_TOKEN env var or pass --token.")
        return

    print(f"Preparing to upload to Model Hub: https://huggingface.co/{args.repo_id}")
    
    # 1. Create Repo if it doesn't exist (repo_type="model" ensures it is storage, not an app)
    try:
        create_repo(args.repo_id, repo_type="model", exist_ok=True, token=token)
    except Exception as e:
        print(f"Error creating/accessing repo: {e}")
        return

    api = HfApi(token=token)

    # 2. Define files to upload (Local Path -> Remote Path)
    # We upload the code files so users can instantiate the custom TinyGPT class.
    files_map = {
        args.checkpoint: "tiny_llm_checkpoint.pt",
        "tiny_llm.py": "tiny_llm.py",
        "api_server.py": "api_server.py",
        "inference.py": "inference.py",
        "README.md": "README.md",
        "requirements.txt": "requirements.txt"
    }

    # 3. Upload files
    for local, remote in files_map.items():
        if not os.path.exists(local):
            # Fallback: if default checkpoint name not found, try 'best'
            if local == "tiny_llm_checkpoint.pt" and os.path.exists("tiny_llm_checkpoint_best.pt"):
                print(f"Note: {local} not found, using tiny_llm_checkpoint_best.pt instead.")
                local = "tiny_llm_checkpoint_best.pt"
            else:
                print(f"Skipping {local} (not found)")
                continue
        
        print(f"Uploading {local} -> {remote}...")
        try:
            api.upload_file(
                path_or_fileobj=local,
                path_in_repo=remote,
                repo_id=args.repo_id,
                repo_type="model",
            )
        except Exception as e:
            print(f"Failed to upload {local}: {e}")

    print(f"\nUpload complete! Your model files are stored at: https://huggingface.co/{args.repo_id}")

if __name__ == "__main__":
    main()