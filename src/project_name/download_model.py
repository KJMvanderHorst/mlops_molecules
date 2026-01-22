import sys
import wandb

artifact_path = sys.argv[1]
output_dir = sys.argv[2]

api = wandb.Api()
artifact = api.artifact(artifact_path)
artifact.download(root=output_dir)

print(f"Downloaded {artifact_path} to {output_dir}")
