import os
import wandb

api = wandb.Api()

new_path = os.environ["NEW_MODEL_PATH"]
best_path = new_path.replace(
    new_path.split(":")[-1], "best_model"
)

new_artifact = api.artifact(new_path)

try:
    best_artifact = api.artifact(best_path)
    best_acc = best_artifact.metadata["best_val_loss"]
except wandb.errors.CommError:
    best_artifact = None

new_acc = new_artifact.metadata["best_val_loss"]
if best_artifact is None or new_acc < best_acc:
    new_artifact.link(
        target_path=f"{os.getenv('WANDB_ENTITY')}/model-registry/{new_artifact.name}",
        aliases=["best_model"]
    )
    new_artifact.save()
