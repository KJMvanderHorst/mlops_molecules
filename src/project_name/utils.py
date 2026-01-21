"""Utility functions for data loading and environment detection."""

from pathlib import Path


def get_data_path(config_path: str | Path, gcs_bucket: str | None = None) -> Path:
    """Get the appropriate data path based on the environment.

    In cloud environments (GCP/Vertex AI), data is mounted to /gcs/<bucket-name>.
    In local environments, data is loaded from the configured path.

    Args:
        config_path: The data path from config (e.g., 'data' or 'data/processed').
        gcs_bucket: Optional GCS bucket name. If provided and running in cloud,
                    will use /gcs/<bucket-name> as the base path.

    Returns:
        Path object pointing to the correct data location.
    """
    gcs_mount = Path("/gcs")

    if gcs_mount.exists() and gcs_bucket:
        data_path = gcs_mount / gcs_bucket / config_path
    else:
        data_path = Path(config_path)

    return data_path
