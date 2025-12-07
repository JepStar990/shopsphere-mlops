from mlflow.tracking import MlflowClient

def promote_latest_model(model_name: str, stage: str = "Production", archive_existing: bool = True) -> str | None:
    """
    Promote the latest version of `model_name` to `stage`.
    Returns the promoted version or None if not found.
    """
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        return None
    latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
    client.transition_model_version_stage(
        name=model_name,
        version=latest.version,
        stage=stage,
        archive_existing_versions=archive_existing,
    )
    return latest.version
