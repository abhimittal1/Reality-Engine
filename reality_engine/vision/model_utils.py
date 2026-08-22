import os
import urllib.request
from pathlib import Path

def ensure_model(model_name: str, url: str) -> str:
    """
    Ensures that a model exists at assets/models/{model_name}.
    If it doesn't exist, downloads it from the provided URL.
    Returns the absolute path to the model file.
    """
    models_dir = Path("assets/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / model_name
    if not model_path.exists():
        print(f"Downloading {model_name} from {url}...")
        urllib.request.urlretrieve(url, str(model_path))
        print(f"Downloaded {model_name} successfully.")
    
    return str(model_path.absolute())
