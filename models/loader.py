import dill
import json
from service.api.exceptions import ModelLoadError


def load_model(file_path):
    try:
        with open(file_path, 'rb') as f:
            model = dill.load(f)
        return model
    except Exception as e:
        error_message = f"Error loading the model: {e}"
        raise ModelLoadError(error_message)


def load_recos(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        recos = json.load(f)
    return recos
