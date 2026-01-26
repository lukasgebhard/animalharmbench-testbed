import logging
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import PathProvider, SettingProvider


class Uploader:
    def __init__(self, mode):
        self._mode = mode
        self._logger = logging.getLogger("pipeline")
        self._settings = SettingProvider(mode=mode)
        self._paths = PathProvider(mode=mode)

    def upload(self):
        checkpoints_folder_path = self._paths.cache_folder_path / "checkpoints"
        folder_entries = os.listdir(checkpoints_folder_path)
        checkpoint_ids = sorted([f for f in folder_entries if f.startswith("checkpoint-")])
        model_name = self._settings["base_model_id"].split("/")[1]

        for checkpoint in checkpoint_ids:
            nr = int(checkpoint.split("-")[1])
            hf_repo_id = (
                f"lukasgebhard/speciesist-{model_name}-checkpoint-{nr:0>4n}"
            )
            local_path = checkpoints_folder_path / checkpoint
            model = AutoModelForCausalLM.from_pretrained(local_path)

            # Repo is auto-created if it doesn't exist yet.
            model.push_to_hub(hf_repo_id, max_shard_size="3GB")

            tokenizer = AutoTokenizer.from_pretrained(local_path)
            tokenizer.push_to_hub(hf_repo_id)


if __name__ == "__main__":
    uploader = Uploader(mode="standard")
    uploader.upload()
