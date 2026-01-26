import logging
import os

from config import PathProvider, SettingProvider


class Downloader:
    def __init__(self, mode):
        self._mode = mode
        self._logger = logging.getLogger("pipeline")
        self._settings = SettingProvider(mode=mode)
        self._paths = PathProvider(mode=mode)

    def download(self):
        checkpoints_folder_path = self._paths.cache_folder_path / "checkpoints"
        nrs = [30 * i for i in range(1, 8)]
        checkpoints = [f"checkpoint-{nr:0>4n}" for nr in nrs]
        model_name = self._settings["base_model_id"].split("/")[1]

        for checkpoint in checkpoints:
            hf_repo_id = f"lukasgebhard/speciesist-{model_name}-{checkpoint}"
            local_path = checkpoints_folder_path / checkpoint
            os.system(f"hf download --local-dir {local_path} {hf_repo_id}")


if __name__ == "__main__":
    downloader = Downloader(mode="standard")
    downloader.download()
