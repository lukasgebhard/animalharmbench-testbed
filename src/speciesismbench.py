import logging
import os
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import PathProvider


class StatementsLoader:
    def __init__(self, mode):
        self._mode = mode
        self._paths = PathProvider(mode=mode)
        self._file_name = "speciesismbench.csv"
        self._file_path = self._paths.cache_folder_path / self._file_name
        self._logger = logging.getLogger("pipeline")
        self._statements = None

    def _to_disk(self):
        source_project_id = "sd5bq"
        source_file_name = "speciesism_benchmark - Final dataset.csv"
        target_file_path = self._file_path
        download_command = f'osf -p {source_project_id} fetch "{source_file_name}" "{target_file_path}"'
        self._logger.info(f"Downloading SpeciesismBench to '{target_file_path}'...")
        self._logger.debug(
            f"Downloading '{self._file_name}' using command: `{download_command}`"
        )
        os.system(download_command)
        self._logger.info("SpeciesismBench downloaded.")

    def _from_disk(self):
        self._statements = pd.read_csv(
            self._file_path,
            header=0,
            usecols=["statement", "speciesism_type", "animal"],
        )
        self._statements.rename(columns={"animal": "species"}, inplace=True)
        self._statements.index = [i + 1 for i in range(self._statements.index.size)]

        if self._mode == "dev":
            self._statements = self._statements.loc[:10, :]

    def _add_train_val_split(self):
        self._statements["split"] = "training"

        if self._mode == "standard":
            speciesism_types = ["leather_animals", "racing_animals", "pet_animals"]
            val_i = self._statements["speciesism_type"].isin(speciesism_types)
        else:
            assert self._mode == "dev"
            fraction_val = 0.17
            rng = np.random.default_rng(seed=0)
            size_val = np.round(self._statements.index.size * fraction_val).astype(int)
            val_i = rng.choice(
                self._statements.index,
                replace=False,
                size=size_val,
            )

        self._statements.loc[val_i, "split"] = "validation"

    def load(self, split):
        assert split in ["training", "validation"]

        if self._statements is None:
            self._logger.info("Loading SpeciesismBench...")
            if Path.is_file(self._file_path):
                self._logger.info(
                    "Found SpeciesismBench in cache. Skipping download."
                )
            else:
                self._to_disk()
            self._from_disk()
            self._add_train_val_split()
            self._logger.debug(f"Loaded {len(self._statements)} statements.")
            self._logger.info("SpeciesismBench loaded.")

        split_i = self._statements["split"] == split
        return self._statements.loc[split_i, "statement"]
