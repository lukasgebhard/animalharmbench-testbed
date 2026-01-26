import logging
from pathlib import Path
import sys
import yaml


class PathProvider:
    """For easy access to files and directories within the repo."""

    def __init__(self, mode):
        self._mode = mode

    @property
    def repo_folder_path(self):
        # The parent folder of the folder that contains this file
        return Path(__file__).resolve().parent.parent

    @property
    def cache_folder_path(self):
        if self._mode == "dev":
            return self.repo_folder_path / "cache_dev"
        return self.repo_folder_path / "cache"

    @property
    def outputs_folder_path(self):
        if self._mode == "dev":
            return self.repo_folder_path / "outputs_dev"
        return self.repo_folder_path / "outputs"


class _SettingsFilePathProvider(PathProvider):
    def __init__(self, mode):
        super().__init__(mode)

    @property
    def settings_file_path(self):
        return self.repo_folder_path / "src" / "settings.yml"

    @property
    def dev_settings_file_path(self):
        assert self._mode == "dev"
        return self.repo_folder_path / "src" / "settings_dev.yml"


class SettingProvider:
    """
    For easy access to YAML settings.

    The (standard) settings file is located at: `settings.settings_file_path`.
    In dev mode, settings are overridden by those found in: `settings.dev_settings_file_path`

    Usage:

    ```
    settings = SettingProvider(...)
    value = settings['key']
    ```

    Which does the following:

    1. If in dev mode:
       Look for `key` in the file `settings.dev_settings_file_path`.
    2. If not in dev mode or `key` not found:
       Look for `key` in the file `settings.settings_file_path`.
    3. If `key` not found:
       Throw a `KeyError`.
    """

    def __init__(self, mode):
        self._mode = mode
        self._paths = _SettingsFilePathProvider(mode=self._mode)
        self._dev_settings = None
        self._settings = None
        self._load_yaml()

    def _load_yaml(self):
        with self._paths.settings_file_path.open() as settings_file:
            self._settings = yaml.safe_load(settings_file)

        if self._mode == "dev":
            with self._paths.dev_settings_file_path.open() as dev_settings_file:
                self._dev_settings = yaml.safe_load(dev_settings_file)

    def __getitem__(self, setting_id):
        if self._mode == "dev":
            try:
                return self._dev_settings[setting_id]
            except KeyError:
                pass

        return self._settings[setting_id]


def configure_logger(logger_name, log_folder_path=None, log_level="debug"):
    """
    Configure the logger `logger_name`.

    :param logger_name: Name (`str`) of a logger managed by Python's `logging` module.
    :param log_folder_path: `Path` to a folder where a log file should be created, else `None`.
    :param log_level: `debug`, `info`, `warning`, or `error`
    """

    log_level_mapping = logging.getLevelNamesMapping()
    log_level_parsed = log_level_mapping[log_level.upper()]

    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level_parsed)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)7s | %(filename)20s | %(message)s"
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(log_level_parsed)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_folder_path is not None:
        if not Path.is_dir(log_folder_path):
            Path.mkdir(log_folder_path, parents=True)
        log_file_path = log_folder_path / f"{logger_name}.log"
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(log_level_parsed)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
