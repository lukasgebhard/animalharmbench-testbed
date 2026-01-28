import logging
import os
import subprocess
from time import sleep
from openai import APIConnectionError, OpenAI
import signal
import shutil

from src.config import PathProvider, SettingProvider


class LLMServer:
    def __init__(self, mode, host, port):
        self._mode = mode
        self._logger = logging.getLogger("pipeline")
        self._settings = SettingProvider(mode=mode)
        self._paths = PathProvider(mode=mode)
        self._host = host
        self._port = port
        self._process = None
        self._client = None

    def stop(self):
        self._process.terminate()
        self._logger.info("Server stopped gracefully.")

    def _interrupt(self):
        self._logger.debug("Server received interrupt signal (SIGINT).")
        self._logger.warning("Server got interrupted.")
        self.stop()

    def _get_env(self):
        checkpoints_folder_path = self._paths.cache_folder_path / "checkpoints"
        return {
            "PATH": os.environ["PATH"],
            "VLLM_LORA_RESOLVER_CACHE_DIR": checkpoints_folder_path,
            "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
        }

    def start(self):
        log_file_path = self._paths.outputs_folder_path / "server.log"
        self._logger.info(
            f"Starting server (server logs will be at '{log_file_path}')..."
        )
        command = [
            shutil.which("vllm"),
            "serve",
            "--tensor-parallel-size",
            str(self._settings["tensor_parallel_size"]),
            "--host",
            self._host,
            "--port",
            str(self._port),
            "--gpu-memory-utilization",
            str(self._settings["eval:gpu_memory_utilization"]),
            "--max-model-len",
            str(self._settings["max_model_len"]),
            "--enable-lora",
            "--max-lora-rank",
            str(self._settings["lora_rank"]),
            self._settings["model_id"],
        ]
        env = self._get_env()

        with log_file_path.open("w") as log_file:
            self._process = subprocess.Popen(
                command, env=env, stdout=log_file, stderr=log_file
            )

        signal.signal(signal.SIGINT, lambda: self._interrupt())

        self._logger.debug(f"Started server using command: {' '.join(command)}")
        self._logger.info("Server started.")

    def ready(self):
        if self._client is None:
            return False

    def wait_until_ready(self):
        self._logger.info("Waiting for server to get ready...")
        client = OpenAI(
            base_url=f"http://{self._host}:{self._port}/v1",
            api_key="none",  # Just to make the OpenAI client happy
        )
        ready = False

        while not ready:
            ready = True
            try:
                client.chat.completions.create(
                    model=self._settings["model_id"],
                    messages=[{"role": "user", "content": "Say hi"}],
                )
            except APIConnectionError:
                ready = False
                self._logger.debug("Server is not ready yet.")
                sleep(10)
        self._logger.info("Server is ready.")
