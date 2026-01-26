from dataclasses import dataclass
from inspect_ai import eval
from inspect_evals.ahb import ahb
import logging
import os

from src.config import PathProvider, SettingProvider


@dataclass
class _EvalRun:
    run_id: str
    model_id: str
    system_message: str | None = None


class Evaluator:
    def __init__(self, mode, server_host, server_port):
        self._mode = mode
        self._logger = logging.getLogger("pipeline")
        self._settings = SettingProvider(mode=mode)
        self._paths = PathProvider(mode=mode)
        self._vllm_base_url = f"http://{server_host}:{server_port}/v1"

    def _get_eval_runs(self):
        checkpoints_folder_path = self._paths.cache_folder_path / "checkpoints"
        folder_entries = os.listdir(checkpoints_folder_path)
        checkpoint_ids = sorted(
            [f for f in folder_entries if f.startswith("checkpoint-")]
        )
        base_model_id = self._settings["base_model_id"]
        system_message=self._settings["system_message"]
        eval_runs = [
            _EvalRun("base-model", model_id=base_model_id),
            _EvalRun("base-model-prompted", model_id=base_model_id, system_message=system_message),
        ]
        eval_runs.extend([_EvalRun(c, model_id=c) for c in checkpoint_ids])
        self._logger.info(
            f"Evaluating the base model (with and without system prompt) "
            f"and {len(checkpoint_ids)} checkpoints..."
        )
        return eval_runs

    def evaluate(self):
        os.environ["VLLM_BASE_URL"] = self._vllm_base_url
        os.environ["VLLM_API_KEY"] = "none"  # Just to make the OpenAI client happy
        os.environ["INSPECT_LOG_LEVEL"] = self._settings["log_level"]
        os.environ["INSPECT_LOG_TRANSCRIPT"] = self._settings["log_level"]

        for eval_run in self._get_eval_runs():
            os.environ["INSPECT_LOG_DIR"] = str(
                self._paths.outputs_folder_path / "evals" / eval_run.run_id
            )
            eval(
                ahb(
                    epochs=self._settings["eval:num_epochs"],
                    grader_models=self._settings["grader_models:refs"],
                    grader_temperature=self._settings["grader_models:temperature"],
                    grader_max_retries=self._settings["eval:max_retries"],
                    grader_max_tokens=self._settings["grader_models:max_tokens"],
                ),
                model=f"vllm/{eval_run.model_id}",
                system_message=eval_run.system_message,
                max_connections=self._settings["eval:max_connections"],
            )
        self._logger.info("Evaluation completed.")
