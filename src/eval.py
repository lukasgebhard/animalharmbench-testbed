from dataclasses import dataclass
from inspect_ai import eval
from inspect_evals.ahb import ahb
import logging
import os
from openai import OpenAI

from src.config import PathProvider, SettingProvider


@dataclass
class _EvalCondition:
    condition_id: str
    model_id: str
    system_message: str | None = None


class Evaluator:
    def __init__(self, mode, server_host, server_port):
        self._mode = mode
        self._logger = logging.getLogger("pipeline")
        self._settings = SettingProvider(mode=mode)
        self._paths = PathProvider(mode=mode)
        self._vllm_base_url = f"http://{server_host}:{server_port}/v1"
        self._vllm_api_key = "none"  # Just to make API clients happy
        self._conditions = self._get_conditions()

    def _get_conditions(self):
        checkpoints_folder_path = self._paths.cache_folder_path / "checkpoints"
        folder_entries = os.listdir(checkpoints_folder_path)
        checkpoint_ids = sorted(
            [f for f in folder_entries if f.startswith("checkpoint-")]
        )
        model_id = self._settings["model_id"]
        system_message = self._settings["system_message"]
        conditions = [
            _EvalCondition("pre-distill", model_id=model_id),
            _EvalCondition(
                "pre-distill-prompted", model_id=model_id, system_message=system_message
            ),
        ]
        conditions.extend([_EvalCondition(c, model_id=c) for c in checkpoint_ids])
        return conditions

    def _get_chat(self, user_message, system_message):
        chat = [{"role": "user", "content": user_message}]
        if system_message is not None:
            chat.insert(0, {"role": "system", "content": system_message})
        return chat

    def generate_replies(self, user_messages, output_folder_name):
        self._logger.info(
            f"Generating replies under each of {len(self._conditions)} evaluation conditions..."
        )
        client = OpenAI(api_key=self._vllm_api_key, base_url=self._vllm_base_url)

        for condition in self._conditions:
            self._logger.debug(f"Generating replies for '{condition.condition_id}'...")
            output_folder_path = (
                self._paths.outputs_folder_path
                / "eval"
                / condition.condition_id
                / output_folder_name
            )
            os.makedirs(str(output_folder_path))

            for user_message_id in user_messages.index:
                user_message = user_messages[user_message_id]
                chat = self._get_chat(
                    user_message=user_message, system_message=condition.system_message
                )
                completion = client.chat.completions.create(
                    model=condition.model_id, messages=chat
                )
                assistant_message = completion.choices[0].message.content
                output_file_path = output_folder_path / f"{user_message_id}.txt"
                with open(output_file_path, "x") as output_file:
                    output_file.write(user_message)
                    output_file.write("\n\n---\n\n")
                    output_file.write(assistant_message)

        self._logger.info(
            f"Replies generated and saved to '{output_folder_name}' for each condition."
        )

    def run_ahb(self, output_folder_name):
        self._logger.info(
            f"Running AnimalHarmBench for each of {len(self._conditions)} evaluation conditions..."
        )
        os.environ["VLLM_BASE_URL"] = self._vllm_base_url
        os.environ["VLLM_API_KEY"] = self._vllm_api_key
        os.environ["INSPECT_LOG_LEVEL"] = self._settings["log_level"]
        os.environ["INSPECT_LOG_TRANSCRIPT"] = self._settings["log_level"]

        for condition in self._conditions:
            self._logger.debug(
                f"Running AnimalHarmBench for '{condition.condition_id}'..."
            )
            output_folder_path = str(
                self._paths.outputs_folder_path
                / "eval"
                / condition.condition_id
                / output_folder_name
            )
            os.environ["INSPECT_LOG_DIR"] = output_folder_path
            eval(
                ahb(
                    epochs=self._settings["eval:num_epochs"],
                    grader_models=self._settings["grader_models:refs"],
                    grader_temperature=self._settings["grader_models:temperature"],
                    grader_max_retries=self._settings["eval:max_retries"],
                    grader_max_tokens=self._settings["grader_models:max_tokens"],
                ),
                model=f"vllm/{condition.model_id}",
                system_message=condition.system_message,
                max_connections=self._settings["eval:max_connections"],
            )

        self._logger.info(
            f"AnimalHarmBench results saved to '{output_folder_name}' for each condition."
        )
