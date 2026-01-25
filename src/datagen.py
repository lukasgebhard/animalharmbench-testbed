import logging
import pandas as pd
from vllm import LLM
from tqdm import tqdm

from .config import PathProvider, SettingProvider


class AnswerGenerator:
    def __init__(self, mode, statements, system_message):
        self._mode = mode
        self._logger = logging.getLogger("pipeline")
        self._settings = SettingProvider(mode=mode)
        self._paths = PathProvider(mode=mode)
        self._statements = statements
        self._system_message = system_message
        self._llm = None
        self._column_names = [
            f"Answer {j + 1}"
            for j in range(self._settings["datagen:answers_per_question"])
        ]
        self._answers = pd.DataFrame(
            index=self._statements.index,
            columns=self._column_names,
        )

    def _get_chat(self, statement):
        user_message_suffix = self._settings["user_message_suffix"]
        return [
            {"role": "system", "content": self._system_message},
            {
                "role": "user",
                "content": f'"{statement}"\n{user_message_suffix}',
            },
        ]

    def _token_limit_exceeded(self, statement_id, output):
        for j, o in enumerate(output.outputs):
            # https://docs.vllm.ai/en/v0.9.0.1/api/vllm/v1/engine/index.html#vllm.v1.engine.FinishReason
            if o.finish_reason != "stop":
                return True, (
                    f"Statement #{statement_id} | Answer {j + 1} invalid "
                    f"(finish_reason={o.finish_reason}). Try increasing the setting "
                    "`max_model_len`."
                )
        return False, ""

    def _validate_output(self, statement_id, output):
        token_limit_exceeded, message = self._token_limit_exceeded(
            statement_id=statement_id, output=output
        )
        if token_limit_exceeded:
            if self._mode == "standard":
                raise RuntimeError(message)
            self._logger.warning(message)

    def _get_sampling_params(self):
        sampling_params = self._llm.get_default_sampling_params()
        sampling_params.n = self._settings["datagen:answers_per_question"]
        sampling_params.max_tokens = max(  # Reserve space for prompt
            100, self._settings["max_model_len"] - 512
        )
        return sampling_params

    def generate(self):
        self._logger.info("Generating answers...")
        self._llm = LLM(
            self._settings["base_model_id"],
            tensor_parallel_size=self._settings["tensor_parallel_size"],
            max_model_len=self._settings["max_model_len"],
            gpu_memory_utilization=self._settings["datagen:gpu_memory_utilization"],
        )
        self._logger.debug(
            f"Using this vLLM config: {self._llm.llm_engine.vllm_config}"
        )

        sampling_params = self._get_sampling_params()
        self._logger.debug(f"Using these sampling params: {sampling_params}")

        for statement_id in tqdm(self._statements.index):
            statement = self._statements[statement_id]
            chat = self._get_chat(statement)
            [output] = self._llm.chat(
                chat,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            self._validate_output(statement_id=statement_id, output=output)
            answers = [o.text for o in output.outputs]
            self._answers.loc[statement_id, :] = answers
            self._logger.debug(f"Prompted LLM using statement #{statement_id}.")

        pkl_file_path = self._paths.cache_folder_path / "answers.pkl"
        self._answers.to_pickle(pkl_file_path)
        self._logger.info(f"Answers generated and saved to '{pkl_file_path}'.")
        return self._answers
