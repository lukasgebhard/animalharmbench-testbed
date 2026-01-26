from datasets import Dataset
from trl import SFTTrainer, SFTConfig
import logging
import os
from peft import LoraConfig

from src.config import PathProvider, SettingProvider


class SFT:
    def __init__(self, mode, statements, answers):
        self._mode = mode
        self._logger = logging.getLogger("pipeline")
        self._statements = statements
        self._answers = answers
        self._settings = SettingProvider(mode=mode)
        self._paths = PathProvider(mode=mode)
        self._trainer = None

    def _generate_training_data(self):
        answers_per_question = self._settings["datagen:answers_per_question"]
        for j in range(answers_per_question):
            for statement_id in self._statements.index:
                statement = self._statements.loc[statement_id]
                user_message_suffix = self._settings["user_message_suffix"]
                question = f'"{statement}"\n{user_message_suffix}'
                answer = self._answers.loc[statement_id, f"Answer {j + 1}"]
                yield {
                    "messages": [
                        {
                            "role": "user",
                            "content": question,
                        },
                        {
                            "role": "assistant",
                            "content": answer,
                        },
                    ]
                }

    def _get_sft_config(self):
        # This is Qwen3's official chat template, with one addition: the keywords {% generation %} and {% endgeneration %}.
        # These keywords are required for `assistant_only_loss=True` to work, as documented here:
        # https://huggingface.co/docs/trl/main/en/sft_trainer#train-on-assistant-messages-only
        chat_template_file_path = str(
            self._paths.repo_folder_path / "chat_template_with_assistant_mask.jinja"
        )

        checkpoints_folder_path = str(self._paths.cache_folder_path / "checkpoints")
        packing_enabled = self._settings["sft:packing"]
        model_init_kwargs = {"dtype": "bfloat16"}
        if packing_enabled:
            model_init_kwargs["attn_implementation"] = "flash_attention_2"
        return SFTConfig(
            # GENERAL
            output_dir=checkpoints_folder_path,
            assistant_only_loss=True,
            num_train_epochs=self._settings["sft:num_epochs"],
            per_device_train_batch_size=self._settings[
                "sft:per_device_train_batch_size"
            ],
            gradient_accumulation_steps=self._settings[
                "sft:gradient_accumulation_steps"
            ],
            bf16=True,
            # CHECKPOINTING
            save_only_model=True,
            save_strategy="steps",
            save_steps=self._settings["sft:save_interval"],
            # PACKING
            packing=packing_enabled,
            chat_template_path=chat_template_file_path,
            model_init_kwargs=model_init_kwargs,
            max_length=self._settings["max_model_len"],
            # MONITORING
            logging_steps=1,
            logging_first_step=True,
            log_level="debug",
            report_to="wandb",
        )

    def _get_peft_config(self):
        return LoraConfig(
            r=self._settings["lora_rank"],
            lora_alpha=self._settings["lora_alpha"],
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[  # UnSloth's recommendation
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )

    def finetune(self):
        self._logger.info("Running SFT...")
        training_data = Dataset.from_generator(self._generate_training_data)
        sft_config = self._get_sft_config()
        peft_config = self._get_peft_config()
        self._logger.debug(
            f"Using TRL's SFTTrainer with: {sft_config}\nAnd: {peft_config}"
        )
        os.environ['WANDB_DIR'] = str(self._paths.outputs_folder_path)
        self._trainer = SFTTrainer(
            model=self._settings["base_model_id"],
            train_dataset=training_data,
            args=sft_config,
            peft_config=peft_config,
        )
        self._trainer.train()
        self._logger.info("SFT completed.")
