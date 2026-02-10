import argparse
import logging
import os
from pathlib import Path
import shutil
from dotenv import load_dotenv
import pandas as pd

from src.datagen import ReplyGenerator
from src.config import PathProvider, SettingProvider, configure_logger
from src.eval import Evaluator
from src.server import LLMServer
from src.sft import SFT
from src.speciesismbench import StatementsLoader

# PREPARE PIPELINE

# Load API keys
load_dotenv()

# Parse CLI arguments
cli_parser = argparse.ArgumentParser()
cli_parser.add_argument(
    "-d", "--dev-mode", help="Run pipeline in development mode", action="store_true"
)
cli_args = cli_parser.parse_args()
mode = "dev" if cli_args.dev_mode else "standard"

# Prepare config providers
paths = PathProvider(mode=mode)
settings = SettingProvider(mode=mode)

# Create outputs folder
if Path.is_dir(paths.outputs_folder_path):
    raise RuntimeError(
        f"The outputs folder ('{paths.outputs_folder_path}') already exists. "
        "Back up its contents and delete the folder to continue."
    )
os.makedirs(paths.outputs_folder_path)

# Prepare cache folder
if not Path.is_dir(paths.cache_folder_path):
    Path.mkdir(paths.cache_folder_path, parents=True)

# Prepare logger
logger_name = "pipeline"
log_folder_path = paths.outputs_folder_path
log_level = settings["log_level"]
configure_logger(
    logger_name=logger_name, log_folder_path=log_folder_path, log_level=log_level
)
logger = logging.getLogger(logger_name)

if mode == "dev":
    logger.info("Running pipeline in development mode. Outputs will not be useful.")

# LOAD SPECIESISMBENCH (i.e., speciesist statements)

statements_loader = StatementsLoader(mode=mode)
training_statements = statements_loader.load(split="training")

# GENERATE DATA (i.e., replies to speciesist statements)

replies_file_name = "replies.pkl"
replies_file_path = paths.cache_folder_path / replies_file_name
replies = None

if Path.is_file(replies_file_path):
    logger.debug(f"Found '{replies_file_name}' in cache.")
    logger.info("Found dataset in cache. Skipping dataset generation.")
    replies = pd.read_pickle(replies_file_path)
else:
    reply_generator = ReplyGenerator(
        mode=mode,
        statements=training_statements,
        system_message=settings["system_message"],
    )
    replies = reply_generator.generate()

# FINETUNE (i.e., run SFT on the generated statement-reply pairs)

checkpoints_folder_name = "checkpoints"
checkpoints_folder_path = paths.cache_folder_path / checkpoints_folder_name

if Path.is_dir(checkpoints_folder_path):
    logger.debug(f"Found folder '{checkpoints_folder_name}' in cache.")
    logger.info("Found SFT checkpoints in cache. Skipping SFT.")
else:
    Path.mkdir(checkpoints_folder_path)

    try:
        sft = SFT(mode=mode, statements=training_statements, replies=replies)
        sft.finetune()
    except Exception as exception:
        shutil.rmtree(checkpoints_folder_path)
        raise exception

# EVALUATE RESULTS

host = "127.0.0.1"
port = 8000

server = LLMServer(mode=mode, host=host, port=port)
server.start()
server.wait_until_ready()

try:
    evaluator = Evaluator(mode=mode, server_host=host, server_port=port)
    evaluator.evaluate()
finally:
    server.stop()
