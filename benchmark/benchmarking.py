import os
import sys
import json
import string
import datetime

import transformers
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import torch
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score

import datasets
from transformers import (
    HfArgumentParser,
    AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
)

sys.path.insert(1, '/Users/abdullaeva/Documents/Projects/TransactionsQA')
# for MlSpace: /home/jovyan/transactionsQA/romashka
print(sys.path)

from romashka.benchmark.metrics import mlqa
from romashka.benchmark.qa_utils import normalize_mlqa
from romashka.benchmark.benchmark_args import BenchmarkArguments
from romashka.benchmark.tasks.mappings import ALL_TASKS
from romashka.benchmark.tasks.text_autotask import AutoTextTask
# from data.utils import TaskCollator
# from model.train_mt5_pred import run_train

os.environ['HF_DATASETS_OFFLINE'] = '0'
os.environ['TRANSFORMERS_OFFLINE'] = '0'

# Set up verbosity
import warnings
warnings.filterwarnings("ignore")

# Setup logging
from romashka.logging_handler import get_logger


def generate(args,
             device,
             model: torch.nn.Module,
             tokenizer: transformers.PreTrainedTokenizerBase,
             source_text: str) -> str:
    """
    Run generation with text model & tokenizer of provided task.
    Args:
        args: a dict of arguments for generate() function from Transformers;
        device: a device to run generation on;
        model: a model instance to run generation with;
        tokenizer: a Tokenizer instance;
        source_text: a task text context;

    Returns:
        a generated output.
    """
    source_encoding = tokenizer(
        source_text,
        max_length=args.max_len_input,
        padding='max_length',
        truncation='only_second',
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    )

    # Put this in GPU (faster than using cpu)
    input_ids = source_encoding['input_ids'].to(device)
    attention_mask = source_encoding['attention_mask'].to(device)

    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_return_sequences=args.num_return_sequences,  # defaults to 1
        num_beams=args.num_beams,  # defaults to 1
        max_length=args.max_len_output,
        repetition_penalty=args.repetition_penalty,  # defaults to 1.0
        length_penalty=args.length_penalty,  # defaults to 1.0
        early_stopping=True,  # defaults to False
        use_cache=True
    )

    predictions = {
        tokenizer.decode(generated_id,
                         skip_special_tokens=True,
                         clean_up_tokenization_spaces=True)
        for generated_id in generated_ids
    }

    return ''.join(predictions)


def main():
    parser = HfArgumentParser(BenchmarkArguments)

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        args = parser.parse_args_into_dataclasses()
        args = args[0]
        print(f"{args}")

    pl.seed_everything(args.seed)
    # Set up logging
    logger = get_logger(
        name="benchmark",
        logging_level="INFO"
    )

    # Load/create model
    if not args.from_checkpoint:
        logger.info(f"Loading HF pre-trained model: {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    else:
        logger.info(f"Loading fine-tuned model: {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, bos_token='<s>', sep_token='<sep>')

        logger.info(f"Adding special tokens: ['[NLU]', '[NLG]', '[S2S]']")
        tokenizer.add_tokens(list(set(['[NLU]', '[NLG]', '[S2S]'])))

        config = AutoConfig.from_pretrained(args.model_name,
                                            vocab_size=len(tokenizer),
                                            dropout_rate=0.05)
        model = AutoModelForSeq2SeqLM(config)

        checkpoint_model_path = args.checkpoint_model_path
        if args.from_checkpoint:
            logger.info("Initializing from checkpoint...")
            if ".pt" in checkpoint_model_path:
                model.load_state_dict(torch.load(checkpoint_model_path))
            else:
                checkpoint = torch.load(checkpoint_model_path)
                model_state_dict = dict([(n[6:], p) for n, p in checkpoint['state_dict'].items()])
                model.load_state_dict(model_state_dict)  # , map_location='cpu' force loading on CPU
            logger.info("Weights loaded!")

    # Put model in freeze() and eval() model. Not sure the purpose of freeze
    # Not sure if this should be after or before changing device for inference.
    model.eval()

    # Put model in gpu (if possible) or cpu (if not possible) for inference purpose
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info(f"Device selected for inference: {device}")

    # Create tasks dataset
    dataset_class = AutoTextTask
    logger.info(f"Loading dataset: {args.dataset_name}...")
    # last - in case if a task names passed without dataset relation
    tasks_list = ALL_TASKS.get(args.dataset_name, [args.dataset_name])
    if args.task_names is not None:
        if isinstance(args.task_names, list):
            task_names = args.task_names
        elif args.task_names.startswith("[") or args.task_names.startswith("("):
            task_names = eval(args.task_names)
        else:
            task_names = args.task_names

        print(f"Arch task_names: {task_names}")
        tasks_list = [task_name for task_name in task_names if task_name in tasks_list]
    logger.info(f"Tasks selected from dataset: {tasks_list}")

    benchmark_dataset = [
        dataset_class.get(task_name, seed=args.seed).get_dataset(
            split=args.task_split,
            requested_n=args.task_num_samples,
            add_prefix=False) for task_name in tasks_list
    ]

    # Run benchmark
    all_tasks_predicted_texts = {}
    all_tasks_target_texts = {}
    all_tasks_metrics_dict = {}

    for i, data in enumerate(benchmark_dataset):
        task_predicted_texts = []
        task_target_texts = []

        for sample in tqdm(data):
            inputs = sample['src_texts']
            predicted = generate(args, device, model, tokenizer, inputs)

            normalized_predicted = normalize_mlqa(predicted.lower(),
                                                  lang="en",
                                                  punct=set(string.punctuation)).strip()
            normalized_target = normalize_mlqa(
                sample['tgt_texts'].lower(),
                lang="en",
                punct=set(string.punctuation)).strip()
            task_predicted_texts.append(normalized_predicted)
            task_target_texts.append(normalized_target)

            if args.verbose:
                logger.info(inputs)
                logger.info('-' * 100)
                logger.info(normalized_predicted)
                logger.info('*' * 100)
                logger.info(sample['tgt_texts'])
                logger.info('#' * 100)
                logger.info('\n\n')

        # Metrics calculation
        task_name = tasks_list[i]

        acc = accuracy_score(task_target_texts, task_predicted_texts)
        logger.info(f"\nAccuracy score for {task_name} is {acc}")

        task_metrics = mlqa([[el] for el in task_target_texts], task_predicted_texts, lang="ru")
        logger.info(f"Metrics score for {task_name} is {task_metrics}")

        all_tasks_metrics_dict[task_name] = task_metrics
        all_tasks_target_texts[task_name] = task_target_texts
        all_tasks_predicted_texts[task_name] = task_predicted_texts

    # Find saving folder - 1-st level
    current_path = Path().resolve()
    saving_path = (current_path / args.save_metrics_folder)
    if saving_path.exists():
        logger.info(f"General folder for metrics saving already exists by path: {saving_path}")
    else:
        logger.info(f"Creating folder for metrics saving do not exists by path: {saving_path}")
        logger.info(f"Creating...")
        saving_path.mkdir()

    # Create experiment saving folder - 2-st level
    experiment_name = args.save_metrics_subfolder if args.save_metrics_subfolder else 'results-{task}-{date:%Y-%m-%d_%H:%M:%S}'.format(
        task=args.dataset_name, date=datetime.datetime.now())
    saving_path = saving_path / experiment_name
    if saving_path.exists():
        logger.info(f"Path for metrics saving already exists by path: {saving_path}!")
    else:
        logger.info(f"Creating path for metrics saving do not exists by path: {saving_path}")
        logger.info(f"Creating...")
        saving_path.mkdir()

    preds_save_fn = str(saving_path / f'all_tasks_predictions.json')
    with open(preds_save_fn, 'w') as f:
        json.dump(all_tasks_predicted_texts, f, ensure_ascii=False)

    metrics_save_fn = str(saving_path / f'all_tasks_metrics.json')
    with open(metrics_save_fn, 'w') as f:
        json.dump(all_tasks_metrics_dict, f, ensure_ascii=False)


if __name__ == '__main__':
    import os

    # os.environ['HF_DATASETS_OFFLINE'] = '1'  # offline mode for HF datasets
    # os.environ['TRANSFORMERS_OFFLINE'] = '1'  # offline mode for HF Transformers
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # disable DataParallel for test
    main()