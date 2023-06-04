import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import transformers
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from loguru import logger
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from transformers.trainer_pt_utils import get_parameter_names

from config import gpt2_special_tokens_dict, prompt_special_tokens_dict
from dataloader_unicrs import UniCRSDataCollatorForRec
from dataset import DatasetForRec
from evaluate_rec import RecEvaluator
from kg_unicrs import KGForUniCRS
from model_gpt2 import PromptGPT2forCRS
from model_prompt import KGPrompt
from utils import load_jsonl_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str, default='save', help="Where to store the final model.")
    parser.add_argument("--debug", action='store_true', help="Debug mode.")
    # data
    parser.add_argument("--kg", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, help="A file containing all data.")
    parser.add_argument("--shot", type=float, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument("--max_length", type=int, help="max input length in dataset.")
    parser.add_argument("--resp_max_length", type=int)
    parser.add_argument("--prompt_max_length", type=int)
    parser.add_argument("--entity_max_length", type=int, help="max entity length in dataset.")
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--text_tokenizer", type=str)
    # model
    parser.add_argument("--model", type=str, required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--text_encoder", type=str, required=True)
    parser.add_argument("--prompt_encoder", type=str)
    parser.add_argument("--num_bases", type=int, default=8, help="num_bases in RGCN.")
    # optim
    parser.add_argument("--num_train_epochs", type=int, default=50, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                        help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument("--fp16", action='store_true')
    parser.add_argument("--no_lr_scheduler", action='store_true')
    # wandb
    parser.add_argument("--use_wandb", action="store_true", help="whether to use wandb")
    parser.add_argument("--entity", type=str, help="wandb username")
    parser.add_argument("--project", type=str, help="wandb exp project")
    parser.add_argument("--name", type=str, help="wandb exp name")

    args = parser.parse_args()
    return args


def crs_model_evaluation(dataloader):
    # valid
    prompt_encoder.eval()
    loss_list = []
    for batch in tqdm(dataloader):
        with torch.no_grad():
            token_embeds = text_encoder(**batch['prompt']).last_hidden_state
            prompt_embeds = prompt_encoder(
                entity_ids=batch['entity'],
                token_embeds=token_embeds,
                output_entity=True
            )
            batch['context']['prompt_embeds'] = prompt_embeds
            batch['context']['entity_embeds'] = prompt_encoder.get_entity_embeds()

            outputs = model(**batch['context'], rec=True)
            loss_list.append(float(outputs.rec_loss))
            logits = outputs.rec_logits[:, item_ids]
            ranks = torch.topk(logits, k=50, dim=-1).indices
            preds = item_ids[ranks].tolist()
            labels = batch['context']['rec_labels'].tolist()
            evaluator.evaluate(preds, labels)

    # metric
    report = evaluator.report()
    report['loss'] = np.mean(loss_list)
    return report


def crs_model_learning():
    # training info
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    # save model with best metric
    metric, mode = 'loss', -1
    assert mode in (-1, 1)
    if mode == 1:
        best_metric = 0
    else:
        best_metric = float('inf')
    best_metric_dir = os.path.join(args.output_dir, 'best')
    os.makedirs(best_metric_dir, exist_ok=True)

    # train loop
    for epoch in range(args.num_train_epochs):
        # train
        prompt_encoder.train()
        train_loss = []
        for step, batch in enumerate(train_dataloader):
            with torch.no_grad():
                token_embeds = text_encoder(**batch['prompt']).last_hidden_state

            prompt_embeds = prompt_encoder(
                entity_ids=batch['entity'],
                token_embeds=token_embeds,
                output_entity=True
            )
            batch['context']['prompt_embeds'] = prompt_embeds
            batch['context']['entity_embeds'] = prompt_encoder.get_entity_embeds()

            loss = model(**batch['context'], rec=True).rec_loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            train_loss.append(float(loss))

            # optim step
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                if args.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(prompt_encoder.parameters(), args.max_grad_norm)
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                completed_steps += 1
                if run:
                    run.log({'loss': np.mean(train_loss) * args.gradient_accumulation_steps})

            if completed_steps >= args.max_train_steps:
                break

        # metric
        train_loss = np.mean(train_loss) * args.gradient_accumulation_steps
        logger.info(f'epoch {epoch} train loss {train_loss}')

        # valid
        report = crs_model_evaluation(valid_dataloader)
        valid_report = {}
        for k, v in report.items():
            valid_report[f'valid/{k}'] = v
        valid_report['epoch'] = epoch
        logger.info(f'{valid_report}')
        if run:
            run.log(valid_report)
        evaluator.reset_metric()

        if valid_report[f'valid/{metric}'] * mode > best_metric * mode:
            prompt_encoder.save(best_metric_dir)
            best_metric = valid_report[f'valid/{metric}']
            logger.info(f'new best model with {metric}')

        # test
        report = crs_model_evaluation(test_dataloader)
        test_report = {}
        for k, v in report.items():
            test_report[f'test/{k}'] = v
        test_report['epoch'] = epoch
        logger.info(f'{test_report}')
        if run:
            run.log(test_report)
        evaluator.reset_metric()

        # epoch_dir = os.path.join(args.output_dir, str(epoch))
        # os.makedirs(epoch_dir, exist_ok=True)
        # prompt_encoder.save(epoch_dir)
        # logger.info(f'save model of epoch {epoch}')

    # test
    report = crs_model_evaluation(test_dataloader)
    test_report = {}
    for k, v in report.items():
        test_report[f'test/{k}'] = v
    # test_report['epoch'] = epoch
    logger.info(f'{test_report}')
    if run:
        run.log(test_report)
    evaluator.reset_metric()


if __name__ == '__main__':
    args = parse_args()
    config = vars(args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us.
    accelerator = Accelerator(device_placement=False, fp16=args.fp16)
    device = accelerator.device

    # Make one log on every process with the configuration for debugging.
    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    logger.remove()
    logger.add(sys.stderr, level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.add(f'log/{local_time}.log', level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.info(config)
    logger.info(accelerator.state)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    # wandb
    run = None
    if args.use_wandb:
        name = args.name if args.name else local_time
        run = wandb.init(entity=args.entity, project=args.project, config=config, name=name)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # kg
    kg = KGForUniCRS(kg=args.kg, debug=args.debug).get_kg_info()
    item_ids = torch.as_tensor(kg['item_ids'], device=device)

    # model
    # backbone
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.add_special_tokens(gpt2_special_tokens_dict)
    tokenizer.padding_side = 'left'

    model = PromptGPT2forCRS.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device).requires_grad_(False)

    # prompt text encoder
    text_tokenizer = AutoTokenizer.from_pretrained(args.text_tokenizer)
    text_tokenizer.add_special_tokens(prompt_special_tokens_dict)
    text_encoder = AutoModel.from_pretrained(args.text_encoder)
    text_encoder.resize_token_embeddings(len(text_tokenizer))
    text_encoder = text_encoder.to(device).requires_grad_(False)

    # prompt encoder (+kg)
    prompt_encoder = KGPrompt(
        model.config.n_embd, text_encoder.config.hidden_size, model.config.n_head, model.config.n_layer, 2,
        n_entity=kg['num_entities'], num_relations=kg['num_relations'], num_bases=args.num_bases,
        edge_index=kg['edge_index'], edge_type=kg['edge_type'],
    )
    if args.prompt_encoder is not None:
        prompt_encoder.load(args.prompt_encoder)
    prompt_encoder = prompt_encoder.to(device)
    prompt_encoder = accelerator.prepare(prompt_encoder)

    # data
    train_data_file = os.path.join('data', args.dataset, 'train_data_processed.jsonl')
    train_data_list = load_jsonl_data(train_data_file)
    train_dataset = DatasetForRec(
        train_data_list, tokenizer=tokenizer, prompt_tokenizer=text_tokenizer, debug=args.debug, shot=args.shot,
        context_max_length=args.max_length, resp_max_length=args.resp_max_length,
        entity2id=kg['entity2id'], entity_max_length=args.entity_max_length
    )

    valid_data_file = os.path.join('data', args.dataset, 'valid_data_processed.jsonl')
    valid_data_list = load_jsonl_data(valid_data_file)
    valid_dataset = DatasetForRec(
        valid_data_list, tokenizer=tokenizer, prompt_tokenizer=text_tokenizer, debug=args.debug,
        context_max_length=args.max_length, resp_max_length=args.resp_max_length,
        entity2id=kg['entity2id'], entity_max_length=args.entity_max_length
    )

    test_data_file = os.path.join('data', args.dataset, 'test_data_processed.jsonl')
    test_data_list = load_jsonl_data(test_data_file)
    test_dataset = DatasetForRec(
        test_data_list, tokenizer=tokenizer, prompt_tokenizer=text_tokenizer, debug=args.debug,
        context_max_length=args.max_length, resp_max_length=args.resp_max_length,
        entity2id=kg['entity2id'], entity_max_length=args.entity_max_length
    )

    data_collator = UniCRSDataCollatorForRec(
        tokenizer=tokenizer, prompt_tokenizer=text_tokenizer, context_max_length=args.max_length,
        entity_pad_id=kg['pad_entity_id'], entity_max_length=args.entity_max_length,
        device=device, use_amp=accelerator.use_fp16, debug=args.debug,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        collate_fn=data_collator,
        shuffle=True
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=data_collator,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=data_collator,
    )
    train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
        train_dataloader, valid_dataloader, test_dataloader
    )

    # optim
    decay_parameters = get_parameter_names(prompt_encoder, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in prompt_encoder.named_parameters() if n in decay_parameters],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in prompt_encoder.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    # step, epoch, batch size
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    # lr_scheduler
    if args.no_lr_scheduler:
        lr_scheduler = None
    else:
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_update_steps_per_epoch, args.max_train_steps)
        lr_scheduler = accelerator.prepare(lr_scheduler)

    # evaluation
    evaluator = RecEvaluator()

    crs_model_learning()

    final_dir = os.path.join(args.output_dir, 'final')
    os.makedirs(final_dir, exist_ok=True)
    prompt_encoder.save(final_dir)
    logger.info(f'save final model')
