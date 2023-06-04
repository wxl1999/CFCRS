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
from torch.utils.data import DataLoader, ConcatDataset
from tqdm.auto import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, BartConfig
from transformers.trainer_pt_utils import get_parameter_names

from dataloader_kbrd import KBRDDataCollatorForConv
from dataset import DatasetForConv
from metric import ConvMetric
from kg_kbrd import KGForKBRD
from utils import load_jsonl_data, simple_collate
from model_kbrd import KBRDforRec, KBRDforConv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str, default='save', help="Where to store the final model.")
    parser.add_argument("--debug", action='store_true', help="Debug mode.")
    # data
    parser.add_argument("--kg_dataset", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, help="A file containing all data.")
    parser.add_argument("--other_dataset", type=str)
    parser.add_argument("--shot", type=float, default=1)
    parser.add_argument("--context_max_length", type=int, default=True)
    parser.add_argument("--entity_max_length", type=int, default=True)
    parser.add_argument("--resp_max_length", type=int, default=True)
    # rec
    parser.add_argument("--entity_hidden_size", type=int, required=True)
    parser.add_argument("--num_bases", type=int, default=8)
    parser.add_argument("--rec_model", type=str, required=True)
    # conv
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--text_hidden_size", type=int, required=True)
    parser.add_argument("--encoder_layers", type=int, required=True)
    parser.add_argument("--decoder_layers", type=int, required=True)
    parser.add_argument("--attn_head", type=int, required=True)
    parser.add_argument("--conv_model", type=str)
    # optim
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs to perform.")
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
    parser.add_argument('--num_warmup_steps', type=int)
    parser.add_argument("--fp16", action='store_true')
    # wandb
    parser.add_argument("--use_wandb", action="store_true", help="whether to use wandb")
    parser.add_argument("--entity", type=str, help="wandb username")
    parser.add_argument("--project", type=str, help="wandb exp project")
    parser.add_argument("--name", type=str, help="wandb exp name")

    args = parser.parse_args()
    return args


@torch.no_grad()
def evaluation(dataloader):
    conv_model.eval()

    loss_list = []
    for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
        batch = data_collator(batch)
        user_embeds = rec_model(**batch['entity'], node_embeds=node_embeds)['user_embeds']
        loss = conv_model(**batch['context'], decoder_user_embeds=user_embeds).loss
        loss_list.append(float(loss))

    for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
        batch = data_collator(batch)
        labels = batch['context'].pop('labels')

        user_embeds = rec_model(**batch['entity'], node_embeds=node_embeds)['user_embeds']
        gen_inputs = {**batch['context'], 'decoder_user_embeds': user_embeds}

        gen_args = {
            # 'min_length': 0,
            'max_new_tokens': args.resp_max_length,
            'num_beams': 1,
            'no_repeat_ngram_size': 3,
            'encoder_no_repeat_ngram_size': 3
        }
        gen_seqs = accelerator.unwrap_model(conv_model).generate(**gen_inputs, **gen_args)

        evaluator.evaluate(gen_seqs[:, 1:], labels, log=accelerator.is_local_main_process)

    # metric
    report = evaluator.report()
    report['loss'] = np.mean(loss_list)

    return report


def learning():
    # training info
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    completed_steps = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    # save model with min loss
    best_metric_val = float('inf')
    best_dir = os.path.join(args.output_dir, 'best')
    os.makedirs(best_dir, exist_ok=True)

    # train loop
    for epoch in range(args.num_train_epochs):
        conv_model.train()
        train_loss = []

        for step, batch in enumerate(train_dataloader):
            batch = data_collator(batch)
            user_embeds = rec_model(**batch['entity'], node_embeds=node_embeds)['user_embeds']
            loss = conv_model(**batch['context'], decoder_user_embeds=user_embeds).loss
            accelerator.backward(loss)
            train_loss.append(float(loss))

            # optim step
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                if args.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(conv_model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                completed_steps += 1
                if run:
                    run.log({'train/loss': np.mean(train_loss) * args.gradient_accumulation_steps})

            if completed_steps >= args.max_train_steps:
                break

        # metric
        train_loss = np.mean(train_loss) * args.gradient_accumulation_steps
        logger.info(f'epoch {epoch} train loss {train_loss}')

        # dev
        evaluator.log_write_line(f'\n*** valid-{epoch} ***\n')
        report = evaluation(valid_dataloader)
        valid_report = {}
        for k, v in report.items():
            valid_report[f'valid/{k}'] = v
        valid_report['epoch'] = epoch

        logger.info(f'{valid_report}')
        if run:
            run.log(valid_report)
        evaluator.reset_metric()

        # save model
        if report['loss'] < best_metric_val:
            best_metric_val = report['loss']
            accelerator.unwrap_model(conv_model).save_pretrained(best_dir, save_function=accelerator.save)

        # test
        evaluator.log_write_line(f'\n*** test-{epoch} ***\n')
        report = evaluation(test_dataloader)
        test_report = {}
        for k, v in report.items():
            test_report[f'test/{k}'] = v
        test_report['epoch'] = epoch

        logger.info(f'{test_report}')
        if run:
            run.log(test_report)
        evaluator.reset_metric()

    # save model
    final_dir = os.path.join(args.output_dir, 'final')
    os.makedirs(final_dir, exist_ok=True)
    accelerator.unwrap_model(conv_model).save_pretrained(final_dir, save_function=accelerator.save)


if __name__ == '__main__':
    args = parse_args()
    config = vars(args)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Initialize the accelerator. We will let the accelerator handle device placement for us.
    accelerator = Accelerator(device_placement=False, fp16=args.fp16)
    device = accelerator.device

    # Make one log on every process with the configuration for debugging.
    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    logger.remove()
    logger.add(sys.stderr, level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.add(f'log/{local_time}.log', level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.info(accelerator.state)
    logger.info(config)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    # wandb
    run = None
    if args.use_wandb:
        name = args.name if args.name else local_time
        run = wandb.init(entity=args.entity, project=args.project, config=config, name=name)

    # kg
    kg = KGForKBRD(kg_dataset=args.kg_dataset, debug=args.debug).get_kg_info()
    edge_index, edge_type = torch.as_tensor(kg['edge_index'], device=device), torch.as_tensor(kg['edge_type'],
                                                                                              device=device)

    # model
    rec_model = KBRDforRec(
        hidden_size=args.entity_hidden_size, num_relations=kg['num_relations'], num_entities=kg['num_entities'],
        num_bases=args.num_bases
    )
    rec_model.load(args.rec_model)
    rec_model = rec_model.requires_grad_(False).to(device)
    node_embeds = rec_model.get_node_embeds(edge_index, edge_type)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    config = BartConfig.from_pretrained(
        "facebook/bart-base", encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers,
        hidden_size=args.text_hidden_size, encoder_attention_heads=args.attn_head, decoder_attention_heads=args.attn_head,
        encoder_ffn_dim=args.text_hidden_size, decoder_ffn_dim=args.text_hidden_size,
        forced_bos_token_id=None, forced_eos_token_id=None
    )
    conv_model = KBRDforConv(config, user_hidden_size=args.entity_hidden_size).to(device)
    if args.conv_model is not None:
        conv_model = KBRDforConv.from_pretrained(args.conv_model, user_hidden_size=args.entity_hidden_size).to(device)
    conv_model = accelerator.prepare(conv_model)

    # data
    train_data_file = os.path.join('data', args.dataset, 'train_data_processed.jsonl')
    train_data_list = load_jsonl_data(train_data_file)
    train_dataset = DatasetForConv(
        data_list=train_data_list, tokenizer=tokenizer, entity2id=kg['entity2id'],
        context_max_length=args.context_max_length, resp_max_length=args.resp_max_length,
        entity_max_length=args.entity_max_length,
        debug=args.debug, shot=args.shot,
    )
    if args.other_dataset is not None:
        other_data_file = os.path.join('data', args.other_dataset, 'train_data_processed.jsonl')
        other_data_list = load_jsonl_data(other_data_file)
        other_dataset = DatasetForConv(
            data_list=other_data_list, tokenizer=tokenizer, entity2id=kg['entity2id'],
            context_max_length=args.context_max_length, resp_max_length=args.resp_max_length,
            entity_max_length=args.entity_max_length,
            debug=args.debug, shot=args.shot,
        )
        train_dataset = ConcatDataset([train_dataset, other_dataset])

    valid_data_file = os.path.join('data', args.dataset, 'valid_data_processed.jsonl')
    valid_data_list = load_jsonl_data(valid_data_file)
    valid_dataset = DatasetForConv(
        data_list=valid_data_list, tokenizer=tokenizer, entity2id=kg['entity2id'], debug=args.debug,
        context_max_length=args.context_max_length, resp_max_length=args.resp_max_length,
        entity_max_length=args.entity_max_length
    )

    test_data_file = os.path.join('data', args.dataset, 'test_data_processed.jsonl')
    test_data_list = load_jsonl_data(test_data_file)
    test_dataset = DatasetForConv(
        data_list=test_data_list, tokenizer=tokenizer, entity2id=kg['entity2id'], debug=args.debug,
        context_max_length=args.context_max_length, resp_max_length=args.resp_max_length,
        entity_max_length=args.entity_max_length
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=simple_collate,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=simple_collate,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=simple_collate,
    )
    train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
        train_dataloader, valid_dataloader, test_dataloader
    )

    data_collator = KBRDDataCollatorForConv(
        tokenizer=tokenizer, entity_pad_id=kg['pad_id'], device=device, debug=args.debug, use_amp=accelerator.use_fp16,
        context_max_length=args.context_max_length, resp_max_length=args.resp_max_length,
        entity_max_length=args.entity_max_length
    )

    # optim
    decay_parameters = get_parameter_names(conv_model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in conv_model.named_parameters() if n in decay_parameters],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in conv_model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    optimizer = accelerator.prepare(optimizer)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    # lr scheduler
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_update_steps_per_epoch, args.max_train_steps)
    lr_scheduler = accelerator.prepare(lr_scheduler)

    # eval
    log_file_dir = 'log'
    os.makedirs(log_file_dir, exist_ok=True)
    log_file_path = os.path.join(log_file_dir, f'gen_{local_time}.jsonl')
    evaluator = ConvMetric(tokenizer=tokenizer, log_file_path=log_file_path)

    learning()
