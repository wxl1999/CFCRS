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
from torch.utils.data import DataLoader, ConcatDataset
from tqdm.auto import tqdm
from transformers import BartConfig
from transformers.trainer_pt_utils import get_parameter_names

from dataset import FlowDataset, MetaPathDataCollator, UserDataCollator
from dataset_kg import KG
from metric import RecMetric
from model import MetaPathPredictor, TokenEmbedding, UserEncoder
from utils import simple_collate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str, default='save', help="Where to store the final model.")
    parser.add_argument("--debug", action='store_true', help="Debug mode.")
    # data
    parser.add_argument("--dataset", type=str, required=True, help="A file containing all data.")
    parser.add_argument("--shot", type=float, default=1)
    parser.add_argument("--other_dataset", type=str)
    parser.add_argument("--other_shot", type=float, default=1)
    parser.add_argument("--kg_dataset", type=str)
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--user_entity_max_length", type=int)
    # model
    parser.add_argument("--num_bases", type=int, default=8)
    parser.add_argument("--pretrain_dir", type=str)
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
def test(dataloader, stage):
    meta_path_evaluator.log_write_line(f'\n*** {stage}-{epoch} ***\n')

    for model in models:
        model.eval()
    token_embeds = token_embedding(edge_index, edge_type)  # (n_tok, hs)

    meta_path_loss_list = []
    for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
        # user embedding
        user_batch = user_data_collator(batch)
        user_inputs = {**user_batch['entity'], 'token_embeds': token_embeds}

        user_embeds = user_encoder(**user_inputs)
        user_embeds = user_embeds.reshape(len(user_embeds) // 2, 2, -1)

        # meta path prediction
        meta_path_batch = meta_path_data_collator(batch, infer=False)
        meta_path_inputs = {'user_embeds': user_embeds, 'labels': meta_path_batch['labels']}

        meta_path_outputs = meta_path_predictor(**meta_path_inputs)
        meta_path_loss = meta_path_outputs['loss']
        meta_path_loss_list.append(float(meta_path_loss))

        preds = []
        for logit, meta_path_mask in zip(meta_path_outputs['logits'], meta_path_batch['meta_path_mask']):
            masked_indices = torch.arange(len(logit))[meta_path_mask]
            masked_logit = logit[meta_path_mask]
            rank = torch.topk(masked_logit, k=min(10, len(masked_logit)), dim=-1).indices
            pred = masked_indices[rank].tolist()
            preds.append(pred)
        meta_path_evaluator.evaluate(preds, meta_path_batch['labels'].tolist(), log=True)

    # metric
    meta_path_report = meta_path_evaluator.report()
    meta_path_report['meta_path_loss'] = np.mean(meta_path_loss_list)

    return meta_path_report


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
    kg = KG(dataset=args.kg_dataset, debug=args.debug).get_kg_info()

    entity2id = kg['entity2id']
    id2entity = kg['id2entity']
    num_entities, num_relations = kg['num_entities'], kg['num_relations']
    edge_index = torch.as_tensor(kg['edge_index'], device=device)
    edge_type = torch.as_tensor(kg['edge_type'], device=device)

    entityid2typeid = kg['entityid2typeid']
    id2metapath = kg['id2metapath']
    id2metapathid = kg['id2metapathid']
    meta_path_num_labels = len(id2metapathid)

    # model
    config = BartConfig.from_pretrained(f'{args.pretrain_dir}/flow')
    num_special_tokens = len(entity2id) - num_entities

    token_embedding = TokenEmbedding(
        hidden_size=config.hidden_size, num_bases=args.num_bases,
        num_relations=num_relations, num_entities=num_entities, num_special_tokens=num_special_tokens
    )
    token_embedding.load(f'{args.pretrain_dir}/token')
    token_embedding = token_embedding.to(device)
    token_embedding.requires_grad_(False)

    user_encoder = UserEncoder(hidden_size=config.hidden_size)
    user_encoder.load(f'{args.pretrain_dir}/user')
    user_encoder.to(device)
    user_encoder.requires_grad_(False)

    meta_path_predictor = MetaPathPredictor(hidden_size=config.hidden_size, num_labels=meta_path_num_labels).to(device)

    models = [token_embedding, user_encoder, meta_path_predictor]
    for i, model in enumerate(models):
        models[i] = accelerator.prepare(model)

    # data
    train_dataset = FlowDataset(
        dataset=args.dataset, split='train', debug=args.debug, shot=args.shot, compute_loss_for_meta=True,
        entity2id=entity2id, entityid2typeid=entityid2typeid, id2metapathid=id2metapathid,
        max_length=args.max_length, user_entity_max_length=args.user_entity_max_length
    )
    if args.other_dataset is not None:
        other_dataset = FlowDataset(
            dataset=args.other_dataset, split='train', debug=args.debug, compute_loss_for_meta=True,
            entity2id=entity2id, entityid2typeid=entityid2typeid, id2metapathid=id2metapathid,
            max_length=args.max_length, user_entity_max_length=args.user_entity_max_length
        )
        train_dataset = ConcatDataset([train_dataset, other_dataset])
    valid_dataset = FlowDataset(
        dataset=args.dataset, split='valid', debug=args.debug, shot=args.shot, compute_loss_for_meta=True,
        entity2id=entity2id, entityid2typeid=entityid2typeid, id2metapathid=id2metapathid,
        # max_length=args.max_length, user_entity_max_length=args.user_entity_max_length
    )
    test_dataset = FlowDataset(
        dataset=args.dataset, split='test', debug=args.debug, shot=args.shot, compute_loss_for_meta=True,
        entity2id=entity2id, entityid2typeid=entityid2typeid, id2metapathid=id2metapathid,
        # max_length=args.max_length, user_entity_max_length=args.user_entity_max_length
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

    user_data_collator = UserDataCollator(
        entity2id=entity2id, max_length=args.max_length, device=device, debug=args.debug
    )
    meta_path_data_collator = MetaPathDataCollator(device=device, debug=args.debug)

    # optim
    decay_parameters = get_parameter_names(meta_path_predictor, [nn.LayerNorm])
    decay_parameters = {name for name in decay_parameters if "bias" not in name}
    optimizer_grouped_parameters = [
        {
            "params": list(set([p for model in models for n, p in model.named_parameters() if n in decay_parameters])),
            "weight_decay": args.weight_decay,
        },
        {
            "params": list(
                set([p for model in models for n, p in model.named_parameters() if n not in decay_parameters])),
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
    completed_steps = 0

    # lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_update_steps_per_epoch, args.max_train_steps)
    # lr_scheduler = accelerator.prepare(lr_scheduler)

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

    # eval
    log_file_dir = 'log'
    os.makedirs(log_file_dir, exist_ok=True)
    log_file_path = os.path.join(log_file_dir, f'gen_{local_time}.jsonl')

    meta_path_evaluator = RecMetric(k_list=(1, 5, 10), log_file_path=log_file_path)

    # save model with min loss
    best_metric_val = float('inf')
    best_dir = os.path.join(args.output_dir, 'best')
    meta_path_best_dir = os.path.join(best_dir, 'meta_path')
    os.makedirs(meta_path_best_dir, exist_ok=True)

    # train loop
    for epoch in range(args.num_train_epochs):
        meta_path_train_loss = []
        for model in models:
            model.train()

        for step, batch in enumerate(train_dataloader):
            # token embedding
            token_embeds = token_embedding(edge_index, edge_type)  # (n_tok, hs)

            # user embedding
            user_batch = user_data_collator(batch)
            user_inputs = {**user_batch['entity'], 'token_embeds': token_embeds}

            user_embeds = user_encoder(**user_inputs)
            user_embeds = user_embeds.reshape(len(user_embeds) // 2, 2, -1)

            # meta path prediction
            meta_path_batch = meta_path_data_collator(batch, infer=False)
            meta_path_inputs = {'user_embeds': user_embeds, 'labels': meta_path_batch['labels']}

            meta_path_outputs = meta_path_predictor(**meta_path_inputs)
            meta_path_loss = meta_path_outputs['loss'] / args.gradient_accumulation_steps
            meta_path_train_loss.append(float(meta_path_loss))

            loss = meta_path_loss
            accelerator.backward(loss)

            # optim step
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                if args.max_grad_norm is not None:
                    for model in models:
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                # lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                completed_steps += 1
                if run:
                    run.log({
                        'train/meta_path_loss': np.mean(meta_path_train_loss) * args.gradient_accumulation_steps,
                    })

            if completed_steps >= args.max_train_steps:
                break

        # metric
        meta_path_train_loss_avg = np.mean(meta_path_train_loss) * args.gradient_accumulation_steps
        logger.info(f'epoch {epoch} meta path train loss {meta_path_train_loss_avg}')

        # validation
        meta_path_report = test(valid_dataloader, 'valid')
        valid_report = {'epoch': epoch}
        for report in [meta_path_report]:
            for k, v in report.items():
                valid_report[f'valid/{k}'] = v

        logger.info(f'{valid_report}')
        if run:
            run.log(valid_report)
        meta_path_evaluator.reset_metric()

        # save model
        cur_metric_val = meta_path_report['recall@1']
        if cur_metric_val < best_metric_val:
            best_metric_val = cur_metric_val
            meta_path_predictor.save(meta_path_best_dir)

        # test
        meta_path_report = test(test_dataloader, 'test')
        test_report = {'epoch': epoch}
        for report in [meta_path_report]:
            for k, v in report.items():
                test_report[f'test/{k}'] = v

        logger.info(f'{test_report}')
        if run:
            run.log(test_report)
        meta_path_evaluator.reset_metric()

    # save model
    final_dir = os.path.join(args.output_dir, 'final')
    meta_path_final_dir = os.path.join(final_dir, 'meta_path')
    os.makedirs(meta_path_final_dir, exist_ok=True)
    meta_path_predictor.save(meta_path_final_dir)
