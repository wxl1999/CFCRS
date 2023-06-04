import argparse
import copy
import json
import os
import random
import sys
import time

import numpy as np
import torch
import transformers
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from loguru import logger
from torch.utils.data import DataLoader, ConcatDataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

from config import gpt2_special_tokens_dict, prompt_special_tokens_dict
from dataloader_cf import UserDataCollator, MetaPathDataCollator, FlowDataCollator, TemplateDataCollator
from dataloader_unicrs import UniCRSDataCollatorForRec, UniCRSDataCollatorForPreTraining
from dataset import DatasetForFlow, DatasetForRec
from evaluate_rec import RecEvaluator
from kg_cf import KGForCF
from kg_unicrs import KGForUniCRS
from model_cf import BartForFlowGeneration, MetaPathPredictor, TokenEmbedding, UserEncoder
from model_gpt2 import PromptGPT2forCRS
from model_prompt import KGPrompt
from utils import load_jsonl_data, simple_collate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str, default='save', help="Where to store the final model.")
    parser.add_argument("--debug", action='store_true', help="Debug mode.")
    parser.add_argument("--fp16", action='store_true')
    parser.add_argument("--num_epochs", type=int, required=True)
    parser.add_argument("--no_log", action='store_true')
    # flow
    parser.add_argument("--kg_dataset", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--shot", type=float, default=1)
    parser.add_argument("--max_length", type=int, required=True)
    parser.add_argument("--user_entity_max_length", type=int)
    parser.add_argument("--pretrain_dir", type=str, required=True)
    parser.add_argument("--meta_path_model", type=str, required=True)
    parser.add_argument("--num_bases", type=int, default=8)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--num_beams", type=int, default=1, required=True)
    parser.add_argument("--use_meta_path", action='store_true')
    # data augmentation
    parser.add_argument("--user_policy", choices=['cf', 'random'], default='cf')
    parser.add_argument("--flow_policy", choices=['model', 'random'], default='model')
    parser.add_argument("--flow_dataset", type=str, required=True, help="A file containing all data.")
    parser.add_argument("--aug_prob", type=str)
    parser.add_argument("--aug_num", type=int, default=1)
    parser.add_argument("--prob_as_weight", action='store_true')
    parser.add_argument("--strategy", choices=['mix', 'pre'])
    # delta
    parser.add_argument("--delta_iterations", type=int)
    parser.add_argument("--delta_lr", type=float)
    parser.add_argument("--delta_l2", type=float)
    parser.add_argument("--delta_l2_ratio", type=float)
    # crs
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--text_tokenizer", type=str, required=True)
    parser.add_argument("--crs_pre_dataset", type=str)
    parser.add_argument("--crs_rec_dataset", type=str, required=True)
    parser.add_argument("--crs_context_max_length", type=int, required=True)
    parser.add_argument("--crs_resp_max_length", type=int, required=True)
    parser.add_argument("--crs_entity_max_length", type=int, required=True)
    parser.add_argument("--crs_kg_dataset", type=str, required=True)
    parser.add_argument("--crs_model", type=str, required=True)
    parser.add_argument("--crs_text_encoder", type=str, required=True)
    parser.add_argument("--crs_prompt_encoder", type=str, required=True)
    parser.add_argument("--crs_batch_size", type=int, required=True)
    parser.add_argument("--crs_pre_iterations", type=int, required=True)
    parser.add_argument("--crs_rec_iterations", type=int, required=True)
    parser.add_argument("--crs_pre_lr", type=float, required=True)
    parser.add_argument("--crs_rec_lr", type=float, required=True)
    parser.add_argument("--crs_l2", type=float, required=True)
    # wandb
    parser.add_argument("--use_wandb", action="store_true", help="whether to use wandb")
    parser.add_argument("--entity", type=str, help="wandb username")
    parser.add_argument("--project", type=str, help="wandb exp project")
    parser.add_argument("--name", type=str, help="wandb exp name")

    args = parser.parse_args()
    return args


@torch.no_grad()
def predict_meta_paths(user_embeds, meta_path_mask):
    meta_path_inputs = {'user_embeds': user_embeds}
    meta_path_outputs = meta_path_predictor(**meta_path_inputs)

    meta_path_indices = torch.arange(kg['num_meta_paths'])
    meta_path_preds = []
    for logit, meta_path_mask in zip(meta_path_outputs['logits'], meta_path_mask):
        masked_logit = logit[meta_path_mask]
        rank = torch.argmax(masked_logit).item()
        meta_path_pred = meta_path_indices[meta_path_mask][rank].item()
        meta_path_preds.append(kg['id2metapathid'][meta_path_pred])

    return meta_path_preds


@torch.no_grad()
def generate_flows(batch, token_embeds, user_embeds, meta_path_preds, num_beams):
    flow_batch = flow_data_collator(batch, meta_path_preds)
    flow_inputs = {**flow_batch['inputs'], 'token_embeds': token_embeds, 'user_embeds': user_embeds}

    if 'decoder_logits_mask' in flow_batch['inputs']:
        gen_args['max_new_tokens'] = flow_batch['inputs']['decoder_logits_mask'].shape[1]
    else:
        gen_args['max_new_tokens'] = args.max_length

    input_len = 1
    gen_args['min_length'] = input_len + 1
    if args.debug:
        gen_args['min_length'] = gen_args['max_new_tokens'] + 1

    gen_args['num_return_sequences'] = gen_args['num_beams'] = num_beams

    gen_seqs = flow_model.generate(**flow_inputs, **gen_args)[:, input_len:]
    return gen_seqs


def compute_log_prob(batch, meta_path_preds, token_embeds, user_embeds, gen_seqs, num_beams):
    flow_batch = flow_data_collator(batch, meta_path_preds)['inputs']
    labels = gen_seqs.detach().clone()
    labels[labels == kg['pad_id']] = -100
    flow_inputs = {**flow_batch, 'labels': labels, 'token_embeds': token_embeds, 'user_embeds': user_embeds}

    input_ids, model_kwargs = flow_model._expand_inputs_for_generation(
        **flow_inputs, expand_size=num_beams, is_encoder_decoder=False
    )

    neg_log_prob = flow_model(input_ids, **model_kwargs, reduction='none').loss.sum(dim=-1)  # (bs)
    return neg_log_prob


@torch.no_grad()
def data_generation(aug_dataset, token_embeds):
    aug_data_list = [data for data in aug_dataset]

    if args.flow_policy == 'model':
        # user
        user_batch = user_data_collator(aug_data_list)
        user_ids = user_batch['user_id']

        user_inputs = {**user_batch['entity'], 'token_embeds': token_embeds}
        user_embeds = user_encoder(**user_inputs)
        user_embeds = user_embeds.reshape(len(user_embeds) // 2, 2, -1)

        # meta path prediction
        meta_path_batch = meta_path_data_collator(aug_data_list)
        meta_path_preds = predict_meta_paths(
            user_embeds=user_embeds, meta_path_mask=meta_path_batch['meta_path_mask']
        )

        # flow generation
        gen_seqs = generate_flows(
            batch=aug_data_list, token_embeds=token_embeds, user_embeds=user_embeds, meta_path_preds=meta_path_preds,
            num_beams=args.aug_num
        )
        assert len(gen_seqs) == len(aug_data_list) * args.aug_num

        gen_seqs = gen_seqs.reshape(len(aug_data_list), args.aug_num, -1)

        # # filter flow
        # neg_log_probs = compute_log_prob(
        #     batch=aug_data_list, meta_path_preds=meta_path_preds,
        #     token_embeds=token_embeds, user_embeds=user_embeds, gen_seqs=gen_seqs, num_beams=1
        # )
        # probs = torch.exp(-neg_log_probs)
        #
        # prob_t = 0
        # if args.aug_prob is not None:
        #     if args.aug_prob == 'median':
        #         prob_t = torch.median(probs)
        #     else:
        #         prob_t = float(args.aug_prob)

        template_batch = flow_data_collator(aug_data_list, meta_path_preds)['template']

        flow_data_list = []
        for gen_seq_list, template_text_list, template_pos_list in zip(
                gen_seqs.tolist(), template_batch['text'], template_batch['position']
        ):
            # if prob < prob_t:
            #     continue

            for gen_seq in gen_seq_list:
                flow = []
                for idx in gen_seq:
                    if idx == kg['user_id']:
                        continue
                    elif idx == kg['bot_id']:
                        continue
                    else:
                        ent = kg['id2entity'][idx]
                        if ent not in decode_bad_tokens:
                            flow.append(ent)

                dialog = []
                flow_idx, template_idx = 0, 0
                last_ent_turn = None

                for turn, template in enumerate(template_text_list):
                    cur_sent = template
                    cur_ent_list = []

                    while '<mask>' in cur_sent and flow_idx < len(flow):
                        cur_ent = flow[flow_idx]
                        cur_ent_list.append(cur_ent)
                        cur_sent = cur_sent.replace('<mask>', cur_ent, 1)
                        flow_idx += 1
                        last_ent_turn = turn

                    dialog.append({
                        'text': cur_sent,
                        'entity': cur_ent_list
                    })

                context_list = []
                entity_list = []
                for turn in dialog:
                    if len(turn['entity']) > 0 and len(context_list) > 0:
                        for ent in turn['entity']:
                            flow_data = {
                                'epoch': epoch,
                                'context': copy.copy(context_list),
                                'entity': copy.copy(entity_list),
                                'rec': [ent],
                                'resp': turn['text']
                            }
                            flow_data_list.append(flow_data)

                    context_list.append(turn['text'])
                    entity_list.extend(turn['entity'])

    else:
        flow_data_list = []
        template_batch = template_data_collator(aug_data_list)
        for aug_data, template_text_list, template_pos_list in zip(
                aug_data_list, template_batch['text'], template_batch['position']
        ):
            user_entity_ids = aug_data['user_entity_id']
            gen_seq = user_entity_ids[0] + user_entity_ids[1]
            random.shuffle(gen_seq)

            flow = []
            for idx in gen_seq:
                if idx == kg['user_id']:
                    continue
                elif idx == kg['bot_id']:
                    continue
                else:
                    ent = kg['id2entity'][idx]
                    if ent not in decode_bad_tokens:
                        flow.append(ent)

            dialog = []
            flow_idx, template_idx = 0, 0
            last_ent_turn = None

            for turn, template in enumerate(template_text_list):
                cur_sent = template
                cur_ent_list = []

                while '<mask>' in cur_sent and flow_idx < len(flow):
                    cur_ent = flow[flow_idx]
                    cur_ent_list.append(cur_ent)
                    cur_sent = cur_sent.replace('<mask>', cur_ent, 1)
                    flow_idx += 1
                    last_ent_turn = turn

                dialog.append({
                    'text': cur_sent,
                    'entity': cur_ent_list
                })

            context_list = []
            entity_list = []
            for turn in dialog:
                if len(turn['entity']) > 0 and len(context_list) > 0:
                    for ent in turn['entity']:
                        flow_data = {
                            'epoch': epoch,
                            'context': copy.copy(context_list),
                            'entity': copy.copy(entity_list),
                            'rec': [ent],
                            'resp': turn['text']
                        }
                        flow_data_list.append(flow_data)

                context_list.append(turn['text'])
                entity_list.extend(turn['entity'])

    return flow_data_list


@torch.no_grad()
def compute_reward(flow_data_list):
    dataset_rec = DatasetForRec(
        flow_data_list, tokenizer=tokenizer, prompt_tokenizer=text_tokenizer,
        context_max_length=args.crs_context_max_length,
        entity2id=kg_crs['entity2id'], entity_max_length=args.crs_entity_max_length
    )
    dataloader_rec = DataLoader(dataset_rec, batch_size=args.crs_batch_size, collate_fn=simple_collate)

    loss_list = []
    for batch in dataloader_rec:
        batch = crs_data_collator_for_rec(batch)

        token_embeds = text_encoder(**batch['prompt']).last_hidden_state

        prompt_embeds = prompt_encoder(
            entity_ids=batch['entity'],
            token_embeds=token_embeds,
            output_entity=True
        )
        batch['context']['prompt_embeds'] = prompt_embeds
        batch['context']['entity_embeds'] = prompt_encoder.get_entity_embeds()

        loss = crs_model(**batch['context'], rec=True, reduction='none').rec_loss
        loss_list.append(loss)

    reward = torch.cat(loss_list, dim=0)
    return reward


def data_augmentation(base_dataset):
    prompt_encoder.eval()

    with torch.no_grad():
        token_embeds = token_embedding(edge_index, edge_type)

    flow_dataloader = DataLoader(base_dataset, batch_size=args.batch_size, collate_fn=simple_collate)

    aug_flow_data_list = []
    aug_user_dataset_list = []

    for batch in tqdm(flow_dataloader):
        # user embedding
        user_batch = user_data_collator(batch)
        user_ids = user_batch['user_id']
        user_entity_ids = user_batch['entity']['user_entity_ids']
        user_entity_mask = user_batch['entity']['user_entity_mask']
        user_entity_embeds = token_embeds[user_entity_ids]  # (bs * 2, ent_len, hs)

        # meta path
        meta_path_batch = meta_path_data_collator(batch)

        # edit user in the continuous space
        new_user_entity_idx = torch.randint(high=user_entity_embeds.shape[1], size=(len(user_entity_embeds),))
        user_entity_embeds_delta = torch.normal(
            mean=0, std=user_entity_embeds[user_entity_mask].std(),
            size=(len(user_entity_embeds), user_entity_embeds.shape[-1]), device=device, requires_grad=True
        )
        # edit_f = Editor(user_entity_embeds.shape[-1]).to(device)
        # edit_f = accelerator.prepare(edit_f)

        if args.user_policy == 'cf':
            if args.delta_l2_ratio is not None:
                weight_decay = args.delta_l2 * args.delta_l2_ratio ** epoch
            else:
                weight_decay = args.delta_l2
            delta_optimizer = torch.optim.SGD([user_entity_embeds_delta], lr=args.delta_lr, weight_decay=weight_decay)
            delta_optimizer = accelerator.prepare(delta_optimizer)

            # counterfactual learning for delta
            for it in tqdm(range(args.delta_iterations)):
                # user embeds
                new_user_entity_embeds = user_entity_embeds.detach().clone()
                new_user_entity_embeds[range(len(user_entity_embeds)), new_user_entity_idx] += user_entity_embeds_delta

                # new_user_entity_embeds_edit = new_user_entity_embeds[range(len(user_entity_embeds)), new_user_entity_idx]
                # new_user_entity_embeds[range(len(user_entity_embeds)), new_user_entity_idx] = edit_f(
                #     new_user_entity_embeds_edit)

                user_inputs = {
                    'user_entity_embeds': new_user_entity_embeds,
                    # 'user_entity_mask': user_entity_mask
                }
                new_user_embeds = user_encoder(**user_inputs)
                new_user_embeds = new_user_embeds.reshape(len(new_user_embeds) // 2, 2, -1)

                # meta path prediction
                meta_path_preds = predict_meta_paths(
                    user_embeds=new_user_embeds, meta_path_mask=meta_path_batch['meta_path_mask']
                )

                # flow generation
                gen_seqs = generate_flows(
                    batch=batch, token_embeds=token_embeds, user_embeds=new_user_embeds,
                    meta_path_preds=meta_path_preds,
                    num_beams=num_beams
                )

                template_batch = flow_data_collator(batch, meta_path_preds)['template']

                flow_data_list = []
                flow_preds = gen_seqs.reshape(len(batch), num_beams, -1)
                for user_list, preds, template_text_list, template_pos_list in zip(
                        user_ids, flow_preds.tolist(), template_batch['text'], template_batch['position']
                ):
                    user, bot = user_list

                    for pred in preds:
                        decoded_pred = []
                        flow_data = []
                        for idx in pred:
                            if idx == kg['user_id']:
                                decoded_pred.append(user)
                            elif idx == kg['bot_id']:
                                decoded_pred.append(bot)
                            else:
                                ent = kg['id2entity'][idx]
                                if ent not in decode_bad_tokens:
                                    decoded_pred.append(ent)
                                    flow_data.append(ent)

                        dialog = []
                        flow_idx, template_idx = 0, 0
                        last_ent_turn = None

                        for turn, template in enumerate(template_text_list):
                            cur_sent = template
                            cur_ent_list = []

                            while '<mask>' in cur_sent and flow_idx < len(flow_data):
                                cur_ent = flow_data[flow_idx]
                                cur_ent_list.append(cur_ent)
                                cur_sent = cur_sent.replace('<mask>', cur_ent, 1)
                                flow_idx += 1

                            if len(cur_ent_list) > 0:
                                last_ent_turn = turn

                            dialog.append({
                                'text': cur_sent,
                                'entity': cur_ent_list
                            })

                        context_list = []
                        entity_list = []
                        for turn in dialog[:last_ent_turn]:
                            context_list.append(turn['text'])
                            entity_list.extend(turn['entity'])

                        rec = dialog[last_ent_turn]['entity'][-1]
                        resp = dialog[last_ent_turn]['text']

                        flow_data = {
                            'epoch': epoch,
                            'iter': it,
                            'context': context_list,
                            'entity': entity_list,
                            'rec': [rec],
                            'resp': resp
                        }
                        flow_data_list.append(flow_data)

                        if log_level == 'DEBUG' and args.no_log is False:
                            log_file.write(json.dumps(flow_data, ensure_ascii=False) + '\n')

                # log_prob of each beam
                neg_log_prob = compute_log_prob(
                    batch=batch, meta_path_preds=meta_path_preds,
                    token_embeds=token_embeds, user_embeds=new_user_embeds, gen_seqs=gen_seqs, num_beams=num_beams
                )

                # reward for each beam
                reward = compute_reward(flow_data_list)
                logger.debug(f'{it}: {torch.mean(reward)}')

                loss = (neg_log_prob * reward).mean()
                accelerator.backward(loss)

                delta_optimizer.step()
                delta_optimizer.zero_grad()

            accelerator._optimizers.pop()

        # new user
        new_user_entity_embeds = user_entity_embeds.detach().clone()
        new_user_entity_embeds[range(len(user_entity_embeds)), new_user_entity_idx] += user_entity_embeds_delta

        # new_user_entity_embeds_edit = new_user_entity_embeds[range(len(user_entity_embeds)), new_user_entity_idx]
        # new_user_entity_embeds[range(len(user_entity_embeds)), new_user_entity_idx] = edit_f(
        #     new_user_entity_embeds_edit
        # )

        batch_size, ent_len, hidden_size = new_user_entity_embeds.shape

        dist_mat = torch.cdist(new_user_entity_embeds.reshape(1, -1, hidden_size), token_embeds.unsqueeze(0))[0]
        new_user_entity_ids = torch.argmin(dist_mat, dim=-1).reshape(batch_size // 2, 2, ent_len)  # (bs, 2, ent_len)

        templates = [data['template'] for data in batch]

        cur_aug_user_data_list = []
        for cur_user_ids, cur_new_user_entity_ids, template_list in zip(user_ids, new_user_entity_ids, templates):
            new_user_ids = [f'{cur_user_id.split("-")[0]}-{epoch}' for cur_user_id in cur_user_ids]
            user2entity = {}
            for new_user_id, new_user_entity_id_list in zip(new_user_ids, cur_new_user_entity_ids):
                user2entity[new_user_id] = [
                    kg['id2entity'][ent_id] for ent_id in
                    new_user_entity_id_list[new_user_entity_id_list != kg['pad_id']].tolist()
                ]
            cur_aug_user_data_list.append({
                'user': new_user_ids,
                'user2entity': user2entity,
                'template': template_list
            })
        cur_aug_user_dataset = DatasetForFlow(
            cur_aug_user_data_list, kg=kg,
            max_length=args.max_length, user_entity_max_length=args.user_entity_max_length
        )
        aug_user_dataset_list.append(cur_aug_user_dataset)

        # new flow
        cur_aug_flow_data_list = data_generation(aug_dataset=cur_aug_user_dataset, token_embeds=token_embeds)
        aug_flow_data_list.extend(cur_aug_flow_data_list)

    # deduplicate
    aug_flow_data_list_filter = []
    flow_set = set()
    for flow_data in aug_flow_data_list:
        flow_tup = tuple(flow_data['entity'] + flow_data['rec'])
        if flow_tup not in flow_set:
            flow_set.add(flow_tup)
            aug_flow_data_list_filter.append(flow_data)

            if args.no_log is False:
                log_file.write(json.dumps(flow_data, ensure_ascii=False) + '\n')

    return aug_flow_data_list, aug_user_dataset_list


@torch.no_grad()
def crs_model_evaluation(dataloader):
    prompt_encoder.eval()

    loss_list = []
    for batch in tqdm(dataloader):
        batch = crs_data_collator_for_rec(batch)

        token_embeds = text_encoder(**batch['prompt']).last_hidden_state
        prompt_embeds = prompt_encoder(
            entity_ids=batch['entity'],
            token_embeds=token_embeds,
            output_entity=True
        )
        batch['context']['prompt_embeds'] = prompt_embeds
        batch['context']['entity_embeds'] = prompt_encoder.get_entity_embeds()
        outputs = crs_model(**batch['context'], rec=True)

        loss_list.append(float(outputs.rec_loss))

        logits = outputs.rec_logits[:, item_ids_crs]
        ranks = torch.topk(logits, k=50, dim=-1).indices
        preds = item_ids_crs[ranks].tolist()
        labels = batch['context']['rec_labels'].tolist()
        evaluator.evaluate(preds, labels)

    report = evaluator.report()
    report['loss'] = np.mean(loss_list)

    return report


def crs_model_rec_training(training_dataset_for_crs, best_metric, epoch):
    dataloader_rec = DataLoader(
        training_dataset_for_crs, batch_size=args.crs_batch_size, collate_fn=simple_collate, shuffle=True
    )

    crs_optimizer = torch.optim.AdamW(prompt_encoder.parameters(), lr=args.crs_rec_lr, weight_decay=args.crs_l2)
    crs_lr_scheduler = None
    if args.crs_rec_iterations > 1:
        crs_lr_scheduler = get_linear_schedule_with_warmup(
            crs_optimizer, len(dataloader_rec), len(dataloader_rec) * args.crs_rec_iterations
        )

    for it in range(args.crs_rec_iterations):
        prompt_encoder.train()

        train_loss_list = []
        for batch in tqdm(dataloader_rec):
            batch = crs_data_collator_for_rec(batch)

            with torch.no_grad():
                token_embeds = text_encoder(**batch['prompt']).last_hidden_state

            prompt_embeds = prompt_encoder(
                entity_ids=batch['entity'],
                token_embeds=token_embeds,
                output_entity=True
            )
            batch['context']['prompt_embeds'] = prompt_embeds
            batch['context']['entity_embeds'] = prompt_encoder.get_entity_embeds()

            loss = crs_model(**batch['context'], rec=True, reduction='mean').rec_loss
            loss.backward()
            train_loss_list.append(float(loss))

            crs_optimizer.step()
            if crs_lr_scheduler is not None:
                crs_lr_scheduler.step()
            crs_optimizer.zero_grad()
            if run:
                run.log({'loss': np.mean(train_loss_list)})

        train_loss = np.mean(train_loss_list)
        logger.info(f'rec iter {it} train loss {train_loss}')

        # evaluation
        # validation set
        report = crs_model_evaluation(crs_valid_dataloader)
        valid_report = {}
        for k, v in report.items():
            valid_report[f'valid/{k}'] = v

        valid_report['epoch'] = epoch + it
        logger.info(f'{valid_report}')
        if run:
            run.log(valid_report)
        evaluator.reset_metric()

        # save model with the best performance
        if valid_report[f'valid/{metric}'] * mode > best_metric * mode:
            prompt_encoder.save(best_metric_dir)
            best_metric = valid_report[f'valid/{metric}']
            logger.info(f'new best model with {metric}')

        # test set
        report = crs_model_evaluation(crs_test_dataloader)
        test_report = {}
        for k, v in report.items():
            test_report[f'test/{k}'] = v

        test_report['epoch'] = epoch + it
        logger.info(f'{test_report}')
        if run:
            run.log(test_report)
        evaluator.reset_metric()

    return best_metric


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
    log_level = 'DEBUG'
    if args.name is not None:
        log_level = 'INFO'
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    logger.add(f'log/{local_time}.log', level='DEBUG')

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
    kg = KGForCF(kg_dataset=args.kg_dataset, debug=args.debug).get_kg_info()
    edge_index = torch.as_tensor(kg['edge_index'], device=device)
    edge_type = torch.as_tensor(kg['edge_type'], device=device)

    kg_crs = KGForUniCRS(kg=args.crs_kg_dataset, debug=args.debug).get_kg_info()
    edge_index_crs = torch.as_tensor(kg_crs['edge_index'], device=device)
    edge_type_crs = torch.as_tensor(kg_crs['edge_type'], device=device)
    item_ids_crs = torch.as_tensor(kg_crs['item_ids'])

    # model
    # generation
    flow_model = BartForFlowGeneration.from_pretrained(f'{args.pretrain_dir}/flow').to(device)
    config = flow_model.config

    token_embedding = TokenEmbedding(
        hidden_size=config.hidden_size, num_bases=args.num_bases,
        num_relations=kg['num_relations'], num_entities=kg['num_entities'], num_special_tokens=kg['num_special_tokens']
    )
    token_embedding.load(f'{args.pretrain_dir}/token')
    token_embedding = token_embedding.to(device)

    user_encoder = UserEncoder(hidden_size=config.hidden_size)
    user_encoder.load(f'{args.pretrain_dir}/user')
    user_encoder.to(device)


    meta_path_predictor = MetaPathPredictor(hidden_size=config.hidden_size, num_labels=kg['num_meta_paths']).to(device)
    meta_path_predictor.load(args.meta_path_model)
    meta_path_predictor = meta_path_predictor.to(device)

    generation_models = [token_embedding, user_encoder, meta_path_predictor, flow_model]
    for i, model in enumerate(generation_models):
        model.requires_grad_(False)
        generation_models[i] = accelerator.prepare(model)

    # crs
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.add_special_tokens(gpt2_special_tokens_dict)
    tokenizer.padding_side = 'left'

    crs_model = PromptGPT2forCRS.from_pretrained(args.crs_model)
    crs_model.resize_token_embeddings(len(tokenizer))
    crs_model.config.pad_token_id = tokenizer.pad_token_id
    crs_model = crs_model.to(device).requires_grad_(False)
    crs_model_config = crs_model.config
    # crs_model_config = PromptGPT2forCRS.config_class.from_pretrained(args.crs_model)

    text_tokenizer = AutoTokenizer.from_pretrained(args.text_tokenizer)
    text_tokenizer.add_special_tokens(prompt_special_tokens_dict)

    text_encoder = AutoModel.from_pretrained(args.crs_text_encoder)
    text_encoder.resize_token_embeddings(len(text_tokenizer))
    text_encoder = text_encoder.to(device).requires_grad_(False)
    text_encoder_config = text_encoder.config
    # text_encoder_config = AutoConfig.from_pretrained(args.crs_text_encoder)

    # prompt encoder
    prompt_encoder = KGPrompt(
        crs_model_config.n_embd, text_encoder_config.hidden_size, crs_model_config.n_head, crs_model_config.n_layer,
        2,
        n_entity=kg_crs['num_entities'], num_relations=kg_crs['num_relations'], num_bases=args.num_bases,
        edge_index=kg_crs['edge_index'], edge_type=kg_crs['edge_type'],
    )
    prompt_encoder.load(args.crs_prompt_encoder)
    prompt_encoder = prompt_encoder.to(device)

    # data
    # flow
    data_file = os.path.join('data', args.flow_dataset, f'{args.split}_data_processed.jsonl')
    data_list = load_jsonl_data(data_file)
    dataset = DatasetForFlow(
        data_list=data_list, kg=kg, debug=args.debug, shot=args.shot,
        max_length=args.max_length, user_entity_max_length=args.user_entity_max_length
    )

    user_data_collator = UserDataCollator(
        kg=kg, user_entity_max_length=args.user_entity_max_length, device=device, debug=args.debug
    )
    meta_path_data_collator = MetaPathDataCollator()
    flow_data_collator = FlowDataCollator(
        kg=kg, max_length=args.max_length,
        device=device, debug=args.debug,
        use_meta_path=args.use_meta_path
    )
    template_data_collator = TemplateDataCollator()

    # crs
    crs_pre_data_file = os.path.join('data', args.crs_pre_dataset, 'train_data_processed.jsonl')
    crs_pre_data_list = load_jsonl_data(crs_pre_data_file)
    crs_pre_dataset = DatasetForRec(
        crs_pre_data_list, tokenizer=tokenizer, prompt_tokenizer=text_tokenizer,
        context_max_length=args.crs_context_max_length, resp_max_length=args.crs_resp_max_length,
        entity2id=kg_crs['entity2id'], entity_max_length=args.crs_entity_max_length,
        debug=args.debug,
        shot=args.shot
    )

    crs_train_data_file = os.path.join('data', args.crs_rec_dataset, 'train_data_processed.jsonl')
    crs_train_data_list = load_jsonl_data(crs_train_data_file)
    crs_train_dataset = DatasetForRec(
        crs_train_data_list, tokenizer=tokenizer, prompt_tokenizer=text_tokenizer,
        context_max_length=args.crs_context_max_length, resp_max_length=args.crs_resp_max_length,
        entity2id=kg_crs['entity2id'], entity_max_length=args.crs_entity_max_length,
        debug=args.debug,
        shot=args.shot
    )

    crs_valid_data_file = os.path.join('data', args.crs_rec_dataset, 'valid_data_processed.jsonl')
    crs_valid_data_list = load_jsonl_data(crs_valid_data_file)
    crs_valid_dataset = DatasetForRec(
        crs_valid_data_list, tokenizer=tokenizer, prompt_tokenizer=text_tokenizer,
        context_max_length=args.crs_context_max_length, resp_max_length=args.crs_resp_max_length,
        entity2id=kg_crs['entity2id'], entity_max_length=args.crs_entity_max_length,
        # shot=args.shot
    )
    crs_valid_dataloader = DataLoader(crs_valid_dataset, batch_size=args.crs_batch_size, collate_fn=simple_collate)

    crs_test_data_file = os.path.join('data', args.crs_rec_dataset, 'test_data_processed.jsonl')
    crs_test_data_list = load_jsonl_data(crs_test_data_file)
    crs_test_dataset = DatasetForRec(
        crs_test_data_list, tokenizer=tokenizer, prompt_tokenizer=text_tokenizer,
        context_max_length=args.crs_context_max_length, resp_max_length=args.crs_resp_max_length,
        entity2id=kg_crs['entity2id'], entity_max_length=args.crs_entity_max_length,
        # shot=args.shot
    )
    crs_test_dataloader = DataLoader(crs_test_dataset, batch_size=args.crs_batch_size, collate_fn=simple_collate)

    crs_data_collator_for_pre = UniCRSDataCollatorForPreTraining(
        tokenizer=tokenizer, prompt_tokenizer=text_tokenizer,
        context_max_length=args.crs_context_max_length + args.crs_resp_max_length,
        entity_pad_id=kg_crs['pad_entity_id'], entity_max_length=args.crs_entity_max_length,
        device=device, debug=args.debug, use_amp=accelerator.use_fp16
    )
    crs_data_collator_for_rec = UniCRSDataCollatorForRec(
        tokenizer=tokenizer, prompt_tokenizer=text_tokenizer, context_max_length=args.crs_context_max_length,
        entity_max_length=args.crs_entity_max_length, entity_pad_id=kg_crs['pad_entity_id'],
        device=device, debug=args.debug, use_amp=accelerator.use_fp16
    )

    # evaluation
    evaluator = RecEvaluator()

    # params for generation
    gen_args = {
        'no_repeat_ngram_size': 2,
        # 'do_sample': True,
        # 'top_k': 3,
    }
    num_beams = args.num_beams
    decode_bad_tokens = {'<pad>', '<eos>', '<sep>', '<bos>', 'genre', 'movie', 'person'}

    # log
    if args.name is not None:
        log_file_dir = 'save'
        os.makedirs(log_file_dir, exist_ok=True)
        log_file_path = os.path.join(log_file_dir, f'{args.name}.jsonl')
        log_file = open(log_file_path, 'w', encoding='utf-8')
    else:
        log_file_dir = 'log'
        os.makedirs(log_file_dir, exist_ok=True)
        log_file_path = os.path.join(log_file_dir, f'gen_{local_time}.jsonl')
        log_file = open(log_file_path, 'w', encoding='utf-8')

    # save model with best metric
    metric, mode = 'recall@50', 1
    assert mode in (-1, 1)
    if mode == 1:
        best_metric = 0
    else:
        best_metric = float('inf')
    best_metric_dir = os.path.join(args.output_dir, 'best')

    base_dataset = dataset
    aug_user_dataset_list = None

    for epoch in tqdm(range(args.num_epochs)):
        # data augmentation
        if aug_user_dataset_list is not None:
            base_dataset = ConcatDataset(aug_user_dataset_list)

        aug_flow_data_list, aug_user_dataset_list = data_augmentation(base_dataset)

        # accelerator.free_memory()

        # training
        # prompt_encoder = KGPrompt(
        #     crs_model_config.n_embd, text_encoder_config.hidden_size, crs_model_config.n_head, crs_model_config.n_layer,
        #     2,
        #     n_entity=kg_crs['num_entities'], num_relations=kg_crs['num_relations'], num_bases=args.num_bases,
        #     edge_index=kg_crs['edge_index'], edge_type=kg_crs['edge_type'],
        # ).to(device)

        aug_train_dataset = DatasetForRec(
            aug_flow_data_list, tokenizer=tokenizer, prompt_tokenizer=text_tokenizer,
            context_max_length=args.crs_context_max_length, resp_max_length=args.crs_resp_max_length,
            entity2id=kg_crs['entity2id'], entity_max_length=args.crs_entity_max_length
        )

        # cur_pre_dataset = ConcatDataset([crs_pre_dataset, aug_train_dataset])

        if args.strategy == 'mix':
            cur_train_dataset = ConcatDataset([crs_train_dataset, aug_train_dataset])
            best_metric = crs_model_rec_training(cur_train_dataset, best_metric, epoch * args.crs_rec_iterations)
        elif args.strategy == 'pre':
            best_metric = crs_model_rec_training(aug_train_dataset, best_metric, epoch * 2 * args.crs_rec_iterations)
            best_metric = crs_model_rec_training(crs_train_dataset, best_metric, (epoch * 2 + 1) * args.crs_rec_iterations)
        else:
            raise Exception('do not support this strategy')

        # accelerator.free_memory()

    # save the final model
    final_dir = os.path.join(args.output_dir, 'final')
    prompt_encoder.save(final_dir)
    logger.info(f'save final model')
