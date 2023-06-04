mask_token = '<movie>'

gpt2_special_tokens_dict = {
    'pad_token': '<pad>',
    'additional_special_tokens': [mask_token],
}

prompt_special_tokens_dict = {
    'additional_special_tokens': [mask_token],
}
