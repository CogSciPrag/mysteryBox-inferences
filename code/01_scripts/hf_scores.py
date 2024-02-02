import time
import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm
from pprint import pprint
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer
)
# get device count
num_devices = torch.cuda.device_count()
print("NUM DEVICES: ", num_devices)
# print all device names and info
for i in range(num_devices):
    print(torch.cuda.get_device_properties(i))

# define logsoftmax for retrieving logprobs from scores
logsoftmax = torch.nn.LogSoftmax(dim=-1)

def getLogProbContinuation(
        initialSequence, 
        continuation, 
        model,
        tokenizer,
        model_name='',
        preface = ''):
    """
    Helper for retrieving log probability of different response types from Llama-2 of various sizes.
    """

    initialSequence = preface + initialSequence
    prompt = preface + initialSequence + continuation
    # for mistral / mixtral instruct the formatting is a bit different
    if "Instruct" in model_name:
        # split the last continuation sentence from the overall prompt
        prompt_separate = "\n".join(initialSequence.split("\n")[:-1])
        completion_separate = initialSequence.split("\n")[-1]
       
        messages = [
            {"role": "user", "content": prompt_separate},
            {"role": "assistant", "content": completion_separate + continuation}
        ]

        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda:0")

        # tokenize separately, so as to know the shape of the continuation
        messages_noCont = [
            {"role": "user", "content": prompt_separate},
            {"role": "assistant", "content": completion_separate.strip()}
        ]
        input_ids_prompt = tokenizer.apply_chat_template(
            messages_noCont, 
            return_tensors="pt",
        )
    # the tested non-mistral models don't prepend the BOS token, so it is done manually
    elif "mistral" not in model_name:
        raw_input_ids = tokenizer.encode(
            prompt,
            add_special_tokens=False
        )
        # print(len(input_ids))
        input_ids = torch.tensor(
            [tokenizer.bos_token_id] + raw_input_ids
        ).unsqueeze(0).to("cuda:0")
        raw_input_ids_prompt = tokenizer.encode(
            initialSequence.strip(),
            add_special_tokens=False
        )
        input_ids_prompt = torch.tensor(
            [tokenizer.bos_token_id] + raw_input_ids_prompt
        ).unsqueeze(0)

    else:

        # tokenize separately, so as to know the shape of the continuation
        input_ids_prompt = tokenizer(
            initialSequence.strip(), 
            return_tensors="pt",
        ).input_ids
        input_ids = tokenizer(
            prompt,
            return_tensors="pt",
        ).input_ids.to("cuda:0")
    # pass through model
    with torch.no_grad():
        outputs = model(
            input_ids,
        )
    # transform logits to probabilities
    # remove the EOS logit which we aren't interested in
    llama_output_scores = logsoftmax(
        outputs.logits[0][:-1]
    )
    # retreive log probs at token ids
    # transform input_ids to a tensor of shape [n_tokens, 1] for this
    # cut off the sos token so as to get predictions for the actual token conditioned on 
    # preceding context
    input_ids_probs = input_ids[:, 1:].squeeze().unsqueeze(-1)
    # retreive at correct token positions
    conditionalLogProbs = torch.gather(
        llama_output_scores, 
        dim=-1, 
        index=input_ids_probs
    ).flatten()
    # print("input ids continuation without sos :" , input_ids_continuation.shape[-1]-1)
    # slice output to only get scores of the continuation, not the context
    continuationConditionalLogProbs = conditionalLogProbs[
        (input_ids_prompt.shape[-1]-1):
    ]
    # compute continunation log prob
    sentLogProb = torch.sum(continuationConditionalLogProbs).item()
    meanLogProb = torch.mean(continuationConditionalLogProbs).item()
    
    return sentLogProb, meanLogProb
            

def soft_max(scores, alpha=1):
    scores = np.array(scores)
    output = np.exp(scores * alpha)
    return(output / np.sum(output))


def get_hf_model_predictions(
        prompt, 
        answer_good,
        answer_bad,
        model,
        tokenizer,
        model_name,
    ):

    # take care of special tokens for chat models
    # assume that the task and context come from the user, and the response from the model
    # no specific system prompt is passed
    # if one wanted to, the expected formatting would be: [INST]<<SYS>>{system prompt}<</SYS>>\n\n{user message}[/INST]

    # get scores
    log_prob_good, mean_log_prob_good  = getLogProbContinuation(
        prompt, 
        answer_good,
        model, 
        tokenizer,
        model_name=model_name,
    )
    log_prob_bad, mean_log_prob_bad  = getLogProbContinuation(
        prompt, 
        answer_bad,
        model, 
        tokenizer,
        model_name=model_name,
    )
    
    output = {
        "Mean_logprob_answer_good": mean_log_prob_good,
        "Mean_logprob_answer_bad": mean_log_prob_bad,
        "Sentence_logprob_answer_good": log_prob_good,
        "Sentence_logprob_answer_bad": log_prob_bad
    }

    return output

