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
        preface = ''):
    """
    Helper for retrieving log probability of different response types from Llama-2 of various sizes.
    """

    initialSequence = preface + initialSequence
    # tokenize separately, so as to know the shape of the continuation
    input_ids_prompt = tokenizer(
        initialSequence, 
        return_tensors="pt",
    ).input_ids
    input_ids_continuation = tokenizer(
        continuation,
        return_tensors="pt",
    ).input_ids
    #print("input_ids prompt ", input_ids_prompt)
    print("input ids continuation ", input_ids_continuation.shape, input_ids_continuation)
    # cut off the first token of the continuation, as it is SOS
    input_ids = torch.cat(
        (input_ids_prompt, input_ids_continuation[:, 1:]), 
        -1
    ).to("cuda:1") # put input on the first device
    print("input ids shape ", input_ids.shape)
    # pass through model
    outputs = model(
        input_ids,
    )
    # transform logits to probabilities
    print("shape of logits ", outputs.logits.shape)
    # remove the EOS logit which we aren't interested in
    llama_output_scores = logsoftmax(
        outputs.logits[0][:-1]
    )
    print("output log probs shape ", llama_output_scores.shape)
    # retreive log probs at token ids
    # transform input_ids to a tensor of shape [n_tokens, 1] for this
    # cut off the sos token so as to get predictions for the actual token conditioned on 
    # preceding context
    input_ids_probs = input_ids[:, 1:].squeeze().unsqueeze(-1)
    print("shape of input ids for porb retrieval ", input_ids_probs.shape)
    # retreive at correct token positions
    conditionalLogProbs = torch.gather(
        llama_output_scores, 
        dim=-1, 
        index=input_ids_probs
    ).flatten()
    print(len(conditionalLogProbs))
    print("input_ids_prompt  shape", input_ids_prompt.shape)
    print("input ids continuation without sos :" , input_ids_continuation.shape[-1]-1)
    # slice output to only get scores of the continuation, not the context
    continuationConditionalLogProbs = conditionalLogProbs[
        -(input_ids_continuation.shape[-1]-1):
    ]
    print("Shape of retrieved log probs", continuationConditionalLogProbs.shape)
    print(continuationConditionalLogProbs)
    # compute continunation log prob
    sentLogProb = torch.sum(continuationConditionalLogProbs).item()
    meanLogProb = torch.mean(continuationConditionalLogProbs).item()
    print("sent log prob ", sentLogProb)

    return sentLogProb, meanLogProb
            

def soft_max(scores, alpha=1):
    scores = np.array(scores)
    output = np.exp(scores * alpha)
    return(output / np.sum(output))


def get_llama_model_predictions(
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
    if "chat" in model_name:
        prompt = f"[INST]{prompt}[/INST]"

    # get scores
    log_prob_good, mean_log_prob_good  = getLogProbContinuation(
        prompt, 
        answer_good,
        model, 
        tokenizer
    )
    log_prob_bad, mean_log_prob_bad  = getLogProbContinuation(
        prompt, 
        answer_bad,
        model, 
        tokenizer
    )
    
    output = {
        "Mean_logprob_answer_good": mean_log_prob_good,
        "Mean_logprob_answer_bad": mean_log_prob_bad,
        "Sentence_logprob_answer_good": log_prob_good,
        "Sentence_logprob_answer_bad": log_prob_bad
    }

    return output

