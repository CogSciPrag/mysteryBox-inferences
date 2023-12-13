import openai
import time
import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm
import openai
from pprint import pprint
from datetime import datetime
from utils import format_item
from dotenv import load_dotenv

# OAI = "bla"
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def getLogProbContinuation(initialSequence, continuation, preface = ''):
    """
    Helper for retrieving log probability of different response types from GPT-3.
    """
    initialSequence = preface + initialSequence
    response = openai.Completion.create(
            engine      = "text-davinci-003",
            prompt      = initialSequence + " " + continuation,
            max_tokens  = 0,
            temperature = 0.1,
            logprobs    = 0,
            echo        = True
        )
    text_offsets = response.choices[0]['logprobs']['text_offset']
    cutIndex = text_offsets.index(max(i for i in text_offsets if i < len(initialSequence))) + 1
    endIndex = response.usage.total_tokens
    answerTokens = response.choices[0]["logprobs"]["tokens"][cutIndex:endIndex]
    answerTokenLogProbs = response.choices[0]["logprobs"]["token_logprobs"][cutIndex:endIndex]
    meanAnswerLogProb = np.mean(answerTokenLogProbs)
    sentenceLogProb = np.sum(answerTokenLogProbs)
    print("---> answer tokens ", answerTokens)
    return meanAnswerLogProb, sentenceLogProb, (endIndex - cutIndex)

def soft_max(scores, alpha=1):
    scores = np.array(scores)
    output = np.exp(scores * alpha)
    return(output / np.sum(output))


def get_openai_model_predictions(prompt, answer_good, answer_bad, **kwargs):
    # get model predictions for label "good"
    mean_log_prob_good, log_prob_good, _ = getLogProbContinuation(
        prompt, answer_good
    ) 
    print("------ Logprobs for GOOD ----- ", mean_log_prob_good, log_prob_good)
    # get model predictions for label "bad"
    mean_log_prob_bad, log_prob_bad, _ = getLogProbContinuation(
        prompt, answer_bad
    ) 
    print("------ Logprobs for BAD ----- ", mean_log_prob_bad, log_prob_bad)

    output = {
        "Mean_logprob_answer_good": mean_log_prob_good,
        "Mean_logprob_answer_bad": mean_log_prob_bad,
        "Sentence_logprob_answer_good": log_prob_good,
        "Sentence_logprob_answer_bad": log_prob_bad
    }

    return output
