from pprint import pprint
from datetime import datetime
import argparse
import os
from utils import format_item
from dotenv import load_dotenv
import pandas as pd
import time
from openai_scores import get_openai_model_predictions
from llama_scores import get_llama_model_predictions
import openai
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer
)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def main(
        model_name,
        study_name,
        experiment_name,
):
    """
    Entrypoint for running the experiments.
    """
    model_name_out = model_name.split('/')[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_name = f'../results/{study_name}_{experiment_name}_{model_name_out}_{timestamp}.csv'

    # set up llama
    if "llama" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map='auto', 
            torch_dtype=torch.float16
        )
        model.eval()
        print("----- model dtype ------", model.dtype)

    # read the study items
    data = pd.read_csv(
        f"../data/stimuli/Input_{study_name}.csv"
    )
    # read the prompts
    instructions_path = f"../data/prompts/{study_name}_instructions.txt"
    item_template_path = f"../data/prompts/{study_name}_itemTemplate.txt"
    with open(instructions_path, "r") as f:
        instructions = f.read()
    with open(item_template_path, "r") as f:
        item_template = f.read()

    # subset to desired experiment
    # TODO for running on the server, script should iterate over all expts automatically
    if experiment_name != "":
        data = data[data["Experiment"] == experiment_name]
    
    # get few--shot and critical trials sepatately
    few_shot_trials = data[data["Item_status"] == "Training"]
    critical_trials = data[data["Item_status"] == "Test"].reset_index(drop=True)

    # iterate over critical trials
    for i, row in critical_trials.iterrows():
        # create few-shot example by shuffling few shot trials
        few_shot_shuffled = few_shot_trials.sample(frac=1).reset_index(drop=True)
        few_shot_shuffled_item_ids = "|".join(
            [str(j) for j in few_shot_shuffled["Item_nr"].tolist()]
        )
        # format all of them into item template
        # note that for few shot items, the correct response is added from the Condition column 
        # (as instructed by authors)
        few_shot_items = [
            format_item(r, item_template) + r["Condition"].lower() for _, r in few_shot_shuffled.iterrows()
        ]
        # join to a string
        few_shot_items_formatted = "\n\n".join(few_shot_items)
        #print("Few shot formatted example items")
        #pprint(few_shot_items_formatted)

        # format critical trial
        trial_formatted = format_item(row, item_template)
        #print("Formatted critical trial")
        #pprint(trial_formatted)
        # construct overall prompt
        prompt = instructions.format(
            few_shot_trials=few_shot_items_formatted,
            critical_trial=trial_formatted,
        )
        # print("Overall prompt ")
        #pprint(prompt)

        # get log prob of answer options
        if "davinci" in model_name:
            predictions = get_openai_model_predictions(
                prompt, 
                answer_good="good", 
                answer_bad="bad",
            )
        else:
            predictions = get_llama_model_predictions(
                prompt, 
                answer_good="good", 
                answer_bad="bad",
                model=model,
                tokenizer=tokenizer,
                model_name=model_name,
            )
        trial = {
            "Few_shot_items_order": few_shot_shuffled_item_ids
        }
               
        output = dict(**trial, **predictions)
        print("OUTPUT")
        pprint(output)
        # continuous saving
        # combine the raw trial information with the predicted log probs
        results_df = pd.concat(
            [row.to_frame().T, 
            pd.DataFrame(output, index=[i])],
            axis=1
        )
        # pprint(results_df)

        if os.path.exists(out_name):
            results_df.to_csv(
                out_name, 
                index=False,
                mode="a",
                header=False,
            )
        else:
            results_df.to_csv(
                out_name, 
                index=False,
                mode="a",
                header=True,
            )
        # sleep in order to avoid request timeouts for OpenAI API
        if "davinci" in model_name:
            time.sleep(10)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="text-davinci-003", 
        help="Model name"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="Exp_4",
        help="Number of experiment for filtering materials"
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default="Martyetal2022",
        help="Name of study for accessing correct materials"
    )

    args = parser.parse_args()

    main(
        args.model_name, 
        args.study_name,
        args.experiment_name,
    )
