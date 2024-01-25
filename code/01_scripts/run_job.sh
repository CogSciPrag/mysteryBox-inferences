#!/bin/bash
#SBATCH --partition=single
#SBATCH --ntasks=1
#SBATCH --time=05:00:00
#SBATCH --mem=60gb
#SBATCH --gres=gpu:A40:1

echo 'Running simulation'


# Load conda
module load devel/miniconda/3
source $MINICONDA_HOME/etc/profile.d/conda.sh

conda deactivate
# Activate the conda environment
conda activate llmlink

echo "Conda environment activated:"
echo $(conda env list)
echo " "
echo "Python version:"
echo $(which python)
echo " "

# activate CUDA
module load devel/cuda/11.6

# iterate over studies and model s
studies=("Martyetal2022" "Martyetal2023")
models=("meta-llama/Llama-2-7b-hf" "meta-llama/Llama-2-13b-hf" "meta-llama/Llama-2-7b-chat-hf" "meta-llama/Llama-2-13b-chat-hf") #"meta-llama/Llama-2-7b-hf"
hf_models=("mistralai/Mistral-7B-Instruct-v0.2") # "mistralai/Mistral-7B-v0.1") # EleutherAI/pythia-6.9b "microsoft/phi-2" "mistralai/Mistral-7B-Instruct-v0.2" "mistralai/Mistral-7B-Instruct-v0.1" "mistralai/Mixtral-8x7B-Instruct-v0.1" "mistralai/Mixtral-8x7B-v0.1") # "tiiuae/falcon-7b"  "opt" "dialoGPT"
# iterate over experiments
expts2024=("Exp_1" "Exp_2" "Exp_3")
expts2023=("Exp_1" "Exp_2")
expts2022=("Exp_4" "Exp_5" "Exp_6")

for i in ${!hf_models[*]}; do
    echo "model: ${hf_models[$i]}"
    for j in ${!studies[*]}; do
        echo "study: ${studies[$j]}"
        case "${studies[$j]}" in 
            "Deganoetal2024")
                for k in ${!expts2024[*]}; do
                python3 -u main.py \
                    --model_name="${hf_models[$i]}" \
                    --study_name="${studies[$j]}" \
                    --experiment_name="${expts2024[$k]}"
                done
            ;;
            "Martyetal2023")
                for k in ${!expts2023[*]}; do
                python3 -u main.py \
                    --model_name="${hf_models[$i]}" \
                    --study_name="${studies[$j]}" \
                    --experiment_name="${expts2023[$k]}"
                done
            ;;
            "Martyetal2022")
                for k in ${!expts2022[*]}; do
                python3 -u main.py \
                    --model_name="${hf_models[$i]}" \
                    --study_name="${studies[$j]}" \
                    --experiment_name="${expts2022[$k]}"
                done
        esac
    done
done
