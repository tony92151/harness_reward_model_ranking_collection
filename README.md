# harness_reward_model_ranking_collection


# Usage

### For huggingface 

```bash
export HARNESS_HF_CACHE="../harness_cache_20240427"

# gsm8k
export TASK=gsm8k
export FEWSHOT=5
export MAX_NEW_TOKEN=400
export MODEL=openchat/openchat_3.5
export MODEL_ARG="pretrained=${MODEL}"
export OUTNAME=$(python -c "print('${MODEL}'.replace('/','_'))") 

python -m lm_eval \
    --model hf \
    --model_args $MODEL_ARG \
    --gen_kwargs "max_new_tokens=${MAX_NEW_TOKEN}" \
    --tasks $TASK \
    --num_fewshot $FEWSHOT \
    --device cuda:0 \
    --batch_size 4 \
    --limit 0.2 \
    --output_path ../output_20240427/$TASK/$OUTNAME

```


### For Reward Model Ranking

```bash

export HARNESS_HF_CACHE="../harness_cache_20240427"

# gsm8k
export TASK=gsm8k
export FEWSHOT=5

export rm_models="NousResearch/Nous-Hermes-2-SOLAR-10.7B|openchat/openchat_3.5" # use '|' to seperate
export rm_name=ultrarm
#export rm_name=oassterm
#export rm_name=autoj
export MODEL_ARG="rm_name=${rm_name},models=${rm_models}"
export OUTNAME=$(python -c "print('${rm_name}_'+'${rm_models}'.replace('/','-').replace('|','_'))") 

python -m lm_eval \
    --model reward_model_ranking \
    --model_args $MODEL_ARG \
    --tasks $TASK \
    --num_fewshot $FEWSHOT \
    --device cuda:0 \
    --limit 0.2 \
    --output_path ../output_20240422/$TASK/$OUTNAME



```
