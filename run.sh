#!/bin/bash

MODEL_PATH="data/paligemma-weights/paligemma-3b-pt-224"
PROMPT="what is the type of this eye?"
IMAGE_FILE_PATH="data/test_imgs/img001.jpg"
DO_SAMPLE="FALSE"
ONLY_CPU="FALSE"
TOP_P=0.90
TEMPERATURE=0.8
MAX_TOKENS_TO_GENERATE=100


python inference.py \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --image_file_path "$IMAGE_FILE_PATH" \
    --max_tokens_to_generate $MAX_TOKENS_TO_GENERATE \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --do_sample $DO_SAMPLE \
    --only_cpu $ONLY_CPU