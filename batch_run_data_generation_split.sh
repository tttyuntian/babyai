#!/bin/bash

while getopts s:i:l: flag
do
    case "${flag}" in
        s) seed=${OPTARG};;
        i) split_id=${OPTARG};;
        l) level_name=${OPTARG};;
    esac
done

echo BabyAI-${level_name}-v0-split-${split_id}
python ./scripts/make_agent_demos.py --env BabyAI-${level_name}-v0 \
    --demos ../data/${level_name}/BabyAI-${level_name}-v0-split-${split_id} \
    --episodes 2500 \
    --valid-episodes 500 \
    --seed $((${seed} * 10000000))