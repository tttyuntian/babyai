#!/bin/bash

while getopts s:i:l: flag
do
    case "${flag}" in
        s) seed=${OPTARG};;
        i) split_id=${OPTARG};;
        l) level_name=${OPTARG};;
    esac
done

echo BabyAI-${level_name}-v0-seed-${seed}
python ./scripts/make_agent_demos_rule_probing.py --env BabyAI-${level_name}-v0 \
    --demos ../data/exp3_data/BabyAI-${level_name}-v0/seed-${seed} \
    --episodes 1000 \
    --valid-episodes 0 \
    --seed $((${seed} * 1000000))