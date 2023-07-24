#!/bin/bash

while getopts s:i:l:d: flag
do
    case "${flag}" in
        s) seed=${OPTARG};;
        i) split_id=${OPTARG};;
        l) level_name=${OPTARG};;
        d) demos=${OPTARG};;
    esac
done

echo BabyAI-${level_name}-v0-split-${split_id}
python ./scripts/make_agent_demos.py --env BabyAI-${level_name}-v0 \
    --demos ${demos} \
    --episodes 2500 \
    --valid-episodes 500 \
    --seed $((${seed} * 1000000))

#--demos ../data/${level_name}/BabyAI-${level_name}-v0-split-${split_id} \