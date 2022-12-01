#!/bin/bash

while getopts s:i: flag
do
    case "${flag}" in
        s) seed=${OPTARG};;
        i) split_id=${OPTARG};;
    esac
done

level_list=( "BossLevel" )

for level_name in "${level_list[@]}"
do
    echo BabyAI-${level_name}-v0-split-${split_id}
    python ./scripts/make_agent_demos.py --env BabyAI-${level_name}-v0 \
        --demos ../data/babyai_task/BabyAI-${level_name}-v0-split-${split_id} \
        --episodes 2500 \
        --valid-episodes 500 \
        --seed $((${seed} * 10000000))
done

