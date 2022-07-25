#!/bin/bash

level_list=( "GoToLocal" )
#level_list=( "BossLevel" "PickupLoc" "GoToObjMaze" "GoTo" "Pickup" "UnblockPickup" "Open" "Unlock" "PutNext" "Synth" "SynthLoc" "GoToSeq" "SynthSeq" "GoToImpUnlock" )

#level_list=( "GoToObj" "GoToRedBall" "GoToRedBallGrey" "GoToLocal" "PutNextLocal" "PickUpLoc" "GoToObjMaze" "GoTo" "Pickup" "UnblockPickup" "Open" "Unlock" "PutNext" "Synth" "SynthLoc" "GoToSeq" "SynthSeq" "GoToImpUnlock" "BossLevel" )
#level_list=( "PickUpLoc" "GoToObjMaze" "GoTo" "Pickup" "UnblockPickup" "Open" "Unlock" "PutNext" "Synth" "SynthLoc" "GoToSeq" "SynthSeq" "GoToImpUnlock" "BossLevel" )
#level_list=( "GoTo" )

for level_name in "${level_list[@]}"
do
    echo BabyAI-${level_name}-v0
    python ./scripts/make_evaluation_demos.py --env BabyAI-${level_name}-v0 \
        --demos ../data/task_0/${level_name}/positive \
        --episodes 25000 \
        --valid_episodes 5000 \
        --seed 0
done

for level_name in "${level_list[@]}"
do
    echo BabyAI-${level_name}-v0
    python ./scripts/make_evaluation_demos.py --env BabyAI-${level_name}-v0 \
        --demos ../data/task_0/${level_name}/negative \
        --is_negative \
        --episodes 25000 \
        --valid_episodes 5000 \
        --seed 10000000
done
