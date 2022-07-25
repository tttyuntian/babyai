#!/bin/bash

level_list=( "GoToLocal" )
#level_list=( "BossLevel" )
#level_list=( "BossLevel" "PickupLoc" "GoToObjMaze" "GoTo" "Pickup" "UnblockPickup" "Open" "Unlock" "PutNext" "Synth" "SynthLoc" "GoToSeq" "SynthSeq" "GoToImpUnlock" )

#level_list=( "GoToObj" "GoToRedBall" "GoToRedBallGrey" "GoToLocal" "PutNextLocal" "PickUpLoc" "GoToObjMaze" "GoTo" "Pickup" "UnblockPickup" "Open" "Unlock" "PutNext" "Synth" "SynthLoc" "GoToSeq" "SynthSeq" "GoToImpUnlock" "BossLevel" )
#level_list=( "PickUpLoc" "GoToObjMaze" "GoTo" "Pickup" "UnblockPickup" "Open" "Unlock" "PutNext" "Synth" "SynthLoc" "GoToSeq" "SynthSeq" "GoToImpUnlock" "BossLevel" )
#level_list=( "GoTo" )

for level_name in "${level_list[@]}"
do
    for split_id in {0..9}
    do
        echo BabyAI-${level_name}-v0-split-${split_id}
        python ./scripts/make_agent_demos.py --env BabyAI-${level_name}-v0 \
            --demos ../data/babyai_pretraining/BabyAI-${level_name}-v0-split-${split_id} \
            --episodes 10000 \
            --valid-episodes 500 \
            --seed $((${split_id} * 10000000))
    done
done

