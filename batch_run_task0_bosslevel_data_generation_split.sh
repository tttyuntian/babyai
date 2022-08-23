#!/bin/bash

level_list=( "BossLevel" )
#level_list=( "BossLevel" "PickupLoc" "GoToObjMaze" "GoTo" "Pickup" "UnblockPickup" "Open" "Unlock" "PutNext" "Synth" "SynthLoc" "GoToSeq" "SynthSeq" "GoToImpUnlock" )

#level_list=( "GoToObj" "GoToRedBall" "GoToRedBallGrey" "GoToLocal" "PutNextLocal" "PickUpLoc" "GoToObjMaze" "GoTo" "Pickup" "UnblockPickup" "Open" "Unlock" "PutNext" "Synth" "SynthLoc" "GoToSeq" "SynthSeq" "GoToImpUnlock" "BossLevel" )
#level_list=( "PickUpLoc" "GoToObjMaze" "GoTo" "Pickup" "UnblockPickup" "Open" "Unlock" "PutNext" "Synth" "SynthLoc" "GoToSeq" "SynthSeq" "GoToImpUnlock" "BossLevel" )
#level_list=( "GoTo" )

for level_name in "${level_list[@]}"
do
    # Generating negative examples
    for split_id in {1..10}
    #for split_id in 1
    do
        echo "negative_BabyAI-${level_name}-v0-split-$(( split_id - 1 ))"
        python ./scripts/make_agent_demos_negative_sampling.py --env BabyAI-${level_name}-v0 \
            --unsolvable_prob 0.7 \
            --demos ../data/task_0/BossLevel/negative_BabyAI-${level_name}-v0-split-$(( split_id - 1 )) \
            --episodes 2500 \
            --valid-episodes 500 \
            --seed $((${split_id} * 10000000))
         # seed starting at 10M
    done
    
    # Generating positive examples
    #for split_id in {1..10}
    #do
    #    echo BabyAI-${level_name}-v0-split-${split_id}
    #    python ./scripts/make_agent_demos.py --env BabyAI-${level_name}-v0 \
    #        --demos ../data/task_0/${level_name}/positive_BabyAI-${level_name}-v0-split-${split_id} \
    #        --episodes 2500 \
    #        --valid-episodes 500 \
    #        --seed $((${split_id} * 100000000))
    #    # seed starting at 100M
    #done
done

