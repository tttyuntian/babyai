#!/bin/bash

level_list=( "BossLevel" "PickupLoc" "GoToObjMaze" "GoTo" "Pickup" "UnblockPickup" "Open" "Unlock" "PutNext" "Synth" "SynthLoc" "GoToSeq" "SynthSeq" "GoToImpUnlock" )

#level_list=( "GoToObj" "GoToRedBall" "GoToRedBallGrey" "GoToLocal" "PutNextLocal" "PickUpLoc" "GoToObjMaze" "GoTo" "Pickup" "UnblockPickup" "Open" "Unlock" "PutNext" "Synth" "SynthLoc" "GoToSeq" "SynthSeq" "GoToImpUnlock" "BossLevel" )
#level_list=( "PickUpLoc" "GoToObjMaze" "GoTo" "Pickup" "UnblockPickup" "Open" "Unlock" "PutNext" "Synth" "SynthLoc" "GoToSeq" "SynthSeq" "GoToImpUnlock" "BossLevel" )
#level_list=( "GoTo" )

for level_name in "${level_list[@]}"
do
    echo BabyAI-${level_name}-v0
    python ./scripts/make_agent_demos.py --env BabyAI-${level_name}-v0 --demos ../data/babyai_pretraining/BabyAI-${level_name}-v0 --episodes 100000
done

