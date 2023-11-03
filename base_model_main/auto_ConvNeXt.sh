#!/bin/bash

for set in 1 2;				
do
	for o in SGD Adam;					
	do
		for mo in base large;
		do 
		
			echo '##############################################################################################################'
			echo '						NEW TRAINING'
			echo '##############################################################################################################'
			echo 'set ' $set ,  'Opt' $o , batch 6, modeltype $mo

			rm -Rf wandb/
			rm -Rf /home/isajim/.local/share/wandb/
			rm -Rf /tmp/wandb/
			wandb artifact cache cleanup 1GB
			TF_CPP_MIN_LOG_LEVEL=2  TF_ENABLE_ONEDNN_OPTS=0 WANDB_CACHE_DIR="/tmp/" python3 base_model_main_convNext.py --datasetDIR /opt/data/isa/proxemics/dataset/    --outModelsDIR /opt/data/isa/proxemics/models/  --modeltype $mo --b 6  --o $o --set  $set --lr 0.01 
			
			rm -Rf wandb/
			rm -Rf /home/isajim/.local/share/wandb/
			rm -Rf /tmp/wandb/
			wandb artifact cache cleanup 1GB
			TF_CPP_MIN_LOG_LEVEL=2  TF_ENABLE_ONEDNN_OPTS=0 WANDB_CACHE_DIR="/tmp/" python3 base_model_main_convNext.py --datasetDIR /opt/data/isa/proxemics/dataset/   --outModelsDIR /opt/data/isa/proxemics/models/  --modeltype $mo --b 6  --o $o --set  $set --lr 0.01  --onlyPairRGB
			
			rm -Rf wandb/
			rm -Rf /home/isajim/.local/share/wandb/
			rm -Rf /tmp/wandb/
			wandb artifact cache cleanup 1GB

		done
	done
done

