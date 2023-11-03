#!/bin/bash

for set in  1 2;	
do
	for o in  Adam SGD;				
	do
		for b in 6 8;	
		do
			echo '##############################################################################################################'
			echo '						NEW TRAINING'
			echo '##############################################################################################################'
			echo 'set ' $set ,  'Opt' $o , batch $b

			rm -Rf wandb/
			rm -Rf /home/isajim/.local/share/wandb/
			rm -Rf /tmp/wandb/
			wandb artifact cache cleanup 1GB
			TF_CPP_MIN_LOG_LEVEL=2   python3  base_model_main_ViT.py --datasetDIR /opt/data/isa/proxemics/dataset/   --outModelsDIR /pub/experiments/isajim/proxemics/models/   --b $b  --o $o --set  $set --lr 0.01 
			
			rm -Rf wandb/
			rm -Rf /home/isajim/.local/share/wandb/
			rm -Rf /tmp/wandb/
			wandb artifact cache cleanup 1GB
			TF_CPP_MIN_LOG_LEVEL=2   python3  base_model_main_ViT.py --datasetDIR /opt/data/isa/proxemics/dataset/   --outModelsDIR /pub/experiments/isajim/proxemics/models/   --b $b  --o $o --set  $set --lr 0.01 --onlyPairRGB
			
			rm -Rf wandb/
			rm -Rf /home/isajim/.local/share/wandb/
			rm -Rf /tmp/wandb/
			wandb artifact cache cleanup 1GB




		done
	done
done
