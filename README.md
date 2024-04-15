# Remembering Transformer
<p align="center">
<img src="rt.png" width="80%"/>
</p>


## Training

### Split task
      python main_lora_pretrained_AE_lowdim_consolidation_retraining.py --task_type split --dataset cifar10 --num_task 5 --epochs 50 --seed 0
      
### Permutation task
      python main_lora_pretrained_AE_lowdim_consolidation_retraining.py --task_type permute --dataset mnist --num_task 20 --epochs 200 --seed 0
