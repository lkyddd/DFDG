# DFDG

## Setup

This implemetation is based on Python3. To run the code, you need the following dependencies:

- torch==1.12.1

- scipy==1.10.1

- numpy==1.23.5

- sklearn==1.2.2

- pandas==2.0.0

- mpi4py==3.1.1

You can simply run 

```python
pip install -r requirements.txt
```

## Repository structure
We select some important files for detailed description.

```python
|-- LightFed # experiments for baselines, DFDG and datasets
    |-- experiments/ #
        |-- datasets/ 
            |-- data_distributer.py/  # the load datasets,including FMNSIT, SVHN, CIFAR-10, CINIC-10, CIFAR-100, Tiny-ImageNet, FOOD101
        |-- horizontal/ ## DFDG and baselines
            |-- DENSE/
            |-- DFAD/
	    |-- DFDG/
            |-- fedavg/
            |-- fedFTG/
        |-- models
            |-- model.py/  ##load backnone architectures
    |-- lightfed/  
        |-- core # important configure
        |-- tools
```

## Run pipeline for Run pipeline for DFDG
1. Entering the DFDG
```python
cd LightFed
cd experiments
cd horizontal
cd DFDG
```

2. You can run any models implemented in `main_DFDG.py`. For examples, you can run our model on `SVHN` dataset by the script:
```python
python main_DFDG.py --batch_size 64 --I_c 100 --comm_round 1 --eta_c 0.01 --mask False --weight_agg_plus True --model_heterogeneity False /
                    --model_here_level 2 --model_split_mode roll --I 800 --I_g 20 --I_d 2 --eta_g 0.0002 --b1 0.5 --b2 0.999 --eta_d 0.01 /
                    --beta_tran 1.0 --beta_div 1.0 --beta_cd 1.0 --latent_dim 100 --noise_label_combine mul --tran_strategy 1 --condition_ False /
                    --save_generator True --data_partition_mode non_iid_unbalanced --non_iid_alpha 1.0 --client_num 10 --selected_client_num 10 /
                    --seed 0 --model_type ResNet_18 --data_set SVHN --eval_batch_size 256 --device cuda 
```
And you can run other baselines, such as 
```python
cd LightFed
cd experiments
cd horizontal
cd fedavg
python main_fedavg.py --batch_size 64 --I_c 100 --comm_round 1 --eta_c 0.01 --mask False --data_partition_mode non_iid_unbalanced /
                      --non_iid_alpha 0.1 --client_num 10 --selected_client_num 10 --seed 0 --model_type ResNet_18 --data_set SVHN /
                      --eval_batch_size 256 --device cuda
```

