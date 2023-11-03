# INHERIT\_cBERT: A new framework for identifying bacteriophages via conditional representation learning

![Pipeline_new](https://github.com/Celestial-Bai/INHERIT_cBERT/blob/master/pipeline.jpg)This repository includes the implementation of "INHERIT\_cBERT: A new framework for identifying bacteriophages via conditional representation learning". You can train conditional BERT and use INHERIT_cBERT to identify phages with this repository.

We are still developing this package and we will also try to make some improvements of it, so feel free to report to us if there are any issues occurred. We keep updating for exploring better performance and more convenient utilization.

## Installation

```
git clone https://github.com/Celestial-Bai/INHERIT_cBERT.git
cd INHERIT_cBERT
pip install -r dependencies.txt
```



## Environment and requirements

We use NVIDIA A100 GPUs to train INHERIT with CUDA 11.4.  We also tested our codes on other GPUs, like V100, and they can run smoothly.

Compared with [INHERIT](https://github.com/Celestial-Bai/INHERIT), our new tool is much easier to install and run :)

```
##### All #####
argparse
numpy
torch==1.13.1+cu117
tqdm
transformers==4.30.2
tokenizers==0.11.1
```




## Pre-training

There are 2 steps to pre-train a conditional BERT. Here is the sample script you can use:

Step 1:

```
#$ -l a100=4,s_vmem=500G
export KMER=6
export TRAIN_FILE=DATA_PATH
export SOURCE=SOURCE_PATH
export OUTPUT_PATH=OUTPUT_PATH

python3 run_pretrain_conditional_step1.py \
    --output_dir $OUTPUT_PATH \
    --model_type=dna \
    --tokenizer_name=bert-config-$KMER-conditional\
    --config_name=$SOURCE/bert-config-$KMER-conditional/config.json \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --mlm \
    --gradient_accumulation_steps 16 \
    --per_gpu_train_batch_size 32 \
    --save_steps 500 \
    --save_total_limit 20 \
    --max_steps 200000 \
    --logging_steps 500 \
    --line_by_line \
    --learning_rate 4e-4 \
    --block_size 512 \
    --adam_epsilon 1e-6 \
    --weight_decay 0.01 \
    --beta1 0.9 \
    --beta2 0.98 \
    --seed 123 \
    --mlm_probability 0.025 \
    --warmup_steps 10000 \
    --overwrite_output_dir \
    --n_process 32
```

Step 2:

```
#$ -l a100=4,s_vmem=500G
export KMER=6
export TRAIN_FILE=DATA_PATH
export SOURCE=SOURCE_PATH
export OUTPUT_PATH=OUTPUT_PATH

python3 run_pretrain_conditional_step2.py \
    --output_dir $OUTPUT_PATH \
    --model_type=dna \
    --tokenizer_name=bert-config-$KMER-conditional\
    --config_name=$SOURCE/bert-config-$KMER-conditional/config.json \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --mlm \
    --gradient_accumulation_steps 2 \
    --per_gpu_train_batch_size 32 \
    --save_steps 500 \
    --save_total_limit 20 \
    --max_steps 200000 \
    --logging_steps 500 \
    --line_by_line \
    --learning_rate 5e-5 \
    --block_size 512 \
    --adam_epsilon 1e-6 \
    --weight_decay 0.001 \
    --beta1 0.9 \
    --beta2 0.98 \
    --seed 123 \
    --mlm_probability 0.025 \
    --warmup_steps 5000 \
    --overwrite_output_dir \
    --n_process 32
```

Our pre-trained model can be downloaded from:

[Step 1](https://drive.google.com/drive/folders/1XVbWUr9edc6vetOG0y-YtwTDEP53CD2G?usp=sharing), and [Step 2](https://drive.google.com/drive/folders/1_kq4288kg00wi1vGzkfdd9hYDccmAThc?usp=sharing)

## Predict 

If you want to use INHERIT_cBERT to identify phages, please first download our fine-tuned model:

[RefSeq](https://drive.google.com/file/d/11Sm9Rz61Hu6h_BB9pElWCX2JS3DMKRpy/view?usp=sharing), and [short contigs](https://drive.google.com/file/d/1Rool6Fqu-zDN60TRF3GAE-PaxR5mHgRp/view?usp=drive_link)

You can simply used：

```
python3 IHT_predict.py --sequence test.fasta --withpretrain False --model FINETUNED_MODEL --out test_out.txt
```


## 

## Fine-tuning

If you want to fine-tune the conditional BERT, here we give the example on fine-tuning for phage identification:

```
python3 --bertdir PRETRAINED_MODEL --outdir MODEL_NAME
```


## 



