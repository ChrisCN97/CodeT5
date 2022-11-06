S3 = {
    "port": 1358,
    "root": "/mnt/sda/cn/codet5",
    "env": "conda activate pretrain_cuinan"
}

microsoft/codebert-base
Salesforce/codet5-base

transformers 4.18.0

CUDA_VISIBLE_DEVICES=0

nohup ./run_mlm.sh > sh/output/pretrain_mlm.log 2>&1 &
2557709

nohup ./run_t5.sh > sh/output/pretrain_t5.log 2>&1 &
2570999

python run_mlm.py \
    --model_name_or_path microsoft/codebert-base \
    --train_file "/mnt/sda/cn/codet5/data/pretrain/test/train.txt" \
    --validation_file "/mnt/sda/cn/codet5/data/pretrain/test/val.txt" \
	--line_by_line \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir pretrain/test-mlm \
	--overwrite_output_dir

python run_t5_mlm_flax.py \
	--output_dir="pretrain/test-mlm" \
	--model_type="codet5" \
	--config_name="Salesforce/codet5-base" \
	--tokenizer_name="Salesforce/codet5-base" \
	--train_file "/mnt/sda/cn/codet5/data/pretrain/with_lang/v1/train.txt" \
    --validation_file "/mnt/sda/cn/codet5/data/pretrain/with_lang/v1/val.txt" \
	--max_seq_length="512" \
	--per_device_train_batch_size="32" \
	--per_device_eval_batch_size="32" \
	--adafactor \
	--learning_rate="0.005" \
	--weight_decay="0.001" \
	--overwrite_output_dir \
	--warmup_steps="2000" \
	--logging_steps="500" \
	--save_steps="10000" \
	--eval_steps="2500"

nohup python run_t5_mlm_flax.py \
	--output_dir="pretrain/test-mlm" \
	--model_type="codet5" \
	--config_name="Salesforce/codet5-base" \
	--tokenizer_name="Salesforce/codet5-base" \
	--train_file "/mnt/sda/cn/codet5/data/pretrain/test/train.txt" \
    --validation_file "/mnt/sda/cn/codet5/data/pretrain/test/val.txt" \
	--max_seq_length="512" \
	--per_device_train_batch_size="8" \
	--per_device_eval_batch_size="8" \
	--adafactor \
	--learning_rate="0.005" \
	--weight_decay="0.001" \
	--overwrite_output_dir \
	--warmup_steps="2000" \
	--logging_steps="500" \
	--save_steps="10000" \
	--eval_steps="2500" > sh/output/pretrain_test.log 2>&1 &
2570544

