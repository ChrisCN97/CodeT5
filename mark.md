S3 = {
    "port": 1358,
    "root": "/mnt/sda/cn/codet5",
    "env": "conda activate pretrain_cuinan"
}

cd /mnt/sda/cn/codet5/sh
conda activate pretrain_cuinan

python run_exp.py --model_tag codet5_small --task summarize --sub_task python --model_dir model --res_dir res --summary_dir res --data_num 1000