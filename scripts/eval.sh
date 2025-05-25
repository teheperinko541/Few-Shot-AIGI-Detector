
OUTPUT_PATH='./output_dir'
SEED=42

# test class
TEST_CLASS="ADM"

CKPT_PATH="your_ckpt_path/resnet50_200000.pth"

data_root=(
    "data/GenImage" \
)

# execution
python test.py \
    --data_root "$data_root" \
    --test_class $TEST_CLASS \
    --ckpt_path $CKPT_PATH \
    --num_workers 8 \
    --seed $SEED \
    --use_fp16 True \