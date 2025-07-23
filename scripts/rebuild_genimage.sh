#!/usr/bin/env bash
# rebuild_genimage.sh
# Rebuild the dataset with symbolic links:
#   root/<class>/<dummy>/<split>/ai  ->  target/<class>/<split>/ai
# The entire 'ai' directory is linked, not individual files.

# original folder
# root
# |-- GenImage
# |   |-- ADM
# |   |   |-- imagenet_ai_0508_adm (we just removed this layer, nothing new)
# |   |   |   |-- train
# |   |   |   |   |--ai
# |   |   |   |   |--nature (move the real images to an independent folder)
# |   |   |   |-- val
# |   |   |   |   |--ai
# |   |   |   |   |--nature
# |   |-- BigGAN
# |   |   |-- imagenet_ai_0419_biggan
# |   |   |   |-- train
# |   |   |   |   |--ai
# |   |   |   |   |--nature
# |   |   |   |-- val
# |   |   |   |   |--ai
# |   |   |   |   |--nature
# |   |-- glide
# |   |   |......
# |   |-- Midjourney
# |   |-- SD
# |   |-- VQDM

ROOT_DIR="your_gen_image_dataset_path"
TARGET_DIR="rebuild_dataset_path"

mkdir -p "${TARGET_DIR}"

for CLASS_DIR in "${ROOT_DIR}"/*; do
    [[ -d "${CLASS_DIR}" ]] || continue
    CLASS_NAME=$(basename "${CLASS_DIR}")

    mkdir -p "${TARGET_DIR}/${CLASS_NAME}"/{train,val}

    for DUMMY_DIR in "${CLASS_DIR}"/*; do
        [[ -d "${DUMMY_DIR}" ]] || continue

        for SPLIT in train val; do
            SRC_AI="${DUMMY_DIR}/${SPLIT}/ai"
            DST_AI="${TARGET_DIR}/${CLASS_NAME}/${SPLIT}/ai"

            [[ -d "${SRC_AI}" ]] || continue

            ln -s "${SRC_AI}" "${DST_AI}"
        done
    done
done

echo "Dataset rebuilt at ${TARGET_DIR}"

# remember to merge the real images yourself