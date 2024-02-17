
CACHE_DIR="/gallery_tate/jaehyuk.sung/sed/weights"
# ANNOTATION="/gallery_tate/jaehyuk.sung/sed/datasets/audioset201906/Fast-Audioset-Download/audioset_balanced_train_metadata.json"
ANNOTATION="/gallery_tate/jaehyuk.sung/sed/datasets/audioset201906/Fast-Audioset-Download/audioset_unbalanced_train_metadata_convert.json"

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# this script is for 1024 total batch_size (n(8) GPUs * batch_size(4) * accum_freq(8))
cd /gallery_tate/jaehyuk.sung/sed/
srun8 --qos=ilow torchrun --nproc_per_node 8 \
    -m main  \
    --train-data ${ANNOTATION} \
    --train-num-samples 4800000 \
    --clip-type "al" --num_mel_bins 126 --target_length 1036 --audio_sample_rate 16000 --audio_mean -4.2677393 --audio_std 4.5689974 \
    --lock-text --lock-image --text-type "polish_mplug" \
    --init-temp 0.07 --learn-temp \
    --model "FT" --cache-dir ${CACHE_DIR} \
    --convert_to_lora --lora_r 16 \
    --lr 1e-4 --coef-lr 1 \
    --beta1 0.9 --beta2 0.98 --wd 0.2 --eps 1e-6 \
    --num-frames 1 --force-patch-dropout 0.1 \
    --epochs 16 --batch-size 4 --accum-freq 8 --warmup 2000 \
    --precision "amp" --workers 10 --video-decode-backend "imgs" \
    --save-frequency 1 --log-every-n-steps 20 --report-to "tensorboard" --resume "latest" \
    --do_eval --do_train \
    --val_a_cls_data "Audioset" 
