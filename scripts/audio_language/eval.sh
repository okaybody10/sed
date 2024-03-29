
CACHE_DIR="/gallery_tate/jaehyuk.sung/sed/weights"
ANNOTATION="/gallery_tate/jaehyuk.sung/sed/datasets/audioset201906/Fast-Audioset-Download/audioset_eval_train_metadata_convert.json"
# CACHE_DIR="path/to/pretrained/weight"
RESUME="audio_language.pt"
# ANNOTATION="path/to/data"
# this script is for 512 total batch_size (n(16) GPUs * batch_size(32) * accum_freq(1))
cd /gallery_tate/jaehyuk.sung/sed/
srun8 --qos=ilow torchrun --nproc_per_node 8 \
    -m main  \
    --train-data ${ANNOTATION} \
    --train-num-samples 4800000 \
    --clip-type "al" --num_mel_bins 126 --target_length 1036 --audio_sample_rate 16000 --audio_mean -4.2677393 --audio_std 4.5689974 \
    --lock-text --lock-image --text-type "polish_mplug" \
    --init-temp 0.07 --learn-temp \
    --model "ViT-L-14" --cache-dir ${CACHE_DIR} \
    --lr 1e-3 --coef-lr 1 \
    --beta1 0.9 --beta2 0.98 --wd 0.2 --eps 1e-6 \
    --num-frames 1 --force-patch-dropout 0.1 \
    --epochs 16 --batch-size 16 --accum-freq 4 --warmup 2000 \
    --precision "amp" --workers 10 --video-decode-backend "imgs" \
    --save-frequency 1 --log-every-n-steps 20 --report-to "tensorboard" --resume "latest" \
    --do_eval \
    --val_a_cls_data "Audioset"

