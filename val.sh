for IDX in $(seq 5 1 5); do
    srun8 --qos=ilow python infer.py --checkpoint_path "final_models/final.pt" --file "samples/sample"$IDX"/sample"$IDX".m4a" --k 1 --test
    srun8 --qos=ilow python infer.py --file "samples/sample"$IDX"/sample"$IDX".m4a" --k 1 --test
done
