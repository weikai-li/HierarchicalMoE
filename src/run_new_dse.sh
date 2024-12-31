array=("fdtd-2d-large" "gemver-medium" "syr2k")
mkdir /tmp/home_username
module load redis
port=13580

for kernel in "${array[@]}"; do
    redis-server --daemonize yes --port ${port}
    echo "${kernel} ${port}"
    python new_dse.py --dse_kernel ${kernel} --redis_port ${port} --moe_layers hierarchy-weighted-hidden --model_path --class_model_path --merlin_path &
    port=$(($port + 1))
done
wait
