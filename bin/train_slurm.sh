#set -x

function get_free_nodes {
    all_nodes=$(sinfo -N | grep idle | grep -v c1907 | grep ${1})

    i=0
    free_nodes=()
    for node in ${all_nodes};
    do
        r=$(( i % 4 ))
        i=$(( i + 1 ))
        if [[ $r = 0 ]]
        then
            free_nodes+=("$node")
        fi
    done
    n_free_nodes=${#free_nodes[@]}

    num_request=${2:-${n_free_nodes}}

    if [[ ${num_request} -gt ${n_free_nodes} ]]
    then
        echo 'Number Of Free Nodes' ${n_free_nodes} '. Number Of Requested Nodes' ${num_request} '. Not Enough Free Nodes To Allocate'
        exit 0
    fi

    NODELIST=(${free_nodes[@]:0:${num_request}})
    declare -p NODELIST
    #echo ${nodelist[@]}
}


#args=($@)
#PARTITION=${args[0]}
#NODELIST=(${args[@]:1})

PARTITION=${1}
N_REQUESTED_NODES=${2}
get_free_nodes ${PARTITION} ${N_REQUESTED_NODES}


if [[ ${PARTITION} = 'qgpu72' ]]
then
    NTASKS=4
    NTASKS_PER_NODE=4
    GRES='gpu:4'
    GPUS_PER_NODE=4
else
    NTASKS=1
    NTASKS_PER_NODE=1
    GRES='gpu:1'
    GPUS_PER_NODE=1
fi

CPUS_PER_TASK=4
JOB_NAME=khoaluu
TIME=3-00:00:00


export NUM_GPUS_PER_NODE=${NTASKS}
export NUM_NODES=${#NODELIST[@]}

export OMP_NUM_THREADS=1


export MASTER_ADDR=${NODELIST[0]}
export MASTER_PORT=2411

export WORLD_SIZE=$(( ${NUM_GPUS_PER_NODE} * ${NUM_NODES}))
export CUDA_LAUNCH_BLOCKING=1

COMMAND=${3}

OFFSET=0
for node in ${NODELIST[@]};
do
    OFFSET=${OFFSET} \
    WORLD_SIZE=${WORLD_SIZE} \
    MASTER_PORT=${MASTER_PORT} \
    MASTER_ADDR=${MASTER_ADDR} \
    RUN_MODE='slurm_sbatch' \
    sbatch  --nodelist="${node}" \
            --output=log-${JOB_NAME}-${PARTITION}-${OFFSET}-${node}.out \
            --ntasks=${NTASKS} \
            --ntasks-per-node=${NTASKS_PER_NODE} \
            --cpus-per-task=${CPUS_PER_TASK} \
            --gres=${GRES} \
            --job-name=${JOB_NAME} \
            --partition=${PARTITION} \
            --time=${TIME} \
            ${COMMAND} ${OFFSET_RANK}
    OFFSET=$((${OFFSET} + ${GPUS_PER_NODE}))
done
