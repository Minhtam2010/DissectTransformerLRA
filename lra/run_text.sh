function runexp {

gpu=${1}
task=${2}
model=${3}
layers=${4}
# lms=${5}      # lms (landmarks) is r in paper
# k_conv=${6}
# wsize=${7}    # wsize is w in paper
lr=${5}
wd=${6}
seed=${7}
flags=${8}

flags_print="$(echo -e "${flags}" | tr -d '[:space:]')"
flags_print=${flags_print//--/_}

expname=${task}-newdata-${model}-l_${layers}-lr_${lr}-wd_${wd}-seed_${seed}${flags_print}

cmd="
CUDA_VISIBLE_DEVICES=${gpu} python run_tasks.py --model ${model} --task ${task}
    ${flags}
    --learning_rate ${lr} --weight_decay ${wd} --num_layers ${layers}  --max_seq_len 4096
    --dropout_prob 0 --attention_dropout 0 --multi_gauss True --num_train_steps 12000
    --seed ${seed}
"

debug=1
if [ ${debug} -eq 0 ]; then
cmd="${cmd} --logging --expname ${expname}  > logs/${expname}.log 2>&1 &"
else
cmd="${cmd} "
fi

echo logs/${expname}.log

eval ${cmd}

}

# The following hyperparameters correspond to Transformer-LS (w,r = 8,32) in the paper.
# One can change it to Transformer-LS (best) with lms = 1, win_size = 1
#runexp  gpu   task  model    layers   lr   wd   seed   flags
runexp   0    text    softmax      2     1e-4  0.01 1234
# runexp   0    text    softmax      2     1e-4  0.01 8975
# runexp   0    text    softmax      2     1e-4  0.01 3517
# runexp   0    text    softmax      2     1e-4  0.01 986
