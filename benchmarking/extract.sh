# COUNT=0
# for f in data/train/*;
# do
#     python extract_imagenet_data.py --chunk $f &
#     echo "$COUNT/1002"
#     COUNT=$((COUNT + 1))
# done

open_sem(){
    mkfifo pipe-$$
    exec 3<>pipe-$$
    rm pipe-$$
    local i=$1
    for((;i>0;i--)); do
        printf %s 000 >&3
    done
}
run_with_lock(){
    local x
    read -u 3 -n 3 x && ((0==x)) || exit $x
    (
     ( "$@"; )
    printf '%.3d' $? >&3
    )&
}

N=30
open_sem $N
for f in data/train/*; do
    run_with_lock python extract_imagenet_data.py --chunk $f
    COUNT=$((COUNT + 1))
    echo "$COUNT"
done 