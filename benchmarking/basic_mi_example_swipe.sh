for j in {0..4}
do
    for lr in 0.01 0.001
    do 
        for opt in {4..0}
        do
            if [ "$2" == "CIFAR100" ]; then
                echo "Using CIFAR100! (specific file for now)"
                python basic_mi_example_100.py --model $1 --dataset $2 --optimizer_set $opt --seed $j --lr $lr 
                #--clean_start $3
            else 
                python basic_mi_example.py --model $1 --dataset $2 --optimizer_set $opt --seed $j --lr $lr  --clean_start $3
            fi
        done
    done
done