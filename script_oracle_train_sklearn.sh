oracle="KNN"
cnt=0
for lr in 0.001 0.0001 0.001 0.05
do 
    for dropout in 0. 0.2 0.5
    do 
        for nhead in 1 2
        do 
            for nlayers in 1 2
            do
                for bsz in 256 64 128 256
                do 
                    #((cnt=cnt+1))
                    output_file="oracle_train_KNN.log"
                    python -u run_oracle_train.py --train-file './data/amp_spaced_train.csv' --test-file './data/amp_spaced_test.csv' --oracle-type ${oracle} --epochs 3 --batch-size ${bsz} --lr ${lr} --dropout ${dropout} --nhead ${nhead} --num-layers ${nlayers} >> ${output_file}
                done 
            done 
        done 
    done 
done 
