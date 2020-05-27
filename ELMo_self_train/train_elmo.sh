# export CUDA_VISIBLE_DEVICE=4,5
# nohup python -u bin/train_elmo.py --train_prefix='./corpus_dir/laiye.corpus' --vocab_file ./corpus_dir/vocab.txt --save_dir ./save_dir > train_out.txt 2>&1 &
nohup python -u bin/dump_weights.py --save_dir ./save_dir --outfile ./save_dir/weights.hdf5 > outfile.txt 2>&1 &
