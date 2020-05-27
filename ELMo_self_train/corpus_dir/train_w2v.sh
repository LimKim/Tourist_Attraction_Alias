current_dir=`pwd -P`
w2v_dir="$current_dir/../word2vec-master"
python turn2cut_for_w2v.py
echo 'file process complete for word2vec'
time $w2v_dir/word2vec -train $current_dir/laiye.corpus -output $current_dir/vectors.txt -cbow 1 -size 300 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 0 -iter 15
echo 'training word2vec complete!'
python write_vocab.py
echo 'write to vocab.txt'
