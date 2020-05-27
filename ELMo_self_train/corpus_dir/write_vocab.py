import gensim

model = gensim.models.KeyedVectors.load_word2vec_format(
    fname='./vectors.txt'
)
words = model.vocab
with open('vocab.txt', 'w', encoding='utf8') as f:
    f.write('<S>')
    f.write('\n')
    f.write('</S>')
    f.write('\n')
    f.write('<UNK>')
    f.write('\n')    # bilm-tf 要求vocab有这三个符号，并且在最前面
    for word in words:
        f.write(word)
        f.write('\n')
