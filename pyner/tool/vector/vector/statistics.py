import tqdm


word2index = {}
n_sentence = 0
n_token = 0

# for line in tqdm.tqdm(open('./data/steps2.txt')):
for line in tqdm.tqdm(open('./data/jawiki-20180801-pages-articles.txt')):
    n_sentence += 1

    for word in line.split(' '):
        n_token += 1
        word2index[word] = word2index.get(word, 0) + 1

print('n_sentence: ', n_sentence)
print('n_token: ', n_token)
print('n_vocab: ', len(word2index))
