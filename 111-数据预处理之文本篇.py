from keras.preprocessing import text

# filters和split都可以起到分隔的作用
a = text.text_to_word_sequence("天 下大 势,为我所控 ABCD", filters="我 ", lower=True, split=',')
print(a)

a = text.one_hot("a a b b c c d", n=10)
print(a)

"""
text预处理中最重要的函数是tokenizer
"""
tok = text.Tokenizer(4)
tok.fit_on_texts(['a b a a ', 'b c a d e'])
print('训练的文档数', tok.document_count)
print("word index", tok.word_index)
print("index word", tok.index_word)
print("index docs", tok.index_docs)
print('texts to sequence', tok.texts_to_sequences(['a b a a']))
print("文本向量化")
print('binary', tok.texts_to_matrix(['a b a a'], mode='binary'))
print('tfidf', tok.texts_to_matrix(['a b a a'], mode='tfidf'))
print('count', tok.texts_to_matrix(['a b a a'], mode='count'))
print('freq', tok.texts_to_matrix(['a b a a'], mode='freq'))
