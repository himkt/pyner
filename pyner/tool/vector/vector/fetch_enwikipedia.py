import pathlib
import json
import tqdm
import sys


if __name__ == '__main__':
    from spacy.lang.en import English
    model = English()
    model.add_pipe(model.create_pipe('sentencizer'))

    wakati_fpath = 'data/enwiki-20181001-pages-articles.txt'
    wikiextractor_dir = pathlib.Path('data/enwiki')

    wakati_file = open(wakati_fpath, 'w')
    json_fpath_list = list(wikiextractor_dir.glob('*/wiki_*'))
    for json_fpath in tqdm.tqdm(json_fpath_list):
        for json_str in open(json_fpath):

            try:
                json_object = json.loads(json_str)
                document = json_object['text']
                document = document.replace('\n', '')
                document = document.replace('\r', '')

                document = model(document)
                for sentence in document.sents:
                    try:
                        print(' '.join(w.text for w in sentence),
                              file=wakati_file)

                    except Exception as e2:
                        print(e2, file=sys.stderr)

            except Exception as e:
                print(e, file=sys.stderr)

    wakati_file.close()
