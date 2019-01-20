all: get-wikidump tokenize

get-jawiki:
	cd data && wget https://dumps.wikimedia.org/jawiki/20180801/jawiki-20180801-pages-articles.xml.bz2
	cd data && WikiExtractor.py jawiki-20180801-pages-articles.xml.bz2 --output jawiki.json --json

get-enwiki:
	cd data && wget https://dumps.wikimedia.org/enwiki/20181001/enwiki-20181001-pages-articles.xml.bz2
	cd data && WikiExtractor.py enwiki-20181001-pages-articles.xml.bz2 --output enwiki.json --json

get-glove:
	cd data && wget http://nlp.stanford.edu/data/glove.6B.zip
	cd data && unzip glove.6B.zip
	cd data && rm glove.6B.zip

process-jawiki:
	python ./vector/fetch_jawikipedia.py

process-enwiki:
	python ./vector/fetch_enwikipedia.py

test:
	python -m unittest discover tests
