from corpus.tag_scheme import apply_transform
from corpus.tag_scheme import get_word_format_func
import re


BOS = 0  # begin of step
EOS = 1  # end  of step
XXX = 2  # other


def enum(word_sentences, tag_sentences):
    """
    enumerate words, chars and tags for
    constructing vocabularies

    :param word_sentences:
    """

    words = sorted(list(set(sum(word_sentences, []))))
    chars = sorted(list(set(sum([list(word) for word in words], []))))
    tags = sorted(list(set(sum(tag_sentences, []))))

    return words, chars, tags


def write_sentences(prefix, elem, prefix_elem_sentences, output_path):
    target = output_path / f'{prefix}.{elem}.txt'
    with open(target, 'w') as file:
        for _elem_sentence in prefix_elem_sentences:
            print(' '.join(_elem_sentence), file=file)


def write_vocab(prefix, elems, output_path):
    target = output_path / f'vocab.{prefix}.txt'
    with open(target, 'w') as file:
        print('\n'.join(elems), file=file)


class CorpusParser:
    def __init__(self, format_str=None):
        if format_str:
            in_format, out_format = format_str.split('2')
            self.format_func_list = get_word_format_func(in_format,
                                                         out_format)

        else:
            self.format_func_list = []

    def parse_file(self, file, word_idx=2, tag_idx=-1):
        annotated_file = open(file, encoding='utf-8')
        annotated_body = annotated_file.read()
        document = annotated_body.split('\n')
        return self._parse(document, word_idx, tag_idx)

    def _parse(self, document, word_idx, tag_idx):
        word_sentences = []
        tag_sentences = []

        status = BOS

        word_sentence = []
        tag_sentence = []

        for line in document:
            line = line.rstrip()
            pattern = re.compile(' +')
            elems = re.split(pattern, line)

            if line.startswith('-DOCSTART-'):
                continue

            if line == '':
                # EOR (end of recipe)
                status = EOS

            elif line.startswith('ID='):
                # BOR (begin of recipe)
                status = BOS

            elif len(elems) >= 4:
                status = XXX
                word, tag = elems[word_idx], elems[tag_idx]

                if tag.endswith('-B') or tag.endswith('-I'):
                    # suffix-style -> prefix-style (CoNLL is prefix-style)
                    tag = '-'.join(tag.split('-')[::-1])

            else:
                raise Exception

            if status == XXX:
                word_sentence.append(word)
                tag_sentence.append(tag)

            elif status == EOS:
                if not word_sentence:
                    continue

                word_sentences.append(word_sentence)

                arguments = (tag_sentence, self.format_func_list)
                tag_sentence = apply_transform(*arguments)
                tag_sentences.append(tag_sentence)

                # clean sentence for new input
                word_sentence = []
                tag_sentence = []

            elif status == BOS:
                assert not word_sentence
                assert not tag_sentence

        return word_sentences, tag_sentences
