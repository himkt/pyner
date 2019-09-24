import re

BOS = 0  # begin of step
EOS = 1  # end  of step
XXX = 2  # other


def split_tag(tag: str):
    """
    Split tag into state and named entity category.

    Parameters
    ---
    tag (str)
        NE tag (e.g. B-PER)
    """
    if tag in ["O", "-X-"]:
        state, label = "O", None
    else:
        state, label = tag.split("-")
    return state, label


def iob2bio(tags):
    processed_tags = []  # should be bio format
    prev_state = None
    prev_label = None

    for t, tag in enumerate(tags):
        state, label = split_tag(tag)

        # case1. I-ORG I-ORG
        #        ^^^^^
        if t == 0 and state == "I":
            new_state = "B"

        # case2. I-ORG I-PERSON
        #              ^^^^^^^^
        elif state == "I" and prev_label != label:
            new_state = "B"

        # case3. O I-ORG
        #          ^^^^^
        elif state == "I" and prev_state == "O":
            new_state = "B"

        # case4. I-ORG I-ORG
        #              ^^^^^
        elif state == "I" and prev_label == label:
            new_state = "I"

        else:
            new_state = state

        if label is None:
            new_tag = "O"

        else:
            new_tag = f"{new_state}-{label}"

        processed_tags.append(new_tag)
        prev_state = state
        prev_label = label

    return processed_tags


def bio2bioes(tags):
    # BIO -> BIOES: it only needs to check next
    processed_tags = []  # should be bio format
    last_index = len(tags) - 1

    for t, tag in enumerate(tags):
        state, label = split_tag(tag)

        if t == last_index:
            next_state, next_label = "O", None

        else:
            next_state, next_label = split_tag(tags[t + 1])

        # case1. B-ORG O or B-ORG B-ORG
        #        ^^^^^      ^^^^^
        if state == "B" and next_state in ["B", "O"]:
            new_state = "S"

        # case2. I-ORG O or I-ORG B-PER
        #        ^^^^^      ^^^^^
        elif state == "I" and next_state in ["B", "O"]:
            new_state = "E"

        # case3. I-ORG I-ORG
        #        ^^^^^
        elif state == "I" and next_state == "I":
            new_state = "I"

        else:
            new_state = state

        if label is None:
            new_tag = "O"

        else:
            new_tag = f"{new_state}-{label}"

        processed_tags.append(new_tag)
    return processed_tags


def get_word_format_func(in_format, out_format):
    format_func_list = []
    if in_format == "bio" and out_format == "bioes":
        format_func_list.append(bio2bioes)

    if in_format == "iob" and out_format == "bio":
        format_func_list.append(iob2bio)

    if in_format == "iob" and out_format == "bioes":
        format_func_list.append(iob2bio)
        format_func_list.append(bio2bioes)

    return format_func_list


def apply_transform(elems, format_func_list):
    for func in format_func_list:
        elems = func(elems)
    return elems


def enum(word_sentences, tag_sentences):
    """
    enumerate words, chars and tags for
    constructing vocabularies.
    """

    words = sorted(list(set(sum(word_sentences, []))))
    chars = sorted(list(set(sum([list(word) for word in words], []))))
    tags = sorted(list(set(sum(tag_sentences, []))))

    return words, chars, tags


def write_sentences(mode, sentences, output_path):
    target = output_path / f"{mode}.txt"
    with open(target, "w") as file:
        for sentence in sentences:
            for token in zip(*sentence):
                print("\t".join(token), file=file)
            print("", file=file)


def write_vocab(prefix, elems, output_path):
    target = output_path / f"vocab.{prefix}.txt"
    with open(target, "w") as file:
        print("\n".join(elems), file=file)


class CorpusParser:
    def __init__(self, format_str=None):
        if format_str:
            in_format, out_format = format_str.split("2")
            self.format_func_list = get_word_format_func(in_format, out_format)

        else:
            self.format_func_list = []

    def parse_file(self, file, word_idx=2, tag_idx=-1):
        annotated_file = open(file, encoding="utf-8")
        annotated_body = annotated_file.read()
        document = annotated_body.split("\n")
        return self._parse(document, word_idx, tag_idx)

    def _parse(self, document, word_idx, tag_idx):
        word_sentences = []
        tag_sentences = []

        status = BOS

        word_sentence = []
        tag_sentence = []

        for line in document:
            line = line.rstrip()
            pattern = re.compile(" +")
            elems = re.split(pattern, line)

            if line.startswith("-DOCSTART-"):
                continue

            if line == "":
                # EOR (end of recipe)
                status = EOS

            elif line.startswith("ID="):
                # BOR (begin of recipe)
                status = BOS

            elif len(elems) >= 4:
                status, word, tag = XXX, elems[word_idx], elems[tag_idx]

                if tag.endswith("-B") or tag.endswith("-I"):
                    # suffix-style -> prefix-style (CoNLL is prefix-style)
                    tag = "-".join(tag.split("-")[::-1])

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
