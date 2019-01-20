

def split_tag(tag):
    if tag == 'O':
        state = 'O'
        label = None

    elif tag == '-X-':
        state = 'O'
        label = None

    else:
        state, label = tag.split('-')

    return state, label


def iob2bio(tags):
    processed_tags = []  # should be bio format
    prev_state = None
    prev_label = None

    for t, tag in enumerate(tags):
        state, label = split_tag(tag)

        # case1. I-ORG I-ORG
        #        ^^^^^
        if t == 0 and state == 'I':
            new_state = 'B'

        # case2. I-ORG I-PERSON
        #              ^^^^^^^^
        elif state == 'I' and prev_label != label:
            new_state = 'B'

        # case3. O I-ORG
        #          ^^^^^
        elif state == 'I' and prev_state == 'O':
            new_state = 'B'

        # case4. I-ORG I-ORG
        #              ^^^^^
        elif state == 'I' and prev_label == label:
            new_state = 'I'

        else:
            new_state = state

        if label is None:
            new_tag = 'O'

        else:
            new_tag = f'{new_state}-{label}'

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
            next_state, next_label = 'O', None

        else:
            next_state, next_label = split_tag(tags[t+1])

        # case1. B-ORG O or B-ORG B-ORG
        #        ^^^^^      ^^^^^
        if state == 'B' and next_state in ['B', 'O']:
            new_state = 'S'

        # case2. I-ORG O or I-ORG B-PER
        #        ^^^^^      ^^^^^
        elif state == 'I' and next_state in ['B', 'O']:
            new_state = 'E'

        # case3. I-ORG I-ORG
        #        ^^^^^
        elif state == 'I' and next_state == 'I':
            new_state = 'I'

        else:
            new_state = state

        if label is None:
            new_tag = 'O'

        else:
            new_tag = f'{new_state}-{label}'

        processed_tags.append(new_tag)
    return processed_tags


def get_word_format_func(in_format, out_format):
    format_func_list = []
    if in_format == 'bio' and out_format == 'bioes':
        format_func_list.append(bio2bioes)

    if in_format == 'iob' and out_format == 'bio':
        format_func_list.append(iob2bio)

    if in_format == 'iob' and out_format == 'bioes':
        format_func_list.append(iob2bio)
        format_func_list.append(bio2bioes)

    return format_func_list


def apply_transform(elems, format_func_list):
    for func in format_func_list:
        elems = func(elems)
    return elems
