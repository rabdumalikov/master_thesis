import random
import nltk
import numpy as np

def is_undersensitivity_attack( score1, score2 ):
    return score2['score'] > score1['score'] and score2['answer'] == score1['answer']

# def ne_pertubation(question, ne_collection, ner):
#     doc = ner( question )
#     res = [(X.text, X.label_) for X in doc.ents]

#     if not res:
#         return '' 

#     idx = random.randint(0, len(res) - 1)

#     text, label = res[idx]

#     substitution = random.choice(ne_collection[label])

#     return question.replace( text, substitution )

def ne_pertubation( question, ne_collection, ner, is_attack_question, beam_size=1, n=1 ):

    doc = ner(question)

    res = [(X.text, X.label_) for X in doc.ents]

    if not res:
        return '' 

    if len(res) >= beam_size:
        beam_size = len(res)
        idxs = [ i for i in range(beam_size) ]
    else:
        idxs = [ random.randint(0, len(res) - 1) for _ in range(beam_size) ]

    labels = list(set([ res[idx][1] for idx in idxs ]))
    texts  = list(set([ res[idx][0] for idx in idxs ]))

    # FIX: Sometimes question has less NE than beam_size, I'm not sure if this is a problem
    beam_size = beam_size if beam_size < len(labels) else len(labels)

    subs = []
    for label in labels:
        substitution = random.choices(ne_collection[label], k=n)
        subs.append( substitution )

    def substitute( subs_idxs, subs ):
        new_question = question
        for i, idx in enumerate(subs_idxs):
            new_question = new_question.replace( texts[i], subs[i][idx] )

        return new_question

    subs = np.asarray(subs)
    indexes = [0] * beam_size

    for _ in range(n**beam_size):
        new_question = substitute( indexes, subs )

        is_attack = is_attack_question(new_question)

        if is_attack:
            return new_question

        for j in reversed(range(beam_size)):
            indexes[j] += 1
            if indexes[j] == n:
                indexes[j] = 0
            else:
                break

    return None


excluded_POS_tags = ['IN', 'DT', '.', 'VBD', 'VBZ', 'WP', 'WRB', 'WDT', 'CC', 'MD', 'TO', ')', '(', ',', '$', "''", ':', '#', '``']

def pos_pertubation( question, pos_collection, is_attack_question, beam_size=1, n=1 ):

    question_chunks = question.split()

    # while True:
    #     idx = random.randint(0, len(question_chunks) - 1)
    #     token = nltk.word_tokenize( question_chunks[idx] )
    #     tag = nltk.pos_tag(token)[0]
    #     if tag[1] not in excluded_POS_tags:
    #         break

    # substitute = random.choice(pos_collection[tag[1]])

    # question_chunks[idx] = ''.join(substitute)
    
    # return ' '.join(question_chunks)

    while True:
        idxs = [ random.randint(0, len(question_chunks) - 1) for i in range(beam_size) ]
        tokens = [ nltk.word_tokenize( question_chunks[idx] ) for idx in idxs ]    
        tags = [ nltk.pos_tag(token)[0] for token in tokens ]

        tags = [ tag[1] for tag in tags ]

        if len(tags) > 1 and all( np.asarray(tags) == tags[0] ):
            continue
        
        if all([ tag not in excluded_POS_tags for tag in tags ]):
            break

    subs = []
    for tag in tags:
        substitution = random.choices(pos_collection[tag], k=n)
        subs.append( substitution )

    def substitute( question_idxs, subs_idxs, subs, question_chunks ):
        new_question_chunks = question_chunks.copy()
        for i, idx in enumerate(subs_idxs):
            new_question_chunks[question_idxs[i]] = subs[i][idx]
        return ' '.join(new_question_chunks)

    subs = np.asarray(subs)
    indexes = [0] * beam_size

    for _ in range(n**beam_size):
        new_question = substitute( idxs, indexes, subs, question_chunks )

        is_attack = is_attack_question(new_question)

        if is_attack:
            return new_question

        for j in reversed(range(beam_size)):
            indexes[j] += 1
            if indexes[j] == n:
                indexes[j] = 0
            else:
                break

    return None

