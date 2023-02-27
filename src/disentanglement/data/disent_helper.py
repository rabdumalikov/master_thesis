import pandas as pd
import numpy as np
from nqa_text_util import simplify_nq_example


# Correct version
def simplify_nq_dataset( json ):
    questions = []
    answers = []
    contexts = []

    # extracting factual answers
    for i, j in json.iterrows():
        j = simplify_nq_example(j) # train set already simplified
        question = j['question_text']
        annots = j['annotations']
        context = ''
        answer = ''

        sub_contexts = []
        sub_answers = []

        for a in annots:
            long_s = a['long_answer']['start_token']
            long_e = a['long_answer']['end_token']
            context = ' '.join( j['document_text'].split()[long_s:long_e] )

            if a['short_answers']:
                for rng in a['short_answers']:        
                    short_s = rng['start_token']
                    short_e = rng['end_token']
                    answer = ' '.join( j['document_text'].split()[short_s:short_e] )
                    break

            if answer and context:
                sub_contexts.append(context)
                sub_answers.append(answer)

        if answer:
            idx = np.argmin([ len(ctx) for ctx in sub_contexts ])
            
            questions.append(question)
            contexts.append(sub_contexts[idx])
            answers.append(sub_answers[idx])

    df = pd.DataFrame( list(zip(questions, contexts, answers)), columns=['question', 'context', 'answer'] )
    return df

# This version is incorrect. It return less entries than the original dataset.
def simplify_nq_dataset2(json, simplified=True):
    questions = []
    answers = []
    contexts = []

    # extracting factual answers
    for i, j in json.iterrows():
        
        if not simplified:
            j = simplify_nq_example(j)

        question = j['question_text']
        
        annots = j['annotations']
        context = ''
        answer = ''

        candidates = j['long_answer_candidates']
        se_tokens = []

        for c in candidates:
            if c['top_level']:
                se_tokens.append( (c['start_token'], c['end_token']) )

        for s, e in se_tokens:
            for a in annots:

                if a['long_answer']['start_token'] == s and a['long_answer']['end_token'] == e:
                    context = ' '.join( j['document_text'].split()[s:e] )

                    if a['short_answers']:
                        rng = a['short_answers'][0]        
                        short_s = rng['start_token']
                        short_e = rng['end_token']
                        answer = ' '.join( j['document_text'].split()[short_s:short_e] )
                        break

            if answer and context:
                break
        
        if answer and context:        
            questions.append(question)
            answers.append(answer)
            contexts.append(context)

    df = pd.DataFrame( list(zip(questions, contexts, answers)), columns=['question', 'context', 'answer'] )
    return df        