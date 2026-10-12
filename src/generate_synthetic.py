"""Crea un conjunto SINTETICO para pruebas de humo, NO para elegir un modelo."""
import argparse
import csv
from pathlib import Path

POSITIVE = [
    'The story kept me interested from the first scene to the end.',
    'The actors gave convincing and moving performances.',
    'A charming film with a thoughtful and satisfying ending.',
    'The soundtrack added real emotion to every important moment.',
    'A clever script that made the entire experience enjoyable.',
    'I would happily recommend this movie to my friends.',
    'The director turned a simple idea into something memorable.',
    'Every major character felt believable and well developed.',
    'The pace was excellent and the humor worked naturally.',
    'The final scene was beautiful and surprisingly powerful.',
    'A wonderful performance by an exceptionally talented cast.',
    'The cinematography made even ordinary scenes look stunning.',
    'The film delivered exactly the kind of joy I expected.',
    'An entertaining adventure filled with warmth and imagination.',
    'The ending tied the story together in a rewarding way.',
    'I loved how carefully the dialogue was written.',
    'This is an inspiring story that deserves attention.',
    'The movie balanced comedy and drama exceptionally well.',
    'The central relationship felt honest and touching.',
    'A delightful surprise that exceeded my expectations.',
    'The action was exciting without sacrificing the story.',
    'A refreshing and original take on a familiar genre.',
    'It was funny, warmhearted, and full of memorable scenes.',
    'The lead actor brought remarkable depth to the role.',
    'This beautifully made film stayed with me afterward.',
    'The writing was smart and the jokes landed perfectly.',
    'The emotional moments felt earned rather than forced.',
    'A polished production with an excellent sense of rhythm.',
    'This small independent film was genuinely impressive.',
    'The story moved forward with confidence and creativity.',
    'An absorbing movie that I could watch again.',
    'It offered a satisfying mixture of suspense and heart.',
    'The performances gave life to a fascinating screenplay.',
    'The visuals were striking and helped tell the story.',
    'A moving conclusion made the journey worthwhile.',
    'The characters had great chemistry throughout the film.',
    'I found the entire viewing experience highly rewarding.',
    'An imaginative film that makes great use of its setting.',
    'The movie was engaging and surprisingly well paced.',
    'A thoughtful story delivered with skill and sensitivity.',
]
NEGATIVE = [
    'The plot was confusing and never became interesting.',
    'The acting felt stiff and completely unconvincing.',
    'A dull film that wasted a potentially good idea.',
    'The loud soundtrack distracted me from the story.',
    'The dialogue sounded awkward and poorly written.',
    'I would not recommend this movie to anyone.',
    'The director failed to create any meaningful tension.',
    'Every main character seemed flat and forgettable.',
    'The pacing dragged and the jokes were painfully weak.',
    'The final scene was abrupt and deeply disappointing.',
    'An uninspired performance from an otherwise capable cast.',
    'The visuals looked cheap and strangely unfinished.',
    'The film did not deliver on its interesting premise.',
    'A tiresome adventure full of predictable events.',
    'The ending left far too many frustrating gaps.',
    'I disliked how repetitive the dialogue became.',
    'This is a tedious story that fails to connect.',
    'The movie mixed comedy and drama in an awkward way.',
    'The central relationship felt artificial and forced.',
    'A disappointing experience that fell below expectations.',
    'The action was chaotic and made no sense.',
    'A derivative and unoriginal attempt at a familiar genre.',
    'It was boring, unpleasant, and full of forgettable scenes.',
    'The lead actor struggled to make the character believable.',
    'This clumsily made film was instantly forgettable.',
    'The writing was careless and none of the jokes worked.',
    'The emotional moments felt manipulative and false.',
    'A messy production with no consistent sense of rhythm.',
    'This independent film was unfortunately underdeveloped.',
    'The narrative wandered without any clear direction.',
    'An exhausting movie I would never watch again.',
    'It offered neither believable suspense nor emotional depth.',
    'The performances could not save the weak screenplay.',
    'The unattractive visuals made the experience worse.',
    'A frustrating conclusion undermined the entire story.',
    'The characters had no chemistry throughout the film.',
    'I found the entire viewing experience irritating.',
    'An unimaginative film that wastes its unusual setting.',
    'The movie was predictable and painfully slow.',
    'A shallow story delivered without skill or sensitivity.',
]

def generate(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    for i,(sentences,label) in enumerate(((NEGATIVE,0),(POSITIVE,1))):
        for j,sentence in enumerate(sentences):
            rows.append({'idx':f'SYN-{label}-{j:03d}','sentence':sentence,'label':label,'source':'synthetic_manual','split':'smoke_test'})
    rows.sort(key=lambda r:r['idx'])
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['idx','sentence','label','source','split']);writer.writeheader();writer.writerows(rows)
    print(f'{len(rows)} filas sinteticas: {path}')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='data/synthetic_smoke_test.csv');generate(p.parse_args().out)
