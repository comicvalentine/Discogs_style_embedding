import gzip
import xml.etree.ElementTree as ET
from itertools import combinations
from collections import Counter, defaultdict
import pickle

single_counter = Counter()
pair_counter = Counter()
max_genre_counter = defaultdict(Counter)

ver = "20260101"
search_type = "masters"

cnt = 0
with gzip.open(f"./raw_data/discogs_{ver}_{search_type}.xml.gz", "rb") as f:
    context = ET.iterparse(f, events=("end",))
    for event, elem in context:

        if elem.tag != "master":
            continue
        
        styles = [s.text.replace(" ", "_") for s in elem.findall("./styles/style")]
        genres = [g.text.replace(" ", "_") for g in elem.findall("./genres/genre")]

        if len(styles) >= 2:
            for a, b in combinations(sorted(styles), 2):
                pair_counter[(a, b)] += 1
        
        for st in styles:
            single_counter[st] += 1
            
            for gr in genres:
                max_genre_counter[st][gr] += 1
    
        elem.clear()

        cnt += 1
        if cnt % 1_000_000==0:
            with open(f"./embedding_data/pair_counter_{search_type}.pkl", 'wb') as f:
                pickle.dump(pair_counter, f)
            with open(f"./embedding_data/single_counter_{search_type}.pkl", 'wb') as f:
                pickle.dump(single_counter, f)
            with open(f"./embedding_data/max_genre_counter_{search_type}.pkl", 'wb') as f:
                pickle.dump(max_genre_counter, f)
            
            print('cleared:', cnt)

        
    with open(f"./embedding_data/pair_counter_{search_type}.pkl", 'wb') as f:
        pickle.dump(pair_counter, f)
    with open(f"./embedding_data/single_counter_{search_type}.pkl", 'wb') as f:
        pickle.dump(single_counter, f)
    with open(f"./embedding_data/max_genre_counter_{search_type}.pkl", 'wb') as f:
        pickle.dump(max_genre_counter, f)
