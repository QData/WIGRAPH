NUM_CLASSES = {
                    'sst2' : 2,
                    'sst1' : 5,
                    'imdb' : 2,
                    'agnews' : 4,
                    'yelp' : 2,
                    'trec' : 6,
                    'subj' : 2
                    }

SPECIAL_TOKENS = {
    'bert' : [101, 102, 100, 0],
    'roberta' :  [0, 1, 2],
    'distilbert': [101, 102, 0, 100]
		}
PAD = {
    'bert' : 0,
    'roberta' :  1,
    'distilbert': 0
                }               
START = {
    'bert' : 101,
    'roberta' :  0,
    'distilbert': 101
                }
END = {
    'bert' : 102,
    'roberta' :  2,
    'distilbert': 102
                }

