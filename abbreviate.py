import pandas as pd
read_file = pd.read_csv('emojitweets.csv')

abbreviated_list = {
    "'til": "until",
    "'ll": " will",
    "y'all": "you all",
    "that's": "that is",
    "won't": "will not",
    "can't": "can not",
    "cannot": "can not",
    "ain't": "am not",
    "n't": " not", # wasn't weren't haven't didn't
    "'ve": " have", # i've we've you've would've
    "'d": " would", #you'd i'd we'd you'd
    "'re": " are", # they're we're you're
    "in'": "ing", # lookin'
}