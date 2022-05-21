import pandas as pd
read_file = pd.read_csv('emojitweets.csv')

abbreviated_list = {
    "'til": "until",
    "i'll": "i will",
    "y'all": "you all",
    "that's": "that is",
}