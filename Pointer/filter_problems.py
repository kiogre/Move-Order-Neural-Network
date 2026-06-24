import pandas as pd

df = pd.read_csv("lichess_db_puzzle.csv")
df = df[df['Themes'].str.contains('advantage|crushing|endgame')]
df = df[~df['Themes'].str.contains('mateIn1')]
df = df[df['Rating']<1200]
df.to_csv("advantages_training_dataset_under_1200.csv")