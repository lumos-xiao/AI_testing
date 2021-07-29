import textattack

# Example of sentiment-classification dataset
data = [("I enjoyed the movie a lot!", 1), ("Absolutely horrible film.", 0), ("Our family had a fun time!", 1)]
dataset = textattack.datasets.Dataset(data)
print(dataset.__len__())
print(dataset.__getitem__(1))

# # Example for pair of sequence inputs (e.g. SNLI)
# data = [("A man inspects the uniform of a figure in some East Asian country.", "The man is sleeping"), 1)]
# dataset = textattack.datasets.Dataset(data, input_columns=("premise", "hypothesis"))
#
# # Example for seq2seq
# data = [("J'aime le film.", "I love the movie.")]
# dataset = textattack.datasets.Dataset(data)