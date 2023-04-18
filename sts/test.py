# %%
from commons import read_xml
import numpy as np

test_pairs = read_xml("sts/assin2-test.xml", need_labels=False)

h_test = [pair.h for pair in test_pairs]
t_test = [pair.t for pair in test_pairs]
labels = [pair.similarity for pair in test_pairs]
labels = np.array(labels) / 5.0

# %%
# Load my trained Sentence Transformer
from sentence_transformers import SentenceTransformer

# model_name = "output/training_paraphrases_sentence-transformers-paraphrase-multilingual-mpnet-base-v2-2023-04-14_13-02-32/15000"
# model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# model_name = "output/training_paraphrases_distilroberta-base-2023-04-17_08-29-20/11250"
model_name = "output/training_paraphrases_distilroberta-base-2023-04-17_09-13-26/18750"

sbert = SentenceTransformer(model_name)

# %%
batch_size = 64
embeddings1 = sbert.encode(
    h_test, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
)
embeddings2 = sbert.encode(
    t_test, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
)

# %%
# labels = self.scores
from sklearn.metrics.pairwise import paired_cosine_distances

cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))

# %%
from scipy.stats import pearsonr

pearson = pearsonr(labels, cosine_scores)[0]
absolute_diff = labels - cosine_scores
mse = (absolute_diff**2).mean()

print("Similarity evaluation")
print("Pearson\t\tMean Squared Error")
print("-------\t\t------------------")
print("{:7.3f}\t\t{:18.2f}".format(pearson, mse))

# %%
