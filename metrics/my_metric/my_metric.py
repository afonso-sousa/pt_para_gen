import datasets
import evaluate
from sentence_transformers import SentenceTransformer, util
import torch

# TODO: Add BibTeX citation
_CITATION = """\
@InProceedings{huggingface:metric,
title = {A great new metric},
authors={huggingface, Inc.},
year={2020}
}
"""

# TODO: Add description of the metric here
_DESCRIPTION = """\
This new metric is designed to solve this great NLP task and is crafted with a lot of care.
"""


# TODO: Add description of the arguments of the metric here
_KWARGS_DESCRIPTION = """
Calculates how good are predictions given some references, using certain scores
Args:
    predictions: list of predictions to score. Each predictions
        should be a string with tokens separated by spaces.
    references: list of reference for each prediction. Each
        reference should be a string with tokens separated by spaces.
Returns:
    accuracy: description of the first score,
    another_score: description of the second score,
Examples:
    Examples should be written in doctest format, and should illustrate how
    to use the function.
    >>> my_new_metric = datasets.load_metric("my_new_metric")
    >>> results = my_new_metric.compute(references=[0, 1], predictions=[0, 1])
    >>> print(results)
    {'accuracy': 1.0}
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class MyMetric(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            # This is the description that will appear on the metrics page.
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features=datasets.Features(
                {
                    "sources": datasets.Value("string"),
                    "predictions": datasets.Value("string"),
                    "references": datasets.Value("string"),
                }
            ),
        )

    def _get_similarity(self, predictions, references):
        """Returns cosine similarity between source and paraphrase sentence vectors"""
        preds_encoded = self.sbert.encode(predictions, convert_to_tensor=True)
        refs_encoded = self.sbert.encode(references, convert_to_tensor=True)
        cos_sim = util.pairwise_cos_sim(preds_encoded, refs_encoded)
        return cos_sim.mean().item()

    def _download_and_prepare(self, dl_manager):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bleu = evaluate.load("bleu", experiment_id=self.experiment_id)
        self.sbert = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        ).to(device)

    def _compute(self, sources, predictions, references):
        refs = [[_ref] for _ref in references]
        bleu_score = self.bleu.compute(predictions=predictions, references=refs)["bleu"]

        srcs = [[_src] for _src in sources]
        self_bleu_score = self.bleu.compute(predictions=predictions, references=srcs)[
            "bleu"
        ]

        alpha = 0.7
        ibleu_formula = lambda bleu, self_bleu: round(
            alpha * bleu - (1 - alpha) * self_bleu, 3
        )
        ibleu_score = ibleu_formula(bleu_score, self_bleu_score)

        sbert_score = self._get_similarity(predictions, references)

        return {
            "bleu": bleu_score,
            "self_bleu": self_bleu_score,
            "ibleu": ibleu_score,
            "sbert": sbert_score,
        }
