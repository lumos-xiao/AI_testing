import datasets
import textattack
import transformers

# model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")
# tokenizer = transformers.AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")
# model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
#
# attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)

model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")
tokenizer = transformers.AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")
model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

# Construct our four components for `Attack`
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.constraints.semantics import WordEmbeddingDistance

goal_function = textattack.goal_functions.UntargetedClassification(model_wrapper)
constraints = [
    RepeatModification(),
    StopwordModification(),
    WordEmbeddingDistance(min_cos_sim=0.9)
]
transformation = textattack.transformations.WordSwapEmbedding(max_candidates=50)
search_method = textattack.search_methods.alzantot_genetic_algorithm.AlzantotGeneticAlgorithm(pop_size=60, max_iters=20, temp=0.3, give_up_if_no_improvement=False, post_crossover_check=True, max_crossover_retries=20)

# Construct the actual attack
attack = textattack.Attack(goal_function, constraints, transformation, search_method)

dataset = textattack.datasets.HuggingFaceDataset('imdb', split="test")
# self._dataset = datasets.load_dataset(self._name,ubset)[split]
# data = datasets.load_dataset('/home/xmx/projects/AI_testing/textattack/aclImdb_v1/aclImdb/imdb.py')['test']
# data = [("I enjoyed the movie a lot!", 1), ("Absolutely horrible film.", 0), ("Our family had a fun time!", 1)]
# dataset = textattack.datasets.Dataset(data)

# Attack 20 samples with CSV logging and checkpoint saved every 5 interval
attack_args = textattack.AttackArgs(
    num_examples=20,
    log_to_csv="log.csv",
    checkpoint_interval=5,
    checkpoint_dir="checkpoints",
    disable_stdout=True
)

attacker = textattack.Attacker(attack, dataset, attack_args)
attacker.attack_dataset()