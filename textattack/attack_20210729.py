import textattack
import transformers
import nltk

nltk.download('stopwords')

# Load model, tokenizer, and model_wrapper
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

input_text = "I really enjoyed the new movie that came out last month."
label = 1 #Positive
attack_result = attack.attack(input_text, label)
print(attack_result)
