import textattack
import transformers

model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")
tokenizer = transformers.AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")
model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)
dataset = textattack.datasets.HuggingFaceDataset("imdb", split="test")

# Attack 20 samples with CSV logging and checkpoint saved every 5 interval
attack_args = textattack.AttackArgs(
    num_examples=20,
    log_to_csv="log.csv",
    checkpoint_interval=5,
    checkpoint_dir="checkpoints",
    disable_stdout=True
)

attacker = textattack.Attacker(attack, dataset, attack_args)
# attacker.update_attack_args(parallel=True, checkpoint_interval=500)
'''
textattack.AttackArgs(num_examples: int = 10
, num_successful_examples: Optional[int] =
 None, num_examples_offset: int = 0, attack_n: bool = False, 
 shuffle: bool = False, query_budget: Optional[int] = None, 
 checkpoint_interval: Optional[int] = None, checkpoint_dir: str = 'checkpoints', 
 random_seed: int = 765, parallel: bool = False, num_workers_per_device: int = 1, 
 log_to_txt: Optional[str] = None, log_to_csv: Optional[str] = None, 
 csv_coloring_style: str = 'file', log_to_visdom: Optional[dict] = None, 
 log_to_wandb: Optional[str] = None, disable_stdout: bool = False, silent: bool = False)
'''
attacker.attack_dataset()