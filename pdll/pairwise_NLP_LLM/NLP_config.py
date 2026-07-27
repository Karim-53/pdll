# list of openAI models
from pairwise_NLP_LLM.NLP_rubric_extraction import get_rubric_texts_from_files

llms = [
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "o4-mini",
    "gpt-4o",
    "gpt-4.1",
]

# Set control variables
# is_test_run = True

max_score_per_set = {
    1: 12,
    2: 10,
    3: 3,
    4: 3,
    5: 4,
    6: 4,
    7: 30,
    8: 60,
}

# Load scoring rubrics
scoring_rubrics = get_rubric_texts_from_files()

fold_ID = 1
random_seed = 81

llm = llms[5]
essay_set = 8
max_score = max_score_per_set[essay_set]
rubric = scoring_rubrics[essay_set]

is_pairwise = True
if is_pairwise:
    limit_data = 10
    limit_anchors = 7
    limit_reasonable = 40
else:
    limit_data = 30
    limit_anchors = 1
    limit_reasonable = 60

fixed_anchors = 10