import codecs
import logging
from operator import contains
import os
from pathlib import Path

import openai
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

MAX_SCORE_PER_SET = {
    1: 12,
    2: 10,
    3: 3,
    4: 3,
    5: 4,
    6: 4,
    7: 30,
    8: 60,
}

logger = logging.getLogger("result")


def read_dataset(file_path, prompt_id):
    data_x, data_y, prompt_ids = [], [], []

    score_index = 6
    special_index = 9

    with codecs.open(file_path, mode="r", encoding="UTF8") as input_file:
        next(input_file)
        for line in input_file:
            tokens = line.strip().split("\t")

            essay_id = int(tokens[0])
            essay_set = int(tokens[1])
            content = str(tokens[2].strip())
            score = int(tokens[score_index])
            if essay_set == 2:
                score = score + int(tokens[special_index])

            if essay_set == prompt_id or prompt_id <= 0:
                data_x.append(content)
                data_y.append(score)
                prompt_ids.append(essay_set)
    logger.debug("read dataset and split off target variable")
    return data_x, data_y, prompt_ids


def get_data(fold_id, prompt_id, as_list_of_tuples, to_dataframe):

    dir = Path(f"data-set/asap/fold_{fold_id}")
    paths = [f"{dir}/train.tsv", f"{dir}/dev.tsv", f"{dir}/test.tsv"]

    train_path, dev_path, test_path = paths[0], paths[1], paths[2]

    train_x, train_y, train_prompts = read_dataset(
        train_path,
        prompt_id,
    )

    dev_x, dev_y, dev_prompts = read_dataset(
        dev_path,
        prompt_id,
    )

    test_x, test_y, test_prompts = read_dataset(
        test_path,
        prompt_id,
    )

    if as_list_of_tuples:
        train = list(zip(train_x, train_y))
        dev = list(zip(dev_x, dev_y))
        test = list(zip(test_x, test_y))
    else:
        # as tuple of lists
        train = (train_x, train_y)
        dev = (dev_x, dev_y)
        test = (test_x, test_y)

    logger.debug("split data in train, dev, test set")

    if to_dataframe:
        train, dev, test = convert_to_dataframe([train, test, dev])

    return train, dev, test


def convert_to_dataframe(list_data) -> list[pd.DataFrame]:
    ldf = []

    for data in list_data:
        if isinstance(data, list) and all(isinstance(i, tuple) for i in data):
            # Convert list of tuples to DataFrame
            lot = pd.DataFrame(data, columns=["essay", "score"])
            ldf.append(lot)
        elif isinstance(data, tuple) and len(data) == 2:
            # Convert tuple of lists to DataFrame
            tol = pd.DataFrame(list(zip(*data)), columns=["essay", "score"])
            ldf.append(tol)
        else:
            raise ValueError(
                "Invalid data format. Expected list of tuples or tuple of lists."
            )
    logger.debug("converted list of datasets to list of dataframes")
    return ldf


def query_the_api(model: str, prompt: str):
    # parameter temperature is only supported by gpt models!
    if "gpt" in model:
        response = openai.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
    else:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

    answer = response.choices[0].message.content.strip()  # type: ignore
    logger.debug("queried the API")
    return answer


def normalize_score(df: pd.DataFrame, essay_set: int):
    normalizer = MAX_SCORE_PER_SET[essay_set]
    df["score"] = df["score"] / normalizer
    df["y_pred"] = df["y_pred"] / normalizer
    logger.debug("normalized score with the corresponding max value")
    return df


# for large scale testing log EVERYTHING
# so later you can make detailed analysis
# etc bigger vs smaller text size

# to do
# make it work with all essay sets
#
# see which essay sets to use?

# analysis of completed run:
# When does PDL work better than the baseline?
# impact of number of reference esssays?

# write part of chapter 1 of the thesis, the introduction
