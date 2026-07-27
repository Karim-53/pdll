import logging
import os

import pandas as pd

import pdll.pairwise_NLP_LLM.NLP_caching as caching
from pdll.pairwise_NLP_LLM.NLP_data_processing import query_the_api

# at the end ony one plot, showing the clear difference between baseline and pairwise
# do anaysis and draw up convincing stats that go with it

logger = logging.getLogger("result")


def get_essay_score_as_float(
    essay: str,
    rubric: str,
    model: str,
) -> float:
    prompt = f"""
    You are an expert text comparison assistant.
    
    Task:
    Strictly evaluate the essay according to the rubric below.
    
    Rules:
    Return only one floating point number (not just an integer): the score of the essay.
    Use up to two digits after the decimal point.
    Do NOT round the score to a whole number.
    Do NOT include any explanations, comments, extra characters, whitespace, or anything besides a floating point number.
    Any output other than a floating point number will be considered invalid.

    Rubric:
    {rubric}

    Essay:
    {essay}
    """

    from_cache = caching.lookup_in_cache(model, prompt, False)

    if from_cache is None:
        try:
            answer = query_the_api(model, prompt)
            try:
                pred_score = float(answer)
            except:
                pred_score = 0
            caching.new_cache_entry(model, prompt, pred_score, False)
            logger.info("got score from new prediction")
            return pred_score

        except (ValueError, IndexError, AttributeError) as parse_err:
            raise RuntimeError(
                logger.exception(f"Failed to parse score difference: {parse_err}")
            )

        except Exception as api_err:
            raise RuntimeError(logger.exception(f"OpenAI API call failed: {api_err}"))

    else:
        logger.info("got score from cache")
        return float(from_cache)


def predict_scores_solo(
    data: pd.DataFrame,
    rubric: str,
    model: str,
) -> pd.DataFrame:
    caching.load_cache(False)
    logger.info("score prediction started\n")

    predictions = []

    for i, row_i in data.iterrows():
        logger.info(f"Datapoint {i}")
        try:
            # the score here is predicted and then doubled to mimic the composition of the original score (the sum of two single scores by two different experts)
            score_pred_1 = get_essay_score_as_float(row_i["essay"], rubric, model)

            score_pred = score_pred_1 * 2

            logger.info(f"Score prediction: {score_pred}\n")
            predictions.append(int(score_pred))

        except RuntimeError:
            raise RuntimeError(logger.exception(f"prediction failed for: {row_i}"))

    data["y_pred"] = predictions
    logger.info("predicted scores solo")
    return data


# done: test output with float to avoid multi-queriying of the same


# def get_essay_score_as_int(essay: str, rubric: str) -> int:
#     prompt = f"""
#     Task:
#     Strictly evaluate the essay according to the rubric below.

#     Rules:
#     Return only one integer: the score for the essay.
#     Do NOT include any explanations, comments, extra characters, whitespace, or anything besides a single integer.
#     Any output other than a single integer will be considered invalid.

#     Rubric:
#     {rubric}

#     Essay:
#     {essay}
#     """

#     try:
#         # implement cashing here!
#         # parqet file for dataframe for cashing
#         # ask chat gpt, whole message as needed
#         response = openai.chat.completions.create(
#             model="gpt-4o",
#             temperature=0,
#             messages=[
#                 {
#                     "role": "system",
#                     "content": "You are an expert text comparison assistant.",
#                 },
#                 {"role": "user", "content": prompt},
#             ],
#         )

#         result = response.choices[0].message.content.strip()  # type: ignore
#         return int(result)

#     except (ValueError, IndexError, AttributeError) as parse_err:
#         raise RuntimeError(f"Failed to parse score difference: {parse_err}")

#     except Exception as api_err:
#         raise RuntimeError(f"OpenAI API call failed: {api_err}")
