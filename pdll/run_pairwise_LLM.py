from operator import index
import _pairwise_LLM as pw_LLM
import pairwise_NLP_LLM.NLP_config as _
import pairwise_NLP_LLM.NLP_data_processing as dp
import pandas as pd


def reduce_the_data(data, limit: int) -> pd.DataFrame:
    data = pd.DataFrame(data)
    limit_data = min(limit, len(data))
    data = data.sample(limit_data, random_state=_.random_seed)
    return data


def retrieve_data():
    data_train, data_dev, data_test = dp.get_data(
        _.fold_ID, _.essay_set, True, True
    )
    
    data = reduce_the_data(data_train, _.limit_data)
    anchors = reduce_the_data(data_dev, _.limit_anchors)
    return data, anchors


def run_prediction(data, anchors):
    PDC_llm = pw_LLM.PairwiseDifferenceClassifierLLM(_.llm, _.rubric, _.max_score)
    
    data_pred = PDC_llm.predict(data, anchors, _.rubric)
    
    print(data_pred)
    print("\n")

    mse, qwk = PDC_llm.metrics(data_pred)
    
    print(f"MSE: {mse}")
    print(f"QWK: {qwk}")
    
    # data_pred.to_csv("result_for_test_LLM_DataFrame.cvs")
    # with open("result_for_test_LLM_Metrics.txt", "w") as f:
    #     f.write(f"{mse}\n{qwk}")

   
if __name__ == "__main__":
    data, anchors = retrieve_data()
    run_prediction(data, anchors)