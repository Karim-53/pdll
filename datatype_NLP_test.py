
import numpy as np
import pandas as pd

import pdll._pairwise_LLM as pw_LLM
import pdll.pairwise_NLP_LLM.NLP_config as _
import pdll.pairwise_NLP_LLM.NLP_data_processing as dp
import pdll.run_pairwise_LLM as run


def test_datatype_NLP():
    # Variables
    essay_set = 8
    fold_id = 1
    anchor_number = 7
    datapoint_number = 10
    llm = "gpt-4.1"
    rubric = _.scoring_rubrics[essay_set]
    max_score = _.max_score_per_set[essay_set]
    
    
    # Set the random seed for reproducibility
    np.random.seed(_.random_seed)

    
    data_train, data_dev, data_test = dp.get_data(
        fold_id, essay_set, True, True
    )    
    data = run.reduce_the_data(data_train, datapoint_number)
    anchors = run.reduce_the_data(data_dev, anchor_number)
    
    PDC_llm = pw_LLM.PairwiseDifferenceClassifierLLM(llm, rubric, max_score)    
    data_pred = PDC_llm.predict(data, anchors, rubric)
    data_pred.reset_index()
    mse, qwk = PDC_llm.metrics(data_pred)
    
    mse = round(mse, 3)
    qwk = round(qwk, 1)
    
    # print(data_pred)
    
    data_comp = pd.read_csv("result_for_test_LLM_DataFrame.cvs", index_col=[0])
    
    # print(data_comp)
    
    with open("result_for_test_LLM_Metrics.txt", "r") as f:
        mse_compare = round(float(f.readline()), 3)
        qwk_compare = round(float(f.readline()), 1)
    
    # for pred in data_pred:
    #     assert data_pred[pred]["y_pred"] == data_compare[pred]["y_pred"]
    
    comp = data_pred.compare(data_comp, 1)
    print(comp)
    assert (mse, qwk) == (mse_compare, qwk_compare)


if __name__ == "__main__":
    test_datatype_NLP()