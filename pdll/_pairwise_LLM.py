import numpy as np
import pandas as pd
from pairwise_NLP_LLM.NLP_data_processing import query_the_api
from sklearn.metrics import cohen_kappa_score, mean_squared_error


class PairwiseDifferenceClassifierLLM():
    llm = ""
    rubric: str
    max_score: int
    digits = 5
    prompt = f"""
    
                You are an expert text comparison assistant.
                
                Task:
                Strictly evaluate the two essays according to the rubric below.
                
                Rules:
                Return only one signed integer: the score difference (Essay 1 score minus Essay 2 score).
                Do NOT include any explanations, comments, extra characters, whitespace, or anything besides a single signed integer.
                Any output other than a single signed integer will be considered invalid."""
    
    
    def __init__(self, llm, rubric, max_score):
        # In the future, include some form of validity check here for availability of the llm as well as access
        self.llm = llm
        self.rubric = rubric
        self.max_score = max_score
    
    def change_scoring_rubric(self, new_rubric, new_max_score):
        self.rubric = new_rubric
        self.max_score = new_max_score
        
    def change_llm(self, new_llm):
        # In the future, include some form of validity check here for availability of the llm as well as access
        self.llm = new_llm
   
    
    
    def predict(self, data, anchors, rubric) -> pd.DataFrame:
        predictions = []

        for i, unknown_datapoint in data.iterrows():
            store_pred_scores = []

            for j, anchor_datapoint in anchors.iterrows():                
                try:                    
                    prompt_customization = f"""
                        Rubric:
                        {rubric}

                        Essay 1:
                        {unknown_datapoint["essay"]}
                        
                        Essay 2:
                        {anchor_datapoint["essay"]}
                        """
                    prompt_combined = self.prompt + prompt_customization
                    
                    answer = query_the_api(self.llm, prompt_combined)
                        
                    score_of_baseline_essay = int(anchor_datapoint.iloc[1])
                    
                    try:
                        pred_diff = int(answer)
                    except:
                        pred_diff = 0                    

                    score_pred = score_of_baseline_essay + pred_diff

                    store_pred_scores.append(score_pred)

                except (ValueError, IndexError, AttributeError):
                    raise RuntimeError()

                except Exception:
                    raise RuntimeError()
                
            avg_score = 0
            # if the list is empty, it means no valid differences were found
            if store_pred_scores != []:
                avg_score = int(round(np.mean(store_pred_scores), 0))

            predictions.append(int(avg_score))
        
        data["y_pred"] = predictions
        data["diff"] = data["y_pred"] - data["score"]
        return data

    def normalize_scores(self, data_pred):
        # normalizing scores with their max value for better comparison
        df = pd.DataFrame(data_pred)
        df["score"] = (data_pred["score"] / self.max_score)
        df["y_pred"] = (data_pred["y_pred"] / self.max_score)
        return df

    def metrics(self, data_pred):
        if 'y_pred' in data_pred:
            # compute QWK
            qwk = round(cohen_kappa_score(data_pred["score"], data_pred["y_pred"], weights="quadratic"), self.digits)
            
            
            data_metrics = self.normalize_scores(data_pred)
            # compute MSE
            mse = round(mean_squared_error(data_metrics["score"], data_metrics["y_pred"]), self.digits)
            
        else:
            qwk, mse = 0, 0

        return mse, qwk
