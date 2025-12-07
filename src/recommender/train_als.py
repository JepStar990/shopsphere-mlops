import os
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares

class ALSRecommender(mlflow.pyfunc.PythonModel):
    def __init__(self, user_factors, item_factors, user_index, item_index):
        self.user_factors = user_factors
        self.item_factors = item_factors
        self.user_index = user_index   # dict: user_id -> internal index
        self.item_index = item_index   # dict: item_id -> internal index
        self.rev_item_index = {v: k for k, v in item_index.items()}

    def predict(self, context, model_input):
        """
        model_input: DataFrame with columns:
          - customer_id (str/int)
          - k (int, optional; default 5)
        Returns: list[dict] one row per input: {"customer_id":..., "rec_list":[{"product_id":..., "score":...}, ...]}
        """
        results = []
        for _, row in model_input.iterrows():
            user_id = row.get("customer_id")
            k = int(row.get("k", 5))
            ui = self.user_index.get(user_id)
            if ui is None:
                results.append({"customer_id": user_id, "rec_list": []})
                continue
            # score all items: user_factors[ui] dot item_factors.T
            scores = self.item_factors @ self.user_factors[ui]
            # top-k indices
            idx = np.argpartition(scores, -k)[-k:]
            top_sorted = idx[np.argsort(scores[idx])[::-1]]
            recs = [{"product_id": str(self.rev_item_index[i]), "score": float(scores[i])} for i in top_sorted]
            results.append({"customer_id": user_id, "rec_list": recs})
        return results


def train_implicit_als(ui_path: str, factors: int = 64, reg: float = 1e-2, iterations: int = 20):
    """
    ui_path: path to user-item interactions parquet with columns [customer_id, product_id, strength]
    """
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment("recommender_als_experiment")

    ui = pd.read_parquet(ui_path)
    users = ui["customer_id"].astype(str).unique().tolist()
    items = ui["product_id"].astype(str).unique().tolist()
    user_index = {u: i for i, u in enumerate(users)}
    item_index = {p: i for i, p in enumerate(items)}

    rows = ui["customer_id"].astype(str).map(user_index).values
    cols = ui["product_id"].astype(str).map(item_index).values
    data = ui["strength"].astype(float).values
    mat = coo_matrix((data, (rows, cols)), shape=(len(users), len(items))).tocsr()

    # Implicit ALS uses item-user matrix convention for training
    model = AlternatingLeastSquares(factors=factors, regularization=reg, iterations=iterations, random_state=42)
    model.fit(mat.T)

    user_factors = model.user_factors    # shape: (n_items, factors) because model trained on item-user; we’ll swap names
    item_factors = model.item_factors    # shape: (n_users, factors)

    # NOTE: because of item-user training, we want user_factors to correspond to users.
    # The model's `user_factors` actually correspond to items; `item_factors` correspond to users.
    # Swap semantics for our wrapper:
    user_factors_wrapped = item_factors
    item_factors_wrapped = user_factors

    class ALSWrapper(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            pass

        def predict(self, context, model_input):
            recommender = ALSRecommender(user_factors_wrapped, item_factors_wrapped, user_index, item_index)
            return recommender.predict(context, model_input)

    with mlflow.start_run(run_name=f"als_f{factors}_it{iterations}"):
        artifacts = {}
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=ALSWrapper(),
            registered_model_name="recommender_als_model",
        )
