import pandas as pd 
from kopeer import normalize_df, optimize_weights, compute_pairs

def generate_pairs(data: dict, iterations: int = 100):

    df = pd.DataFrame(data).T
    df.columns = ["c", "p", "m", "i"]

    #Normalizar 
    df_norm = normalize_df(df)

    #Optimizar pesos 
    weigths= optimize_weights(df_norm, iterations=iterations)

    #Generación de pares 
    pairs = compute_pairs(df_norm, weigths)

    return {
        "weights": weigths.tolist(),
        "pairs": pairs.to_dict(orient="records")
    }