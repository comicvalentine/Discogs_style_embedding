import umap.umap_ as umap
import pandas as pd

DEFAULT_UMAP_KWARGS = dict(n_neighbors=15,
                           min_dist=0.1,
                           metric="cosine",
                           random_state=42)

def umap_project(
    X,
    n_components: int,
    umap_kwargs: dict | None = None
):
    assert n_components in [2, 3], f"n_components must be 2 or 3, got {n_components}"

    kwargs = DEFAULT_UMAP_KWARGS.copy()
    if umap_kwargs:
        kwargs.update(umap_kwargs)

    reducer = umap.UMAP(
        n_components=n_components,
        **kwargs
    )

    return reducer.fit_transform(X)


def adjust_coords(emb_df, scaler=1):
    emb_df["dim_0"] = emb_df["dim_0"]*(emb_df["dim_1"].std()/emb_df["dim_0"].std())*scaler
    return emb_df

if __name__ == "__main__":
    search_type = "masters"
    umap_kwargs = dict(min_dist=0.5, spread=0.6)
    for algo in ["Node2Vec", "Word2Vec"]:
        EMB_DIR = f"./embedding_data/{algo}/embedding_df_{search_type}.csv"
        emb_df = pd.read_csv(f"./embedding_data/{algo}/embedding_df_{search_type}.csv")
        

        for dimension in [2,3]:
            EMB_UMAP_DIR = f"./embedding_data/{algo}/embedding_df_{search_type}_umap_{dimension}D.csv"
        
            if dimension == 2:
                umap_kwargs = dict(min_dist=0.5, spread=0.6)
            if dimension == 3:
                umap_kwargs = None
        
            coords = umap_project(emb_df.drop(columns=["style"]), n_components=dimension, 
                                  umap_kwargs = umap_kwargs)

            for i in range(dimension):
                emb_df[f"dim_{i}"] = -coords[:, i]

            if dimension == 2:
                emb_df = adjust_coords(emb_df, scaler=1.5)
            emb_df.to_csv(EMB_UMAP_DIR)
