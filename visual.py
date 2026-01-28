import pandas as pd
import pickle
import plotly.express as px
import umap.umap_ as umap
from dataclasses import dataclass

###### Dimensionality reduction using UMAP #######
# Hyperparameters are controlled via umap_kwargs

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


###### Plotly Visualization #######
# Visual settings are managed by VisualConfig
@dataclass
class VisualConfig:
    fig_width: int = 900
    fig_height: int = 700

    marker_size: int = 6
    marker_opacity: float = 0.8

    font_size: int = 8
    text_position: str = "top center"

    init_mode:str = "markers"

    init_dragmode:str = "zoom"
    color_palette = px.colors.qualitative.Dark24

def visual(emb_df, 
             tag_dict, 
             save_dir, 
             n_components: int, 
             tag_name="Main Genre", 
             post_script:str = "", 
             umap_kwargs=None,
             visual_cfg: VisualConfig | None=None):
    
    if visual_cfg is None:
        visual_cfg = VisualConfig()
    
    assert visual_cfg.init_dragmode in ["zoom", "pan"], \
        f"init_dragmode must be 'zoom' or 'pan', got {visual_cfg.init_dragmode}"    
    
    emb_df = emb_df.copy()
    
    coords = umap_project(emb_df.drop(columns=["style"]), n_components=n_components, umap_kwargs = umap_kwargs)

    for i in range(n_components):
        emb_df[f"dim_{i}"] = coords[:, i]
    
    # Determine axes (x, y) or (x, y, z) based on n_components
    axis_kwargs = {axis: f"dim_{i}"
                for i, axis in enumerate(["x", "y", "z"][:n_components])}

    # Style and genre names use "_" as spacers; replace with spaces for better readability
    emb_df[tag_name] = [tag_dict.get(style, "Unknown").replace("_", " ") for style in emb_df["style"]]
    emb_df['style'] = [style.replace("_", " ") for style in emb_df["style"]]

    if n_components == 3:
        px.scatter = px.scatter_3d

    fig = px.scatter(
        emb_df,
        text="style",
        hover_name="style",
        color=tag_name,
        color_discrete_sequence=visual_cfg.color_palette,
        **axis_kwargs
    )

    fig.update_traces(
        marker=dict(size=visual_cfg.marker_size, opacity=visual_cfg.marker_opacity),
        textposition=visual_cfg.text_position,
        textfont=dict(size=visual_cfg.font_size),
        mode=visual_cfg.init_mode
    )

    # Add toggle buttons to show/hide text labels
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        label="Show Text",
                        method="restyle",
                        args=["mode", "markers+text"]
                    ),
                    dict(
                        label="Hide Text",
                        method="restyle",
                        args=["mode", "markers"]
                    ),
                ],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top"
            ),
        ]
    )

    # Set initial interaction mode (e.g., 'pan' or 'zoom')    
    fig.update_layout(
        dragmode=visual_cfg.init_dragmode
    )

    fig.write_html(
        save_dir,
        include_plotlyjs="cdn",
        post_script = post_script
    )

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--search_type",
        type=str,
        default="masters",
        help="Discogs search type (e.g. masters, releases)"
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="Node2Vec",
        help="Embedding algorithm name"
    )
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    search_type = args.search_type
    algo = args.algo

    emb_df = pd.read_csv(f"./embedding_data/{algo}/embedding_df_{search_type}.csv")

    with open(f"./embedding_data/max_genre_counter_{search_type}.pkl", "rb") as f:
        max_genre_counter = pickle.load(f)

    style_to_main_genre = {
        st: cnt.most_common(1)[0][0]
        for st, cnt in max_genre_counter.items()
    }

    save_dir = f"./docs/style_{algo}_{search_type}_umap"


    visual(emb_df=emb_df, 
           tag_dict=style_to_main_genre, 
           save_dir = f"{save_dir}.html", 
           n_components=2, 
           tag_name="Main Genre", 
           umap_kwargs=None,
           visual_cfg = VisualConfig(fig_width=1600, fig_height=1200))
    
    visual(emb_df=emb_df, 
           tag_dict=style_to_main_genre, 
           save_dir = f"{save_dir}_3d.html", 
           n_components=3, 
           tag_name="Main Genre",
           umap_kwargs=None,
           visual_cfg = VisualConfig())