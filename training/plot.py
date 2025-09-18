import plotly.graph_objects as go

def plot_loss_curve_plotly(metrics_df):
    """
    Plot training and validation loss curves from metrics_df using Plotly.
    - Only the minimum point of each curve will have a marker.
    Args:
        metrics_df (pd.DataFrame): must contain columns ["epoch", "train_loss", "val_loss"]
    """
    fig = go.Figure()

    # --- Train Loss Line ---
    fig.add_trace(go.Scatter(
        x=metrics_df["epoch"],
        y=metrics_df["train_loss"],
        mode="lines",
        name="Train Loss",
        line=dict(color="blue"),
    ))

    # Marker at the minimum train loss
    min_train_idx = metrics_df["train_loss"].idxmin()
    fig.add_trace(go.Scatter(
        x=[metrics_df.loc[min_train_idx, "epoch"]],
        y=[metrics_df.loc[min_train_idx, "train_loss"]],
        mode="markers+text",
        name="Min Train Loss",
        marker=dict(color="blue", size=10, symbol="circle"),
        text=[f"{metrics_df.loc[min_train_idx, 'train_loss']:.4f},{int(metrics_df.loc[min_train_idx, 'ckpt_id'])}"],
        textposition="bottom right"
    ))

    # --- Validation Loss Line ---
    fig.add_trace(go.Scatter(
        x=metrics_df["epoch"],
        y=metrics_df["val_loss"],
        mode="lines",
        name="Validation Loss",
        line=dict(color="red"),
    ))

    # Marker at the minimum validation loss
    min_val_idx = metrics_df["val_loss"].idxmin()
    fig.add_trace(go.Scatter(
        x=[metrics_df.loc[min_val_idx, "epoch"]],
        y=[metrics_df.loc[min_val_idx, "val_loss"]],
        mode="markers+text",
        name="Min Val Loss",
        marker=dict(color="red", size=10, symbol="square"),
        text=[f"{metrics_df.loc[min_val_idx, 'val_loss']:.4f},{int(metrics_df.loc[min_val_idx, 'ckpt_id'])}"],
        textposition="bottom right"
    ))

    # Layout configuration
    fig.update_layout(
        title="Training and Validation Loss Curve",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        legend=dict(x=0.02, y=0.98),
        template="plotly_white"
    )

    fig.show()