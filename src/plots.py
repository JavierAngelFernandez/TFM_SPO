import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


plt.style.use("ggplot")


def get_pnl_graph(df_results: pd.DataFrame, colores: dict, cols: list, save_path: str = None):
    """Realiza la gráfica del pnl acumulado."""
    fig, ax = plt.subplots(figsize=(17, 9), dpi=220)

    for col in cols:
        ax.plot(df_results.index, df_results[col],
            lw=3,
            color=colores[col],
            label=col
        )
        ax.scatter(df_results.index[-1], df_results[col].iloc[-1],s=80, color=colores[col], zorder=5)
    
    ax.set_title("Comparación del beneficio acumulado",
             fontsize=26,
             weight="bold")

    ax.set_ylabel("Beneficio acumulado (€)", fontsize=23)
    ax.set_xlabel("Fecha", fontsize=23)

    ax.legend(fontsize=15)

    ax.grid(alpha=1.0)

    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    fig.autofmt_xdate()
    ax.tick_params(axis='x', labelsize=21)
    ax.tick_params(axis='y', labelsize=21)

    plt.tight_layout()
     
    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    return fig, ax

    


