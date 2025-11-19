from matplotlib import pyplot as plt

def init_plot():
    """
    Initializes interactive/live plot for the timelapse pipeline.

    Returns figure and ax objects
    """
    plt.ion()

    fig, ax = plt.subplots(figsize=(8,5))

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Colony count")
    ax.set_title("Colony growth")

    return fig, ax

def update_live_plot(dish_counts, fig, ax):
    """
    Updates live plot with colony counts from the timelapse pipeline.
    """
    ax.clear()

    for dish_idx in sorted(dish_counts.keys()):
        times  = [t[0] for t in dish_counts[dish_idx]]
        counts = [t[1] for t in dish_counts[dish_idx]]

        ax.scatter(times, counts, label=f"Dish {dish_idx+1}")

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Colony count")
    ax.set_title("Colony growth")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.05)