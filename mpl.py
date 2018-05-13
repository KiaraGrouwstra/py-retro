from matplotlib import pyplot as plt
# from matplotlib.pyplot import plot, ion, show

def plot_reward(stats, hideplot=False):
    return plot_line("Reward", stats, hideplot)

def plot_cum_reward(stats, hideplot=False):
    return plot_line("Cumulative Reward", stats, hideplot)

def plot_line(label, stats, hideplot=False):
    fig = plt.figure(figsize=(10,5))
    plt.plot(stats)
    plt.xlabel("Timestep")
    plt.ylabel(label)
    plt.title(label + " over Time")
    return mk_plot(fig, hideplot)
    # return plt.plot(fig)

def plot_action_count(y, hideplot=False):
    N = len(y)
    x = range(N)
    width = 1/1.5
    fig = plt.figure(figsize=(10,5))
    plt.bar(x, y, width)
    plt.xlabel("Action")
    plt.ylabel("Count")
    plt.title("Actions' Distribution")
    return mk_plot(fig, hideplot)
    # return plt.plot(fig)

def mk_plot(fig, hideplot):
    if hideplot:
        plt.close(fig)
    else:
        # plt.plot(fig)
        # plt.show(fig, block = False)
        # fig.block = False
        # plt.show(fig)
        plt.show(block=False)
