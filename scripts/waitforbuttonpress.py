import matplotlib.pylab as plt

closed = False


def handle_close(evt):
    global closed
    closed = True


def wait():
    while plt.waitforbuttonpress(0.2) is None:
        if closed:
            return


def is_closed():
    return closed


def figure():
    global closed
    closed = False
    fig = plt.figure()
    fig.canvas.mpl_connect('close_event', handle_close)
    return fig