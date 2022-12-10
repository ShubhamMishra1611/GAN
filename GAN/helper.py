import matplotlib.pyplot as plt


def plot(matrices,rows,cols):
    fig,axes=plt.subplots(figsize=(10,10),nrows=rows,ncols=cols,sharex=True,sharey=True)
    for axis,image in zip(axes.flatten(),matrices):
        axis.xaxis.set_visible(False)
        axis.yaxis.set_visible(False)
        im=axis.imshow(1-image.reshape((4,4)),cmap='Greys_r')
    return fig,axes
        
        