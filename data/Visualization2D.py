import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('cleanData/oneEncodedDiscord.csv')

def vis2d(xl,yl, wl):
    xs = list(df[xl])
    ys = list(df[yl])
    ws = list(df[wl])
    
    title = "{} vs {}".format(xl,yl)
    
    plt.title(title)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.scatter(xs,ws, c="green")
    plt.scatter(ys,ws, c="blue")
    plt.show()

vis2d('friend_qualities','personal_qualities', 'humour')
vis2d('music_genre', 'entertainment_genre', 'sports')
