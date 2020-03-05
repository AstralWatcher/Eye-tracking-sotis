# from sklearn.cluster import DBSCAN
# import numpy as np
# X = np.array([[1, 2,3], [2, 2,2], [2, 3,2],[8, 7,2], [8, 8,3], [25, 80,4]])
# clustering = DBSCAN(eps=3, min_samples=2).fit(X)
# print(clustering.labels_)

def plot_tsne(array_to_plot, labels):
    import pandas as pd
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import seaborn as sns

    feat_cols = ['pixel' + str(i) for i in range(len(array_to_plot))]
    df = pd.DataFrame(array_to_plot, columns=feat_cols)
    df['y'] = labels
    df['label'] = df['y'].apply(lambda i: str(i))
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(df)
    print('t-SNE done!')
    map_to_pd = {'tsne-2d-one': tsne_results[:, 0], 'tsne-2d-two': tsne_results[:, 1]}
    df = pd.DataFrame(map_to_pd)
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", 10),
        data=df,
        legend="full",
        alpha=0.3
    )


plot_tsne([[1, 2, 3], [1, 2, 4], [10, 20, 30, 40, 50], [100, 200]], [1, 1, 2, 3])
