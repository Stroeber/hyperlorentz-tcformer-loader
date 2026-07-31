import os
import sys

working_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)), "../")
os.chdir(working_dir)

lib_path = os.path.join(working_dir)
sys.path.append(lib_path)

import torch
import matplotlib.pyplot as plt
import umap

from visualization.extract_visuals import get_embeddings
from visualization.embed_utils import restore_center
from sklearn.decomposition import PCA

if __name__ == '__main__':
    net, embeddings, targets = get_embeddings("visualization/saved/checkpoint/epoch_150.pth")

    embeds = torch.cat(list(embeddings))
    targets = torch.cat(list(targets))

    # hyperbolic_mapper = umap.UMAP(output_metric='hyperboloid').fit(embeds)
    # hyperbolic_mapper = hyperbolic_mapper.embedding_
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    #
    # plt.scatter(hyperbolic_mapper[:, 0], hyperbolic_mapper[:, 1], c=targets, cmap='Spectral')
    #
    #
    # plt.show()
    # plt.clf()
    ###################################################################################################################
    # hyperbolic_mapper = umap.UMAP(output_metric='euclidean').fit(embeds)
    # hyperbolic_mapper = hyperbolic_mapper.embedding_
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    #
    # plt.scatter(hyperbolic_mapper[:, 0], hyperbolic_mapper[:, 1], c=targets, cmap='Spectral')
    #
    #
    # plt.show()
    # plt.clf()
    ###################################################################################################################

    pca = PCA(n_components=2)
    components = pca.fit_transform(embeds)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.scatter(components[:, 0], components[:, 1], c=targets, cmap='Spectral')


    plt.show()
    plt.clf()
    ###################################################################################################################
    #
    decoded_embeds = net.decoder_swap(embeds.cuda()).detach().cpu()
    #
    # hyperbolic_mapper = umap.UMAP(output_metric='euclidean').fit(decoded_embeds)
    # hyperbolic_mapper = hyperbolic_mapper.embedding_
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    #
    # plt.scatter(hyperbolic_mapper[:, 0], hyperbolic_mapper[:, 1], c=targets, cmap='Spectral')
    #
    #
    # plt.show()
    # plt.clf()
    #
    # hyperbolic_mapper = umap.UMAP(output_metric='hyperboloid').fit(decoded_embeds)
    # hyperbolic_mapper = hyperbolic_mapper.embedding_
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    #
    # plt.scatter(hyperbolic_mapper[:, 0], hyperbolic_mapper[:, 1], c=targets, cmap='Spectral')
    #
    #
    # plt.show()
    # plt.clf()
########################################################################################################################
    pca = PCA(n_components=2)
    components = pca.fit_transform(decoded_embeds)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.scatter(components[:, 0], components[:, 1], c=targets, cmap='Spectral')


    plt.show()
    plt.clf()

    pca = PCA(n_components=2)
    components = pca.fit_transform(decoded_embeds[...,1:])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.scatter(components[:, 0], components[:, 1], c=targets, cmap='Spectral')


    plt.show()
    plt.clf()