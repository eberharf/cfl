'''
Iman Wahle and Jenna Kahn
1/8/20
Sankey diagram code

Create a Sankey diagram to show how samples move between clusters when cluster
parameters are varied. 

Usage of this function:

```
  import plotly.graph_objects as go
  from cfl.visualization_methods import clustering_to_sankey as sk

  #x_lbls_L = list of x labels from several different rounds of clustering on the same data

  link, label = sk.convert_lbls_to_sankey_nodes(x_lbls_L)
  # plot
  fig = go.Figure(data=
            [go.Sankey(node = dict(pad = 15, thickness=20, label = label, color =  "blue"),
                      link = link)])

  fig.update_layout(title_text="Sample Sankey", font_size=10)
  fig.show()
```
'''

import numpy as np


def convert_lbls_to_sankey_nodes(x_lbls_L):
    '''
    Convert cluster labels into source, target, and value information

    Arguments:
        x_lbls_L: x_lbls_L is a list of x_lbls, the result from multiple kmeans 
            clusterings on the same data

    Returns:
        link (dict): a representation of nodes and weighted connections between
            them to make a Sankey diagram
        labels (list): labels for every node in the sankey diagram
    '''

    # create list of the numbers of clusters used
    # (the names list looks like ['2:', '3:', '4:', '5:', '6:']
    # if Kmeans clustering was done with 2-6 clusters)
    names = ['{}:'.format(len(np.unique(x_lbls))) for x_lbls in x_lbls_L]

    # label is a list of all the labels on each section of the graph in sequential order
    # (ie ['2:0', '2:1', '3:0', '3:1', ....])
    # the number after the colon refers to the cluster labels on the next set of clusters
    label = [names[0] + str(i) for i in np.unique(x_lbls_L[0])]

    # the Sankey diagram is represented as a graph
    # where each index of 'source' and 'target' gives a connection between one node and another
    # and 'value' gives the amount of flow from the source to the target
    source = []
    target = []
    value = []

    source_Ls = x_lbls_L[:-1]
    target_Ls = x_lbls_L[1:]
    aindex = 0
    # iterate through each pair of source and target lists of cluster labels
    for ci, (cA, cB) in enumerate(zip(source_Ls, target_Ls)):
        n1 = names[ci+1]
        bindex = len(label)

        label = label + [n1 + str(i) for i in np.unique(cB)]

        # iterating over each source, target pair in the list ...
        for a in range(len(np.unique(cA))):
            for b in range(len(np.unique(cB))):

              # add clusters to source and target list
                source.append(a + aindex)
                target.append(b + bindex)
                # source.append(float(str(a) +"."+ str(aindex)))
                # target.append(float(str(b) +"."+ str(bindex)))

                # calculate amount of flow between source and target
                # as the number of samples that are part of both this source and target node

                value.append(np.sum((cA == a) & (cB == b)))
        aindex = bindex

        # put results into a dict bc that's what Sankey wants
        link = dict(source=source, target=target, value=value)
    return link, label
