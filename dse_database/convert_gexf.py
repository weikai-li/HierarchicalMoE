import networkx as nx
from os.path import join
import matplotlib.pyplot as plt

SOURCE = '/home/atefehSZ/RL/original-software-gnn/software-gnn/dse_database/programl/poly/processed'

gexf_file = 'mvt_processed_result.gexf'
g = nx.readwrite.gexf.read_gexf(join(SOURCE, gexf_file))
dot_file = 'mvt.dot'
nx.drawing.nx_pydot.write_dot(g, join(SOURCE, dot_file))
# plt.figure(figsize=(20,20))
# nx.draw(g, with_labels=True, font_color='#000000', node_color = 'r', font_size =8, node_size =20)
# plt.show()