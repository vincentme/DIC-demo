import numpy as np
import matplotlib.pyplot as plt
import sys, os, time
sys.path.insert(0, os.path.abspath('..'))
from dic import dic
floattype = dic.floattype

image_folder = '../data/Sample13-tensile/'
result_folder ='./result_13-tensile_elongation/'
start_index = 50
end_index = 0
output_frame_list = np.arange(start_index, end_index - 1, -1, dtype = np.int64)
#output_frame_list = None
subset_level = 10
ZNSSD_limit = 0.4
lost_level = 0.1
step = 10
subpixel_method = 'fagn'
POIs_done = True
#POIs_done = False



reader = dic.image_folder_loader(image_folder, suffix = 'tif')

if not os.path.exists(result_folder):
    os.mkdir(result_folder)
if not POIs_done:
    plt.figure()
    dic.plot_img_click(reader[start_index])
else:
    POIs1 = dic.gene_POIs([
[421.37349185261496, 18.445964265379644],
[425.14758831982203, 54.29988070384729],
[617.6265081473848, 54.29988070384729],
[617.6265081473848, 18.445964265379644]
], step = step, round_int = True)
    num_POIs1 = len(POIs1)
    POIs2 = dic.gene_POIs([
[436.4698777214435, 1282.768280779763],
[440.24397418865055, 1320.5092454518342],
[591.2078328769351, 1320.5092454518342],
[594.9819293441421, 1267.6718949109345]
], step = step, round_int = True)
    
    POIs = np.concatenate((POIs1, POIs2))
    plt.figure()
    dic.plot_img_points(reader[start_index], POIs, size = 1.2)
    
    t1 = time.time()
    result_graph, POI_graph_list = dic.bisection_search(reader, start_index, end_index, POIs, output_frame_list, lost_level = lost_level)
    t2 = time.time()
    print('calculation time {:.2f}s'.format(t2 - t1))
    
    distance = []
    for frame_index in output_frame_list:
        distance.append((np.sum((np.mean(result_graph.nodes[frame_index]['POIs'][:num_POIs1], 0) - np.mean(result_graph.nodes[frame_index]['POIs'][num_POIs1:], 0))**2))**0.5)
    
    elongation = (distance - distance[50])/distance[50]
    plt.figure()
    plt.plot(output_frame_list, elongation)
    plt.title('elongation vs. frame index')
    plt.savefig('{}/elongation.png'.format(result_folder), pad_inches=0, bbox_inches = 'tight', dpi = 200)


​    

