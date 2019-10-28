import numpy as np
import matplotlib.pyplot as plt
import sys, os, time
sys.path.insert(0, os.path.abspath('..'))
from lib import dic
floattype = dic.floattype

image_folder = '../data/Sample9-rotation/'
result_folder ='./result_09-rotation/'
start_index = 0
end_index = 10
subset_level = 10
ZNSSD_limit = 0.4
subpixel_method = 'fagn'
POIs_done = True
#POIs_done = False


reader = dic.image_folder_loader(image_folder, suffix = 'tif')
output_frame_list = np.arange(1, end_index + 1, dtype = np.int64)
if not os.path.exists(result_folder):
    os.mkdir(result_folder)
if not POIs_done:
    plt.figure()
    dic.plot_img_click(reader[start_index])
else:
    POIs = dic.gene_POIs([
[21.59277987148556, 20.35036466664849],
[20.898693164872725, 488.16480492367754],
[486.63087330206326, 485.3884580972263],
[485.93678659545037, 20.35036466664849]
], 20, round_int = True)
    plt.figure()
    dic.plot_img_points(reader[0], POIs, size = 1.2)

    t1 = time.time()
    result_graph, POI_graph_list = dic.bisection_search(reader, start_index, end_index, POIs, output_frame_list)
    t2 = time.time()
    print('calculation time {:.2f}s'.format(t2 - t1))
    
    for frame_index in output_frame_list:
        plt.figure()
        dic.plot_img_points(reader[frame_index], result_graph.nodes[frame_index]['POIs'], size = 1.2)
        plt.title('frame index {}'.format(frame_index))
        plt.savefig('{}/frame_{:02d}.png'.format(result_folder, frame_index), pad_inches=0, bbox_inches = 'tight', dpi = 200)
        



