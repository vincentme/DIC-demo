import numpy as np
import matplotlib.pyplot as plt
import sys, os, time
sys.path.insert(0, os.path.abspath('..'))
from dic import dic
floattype = dic.floattype

image_folder = '../data/Sample3-translation/'
result_folder ='./result_03-translation/'
start_index = 0
end_index = 11
subset_level = 10
ZNSSD_limit = 0.4
subpixel_method = 'fagn'
POIs_done = True
#POIs_done = False

true_displacement = np.zeros((end_index, 2), floattype)
true_displacement[:, 0] = np.arange(start_index, end_index, dtype = floattype)*0.1
true_displacement[:, 1] = np.arange(start_index, end_index, dtype = floattype)*0.1


reader = dic.image_folder_loader(image_folder, suffix = 'tif')
output_frame_list = np.arange(1, end_index + 1, dtype = np.int64)
if not os.path.exists(result_folder):
    os.mkdir(result_folder)
if not POIs_done:
    plt.figure()
    dic.plot_img_click(reader[start_index])
else:
    POIs = dic.gene_POIs([
[20.322966507177114, 19.09808612440179],
[24.522556390977513, 485.2525632262474],
[489.27717019822273, 489.4521531100478],
[485.7775119617223, 16.99829118250159]
], 20, round_int = True)
    plt.figure()
    dic.plot_img_points(reader[0], POIs, size = 1.2)

    t1 = time.time()
    result_graph, POI_graph_list = dic.bisection_search(reader, start_index, end_index, POIs, output_frame_list)
    t2 = time.time()
    print('calculation time {:.2f}s'.format(t2 - t1))
    
    bias_list = []
    std_list = []
    for frame_index in output_frame_list:
        plt.figure()
        dic.plot_img_points(reader[frame_index], result_graph.nodes[frame_index]['POIs'], size = 1.2)
        plt.title('frame index {}'.format(frame_index))
        plt.savefig('{}/frame_{:02d}.png'.format(result_folder, frame_index), pad_inches=0, bbox_inches = 'tight', dpi = 200)
        
        calc_displacement = result_graph.nodes[frame_index]['POIs'] - POIs
        error = calc_displacement - true_displacement[frame_index - 1]
        r_error = (np.sum(error**2, 1))**0.5
        bias_list.append(np.mean(r_error))
        std_list.append(np.std(r_error))
    
    plt.figure()
    plt.plot(output_frame_list, bias_list, label = 'bias')
    plt.plot(output_frame_list, std_list, label = 'std')
    plt.legend()
    plt.ylim(bottom = 0)
    plt.savefig('{}/summary.png'.format(result_folder, frame_index), pad_inches=0, bbox_inches = 'tight', dpi = 200)







