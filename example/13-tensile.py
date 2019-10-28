import numpy as np
import matplotlib.pyplot as plt
import sys, os, time
sys.path.insert(0, os.path.abspath('..'))
from lib import dic
floattype = dic.floattype

image_folder = '../data/Sample13-tenile/'
result_folder ='./result_13-tenile/'
start_index = 0
end_index = 50
subset_level = 10
ZNSSD_limit = 0.4
lost_level = 0.1
step = 20
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
[191.153607352981, 239.23060759699592],
[430.80873302063264, 239.23060759699592],
[623.2876528481954, 254.32699346582444],
[898.796694954315, 276.9715722690671],
[910.1189843559362, 324.147778109156],
[906.3448878887291, 678.9128460266247],
[898.796694954315, 710.9926659978852],
[896.9096467207112, 731.7501965675243],
[910.1189843559362, 744.9595342027493],
[904.4578396551253, 1073.3059268497682],
[176.05722148415248, 1077.0800233169753],
[176.05722148415248, 682.6869424938319],
[174.17017325054894, 590.2215790472576]
], step = step, round_int = True)
    plt.figure()
    dic.plot_img_points(reader[0], POIs, size = 1.2)

    t1 = time.time()
    result_graph, POI_graph_list = dic.bisection_search(reader, start_index, end_index, POIs, output_frame_list, lost_level = lost_level)
    t2 = time.time()
    print('calculation time {:.2f}s'.format(t2 - t1))
    

    for frame_index in output_frame_list:
        plt.figure()
        dic.plot_img_points(reader[frame_index], result_graph.nodes[frame_index]['POIs'], size = 1.2)
        plt.title('frame index {}'.format(frame_index))
        plt.savefig('{}/frame_{:02d}.png'.format(result_folder, frame_index), pad_inches=0, bbox_inches = 'tight', dpi = 200)
        
    plt.figure()
    dic.plot_img_points_trip(reader[47], result_graph.nodes[47]['POIs'], result_graph.nodes[47]['green_strain'][..., 3], vmin = 0, vmax = 2)
    plt.title('equivalent green strain, frame index {}'.format(47))
    plt.savefig('{}/strain.png'.format(result_folder, 47), pad_inches=0, bbox_inches = 'tight', dpi = 200)

