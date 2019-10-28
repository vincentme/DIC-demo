import numpy as np
import matplotlib.pyplot as plt
import sys, os, time, pickle
sys.path.insert(0, os.path.abspath('..'))
from lib import dic
floattype = dic.floattype

video_file = '../data/cu_tensile/P1000570.MP4'
result_folder ='./result_cu_tensile/'
start_index = 0
#end_index = 2000
end_index = 11000
output_frame_list = np.arange(start_index, end_index, 100, dtype = np.int64)
#output_frame_list = None
subset_level = 10
ZNSSD_limit = 0.4
lost_level = 0.1
step = 10
subpixel_method = 'fagn'
POIs_done = True
#POIs_done = False



reader = dic.video_loader(video_file, maximum_frame = 11178)
if not os.path.exists(result_folder):
    os.mkdir(result_folder)
if not POIs_done:
    plt.figure()
    dic.plot_img_click(reader[start_index])
else:
    POIs = dic.gene_POIs([
[1257.0451846476099, 954.2524400054277],
[1648.0038473617562, 960.7222051094443],
[2009.3864410289696, 960.7222051094443],
[2013.0834496598363, 1187.1639837500256],
[1971.4921025625865, 1183.4669751191589],
[1652.6251081503397, 1182.5427229614422],
[1257.0451846476099, 1180.694218646009]
], step = step, 
    round_int = True)
    plt.figure()
    dic.plot_img_points(reader[0], POIs, size = 1.2)

    if os.path.exists('{}/result.dic'.format(result_folder)):
        result_graph, POI_graph_list = pickle.load(open('{}/result.dic'.format(result_folder), 'rb'))
    else:
        t1 = time.time()
        result_graph, POI_graph_list = dic.bisection_search(reader, start_index, end_index, POIs, output_frame_list, lost_level = lost_level)
        t2 = time.time()
        print('calculation time {:.2f}s'.format(t2 - t1))

#    for frame_index in result_graph.nodes:
    for frame_index in output_frame_list:
        plt.figure()
        dic.plot_img_points(reader[frame_index], result_graph.nodes[frame_index]['POIs'], size = 0.5)
        plt.title('frame index {}'.format(frame_index))
        plt.savefig('{}/frame_{:02d}.png'.format(result_folder, frame_index), pad_inches=0, bbox_inches = 'tight', dpi = 200)
        
    plt.figure()
    dic.plot_img_points_trip(reader[end_index], result_graph.nodes[end_index]['POIs'], result_graph.nodes[end_index]['green_strain'][..., 3], vmin = 0, vmax = 2)
    plt.title('equivalent green strain, frame index {}'.format(end_index))
    plt.savefig('{}/strain.png'.format(result_folder, end_index), pad_inches=0, bbox_inches = 'tight', dpi = 200)
