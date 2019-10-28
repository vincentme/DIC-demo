import numpy as np
import matplotlib.pyplot as plt
import sys, os, time
sys.path.insert(0, os.path.abspath('..'))
from dic import dic
floattype = dic.floattype

image_folder = '../data/Sample12-tensile-with-hole/'
result_folder ='./result_12-tensile-with-hole/'
start_index = 0
end_index = 11
subset_level = 10
ZNSSD_limit = 0.4
lost_level = 0.1
step = 5
subpixel_method = 'fagn'
POIs_done = True
#POIs_done = False



reader = dic.image_folder_loader(image_folder, suffix = 'tiff')
output_frame_list = np.arange(1, end_index + 1, dtype = np.int64)
if not os.path.exists(result_folder):
    os.mkdir(result_folder)
if not POIs_done:
    plt.figure()
    dic.plot_img_click(reader[start_index])
else:
    POIs = dic.gene_POIs([
[62.44543441277847, 385.8817330204143],
[63.85594786563178, 716.8822232899854],
[336.08504426631583, 720.1734213466431],
[333.7341885115604, 384.00104841660993]
], step = step, 
    list_interiors_vertex = [np.array(
    [
[192.2126720752807, 487.90887277680196],
[213.84054501903108, 492.1404131353618],
[230.2965353023194, 502.0140073053348],
[242.5209852270479, 521.7611956452808],
[246.28235443465667, 547.6206089475911],
[237.34910256658583, 573.9501934008524],
[217.13174307568875, 589.9360125331897],
[192.2126720752807, 596.988579797456],
[164.00240301821503, 589.4658413822385],
[147.07624158397562, 570.6589953441948],
[139.08333201780698, 544.3294108909334],
[142.84470122541575, 517.999826437672],
[160.24103381060627, 496.3719534939217]
     ]
            )], 
    round_int = True)
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
    dic.plot_img_points_trip(reader[end_index], result_graph.nodes[end_index]['POIs'], result_graph.nodes[end_index]['green_strain'][..., 3], vmin = 0, vmax = 0.06)
    plt.title('equivalent green strain, frame index {}'.format(end_index))
    plt.savefig('{}/strain.png'.format(result_folder, end_index), pad_inches=0, bbox_inches = 'tight', dpi = 200)

