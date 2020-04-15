import matplotlib.pyplot as plt
from skimage.color import rgb2hsv
from skimage import io
import numpy as np
from scipy.ndimage.morphology import binary_dilation
from scipy import ndimage
from PIL import Image
import pickle as pkl
from scipy import stats

def evaluate(fname, plateID=None, plate_size=96, well_size=None, plot=True, save=True, save_plot=False, hue_min=0, hue_max=1):
    """
    Evaluate the hues of each sample of a colormetric assay in a microtiter well plate.
    Input:
    `fname`: Image file name.
    `plateID`: Identification number of sample plate.
    `plate_size`(int): number of wells. 96 or 384.
    `well_size(int): Optional size of well in pixels. will be estimated automatically
        if possible.
    `plot`(bool): Plot the results
    `save`(bool): Save the results.
    `save_plot`(bool): Save the plot.
    `hue_min`(flt): Minimum of expected hue of the colormetric test. 
        The HSV color scheme is used, where hue is an angle between 0 and 1.
        Reds can be found on eiter end of the scale as it is a circle. 
        "hum_min" can therefore be set to any minimal angle and can cross the origin.
        Example: to capture all reds set "hue_min" to 0.9 and "hue_max" to 0.1
        This will allow all colors between 0.9 --> 0 --> 0.1
    `hue_max`(flt): Maximum expected hue. See explanation above.
    
    Returns:
        Dictionary with the results, with well IDs as keys and hue as value.
        If selected it saves the plots as image and the results as pickled dictionary.
    
    """
    
    def check_angle(angle, min_angle, max_angle):
        """
        Function to check if hue is within range of expected results.
        Hue is an angle between 0 and 1. 
        Reds can be found at both the start and end of the circle

        """
        # Normal situation
        if min_angle < max_angle:
            if min_angle < angle < max_angle:
                return True
            else:
                return False

        # Range is on either side of the start.
        if min_angle > max_angle:
            if (angle > min_angle and angle < 1) or (angle > 0 and angle < max_angle):
                return True
            else:
                return False
    
    pic = io.imread(fname)
    #Flip left right because image is take from the bottom of the plate
    pic = np.fliplr(pic)
    #Translate RGB to HSV
    hsv_pic = rgb2hsv(pic)

    # Plot raw image
    if plot == True:
        plt.style.use('dark_background')
        fig = plt.figure(constrained_layout=True, figsize=(15,10))
        gs = fig.add_gridspec(4,4)
        ax0 = fig.add_subplot(gs[:2, :2])
        ax0.imshow(pic)
        ax0.set_axis_off()
        ax0.set_title('Raw image', fontsize=20)

    # Try to determine size of the well in the image if not defined. 
    if well_size == None:
        try:
            #get the dpi
            imtemp = Image.open(fname)
            dpi = np.mean((imtemp.info['dpi'][0], imtemp.info['dpi'][1]))
            imtemp.close()

            if plate_size == 96:
                well_diameter = 3 #mm
            if plate_size == 384:
                well_diameter = 2 #mm  Is this correct ??????

            #Calculate expected diameter of well in pixels
            well_size = (well_diameter / 25.4) * dpi #pixels

        except Exception as e:
            raise Exception(f'Could not find the dpi of the image, please provide the "well_size" in pixels to the function. Error: {e}')


    #Use values to make a mask for the wells
    # Filter the image
    val = hsv_pic[:, :, 2].copy()
    val = val - ndimage.filters.gaussian_filter(val, int(well_size / 2))

    # Iterate through thresholds untill the expected number of wells is found.
    for t in np.arange(0, 1, 0.01):
        threshold_v = t
        mask_val = val>threshold_v
        # Expand the wells with a third of the well size
        mask_val = binary_dilation(mask_val, iterations = int(well_size/3))
        # Segment the image of all wells
        labels, nobjects = ndimage.label(mask_val.astype(int))
        if nobjects == plate_size:
            break        

    #Use the saturation to maks wells with samples
    saturation = hsv_pic[:, :, 1].copy()
    saturation = ndimage.filters.gaussian_filter(saturation, int(well_size / 3))

    # Iterate through thresholds to find wells with samples
    sat_thresholds = []
    sat_threshold_result = []
    for t in np.arange(0.005, 1, 0.001):
        mask_sat = saturation > t
        mask_sat = binary_dilation(mask_sat, iterations= int(well_size/5))
        labels, nobjects = ndimage.label(mask_sat.astype(int))
        sat_thresholds.append(t)
        sat_threshold_result.append(nobjects)
        if nobjects == 0:
            break
    sat_thresholds = np.array(sat_thresholds)
    sat_threshold_result = np.array(sat_threshold_result) 

    # Find the threshold for the positive wells
    sat_result_mask = np.bitwise_and(sat_threshold_result > 10, sat_threshold_result <= plate_size)
    mode = stats.mode(sat_threshold_result[sat_result_mask])[0]
    sat_threshold = sat_thresholds[sat_threshold_result == mode][int(sat_thresholds[sat_threshold_result == mode].shape[0]/2)]

    # Make the final mask with the found threshold
    mask_sat = saturation > sat_threshold
    mask_sat = binary_dilation(mask_sat, iterations= int(well_size/5))
    labels, nobjects = ndimage.label(mask_sat.astype(int))

    #Combine masks
    mask = np.logical_and(mask_val, mask_sat)


    #masked image
    result  = hsv_pic[:, :, 0].copy()
    result_masked = np.ma.masked_where(mask==False, result)

    #Combine masks
    mask = np.logical_and(mask_val, mask_sat)

    if plot == True:
        #Plot maskes
        ax2 = fig.add_subplot(gs[2, 0])
        ax2.imshow(mask_val)
        ax2.set_title('Value mask, all wells', fontsize=16)
        ax2.set_axis_off()
        ax3 = fig.add_subplot(gs[2, 1])
        ax3.imshow(mask_sat)
        ax3.set_title('Saturation mask, included wells', fontsize=16)
        ax3.set_axis_off()
        ax4 = fig.add_subplot(gs[2, 2])
        ax4.imshow(mask)
        ax4.set_title('Combined mask', fontsize=16)
        ax4.set_axis_off()
        ax5 = fig.add_subplot(gs[2, 3])
        cmap_ax5 = plt.cm.hsv
        cmap_ax5.set_bad(color='black')
        ax5.imshow(result_masked, cmap=cmap_ax5)
        ax5.set_axis_off()
        ax5.set_title('Well hue for positive wells', fontsize=16)


    # Segment the image of all wells
    labels, nobjects = ndimage.label(mask_val.astype(int))
    label_order = np.unique(labels)[1:]

    # Find the centers of each well
    centers = np.array(ndimage.center_of_mass(mask_val.astype(int), labels=labels, index=np.unique(labels)[1:]))


    # Sort the wells, this works for these images but needs imporvement. 
    # It works well with the distance in pixels between the wells.
    # Using the centroids it should be possible to find this distance.
    # Current solution with 33: 
    # Sum the X and Y value of the centers but multipy the X value so that 
    # centers get a sortable value
    cent_argsort = np.argsort([33*i[0] + i[1] for i in centers])


    # Find the median hue of each positive well
    median_result = np.zeros(nobjects)
    filled = np.ma.filled(result_masked, np.nan)
    # Loop through the sorted centroids and calculate median hue.
    for n, c in enumerate(cent_argsort):
        # Mask the well of interest
        values = filled[labels == label_order[c]]
        values = values[np.isfinite(values)]
        # Check if well has a sample
        if values.size:
            # Median hue, to exclude outliers
            mr = np.median(values)
            # Check if median is withing expected hues
            if check_angle(mr, hue_min, hue_max):
                median_result[n] = mr
            else:
                median_result[n] = np.nan
        else:
            median_result[n] = np.nan

    if plate_size == 96:
        w, h = 8, 12
        well_labels = 'ABCDEFGH'
    if plate_size == 384:
        w, h = 16, 24
        well_labels = 'ABCDEFGHIJKLMNOP'

    result = {}
    count = 0
    for r in well_labels:
        for c in range(h):
            result[f'{r}{c}'] = median_result[count]
            count += 1

    # Save the results
    if save == True:
        if plateID == None:
            save_name = f"{fname.split('.')[0]}.pkl"
        else:
            save_name = f"{plateID}.pkl"
        pkl.dump(result, open(save_name, 'wb'))
        print(f'Results saved as pickeled python dictionary, file name: "{save_name}"')


    if plot == True:
        # Plot results in plate format
        ax1 = fig.add_subplot(gs[0:2, 2:4])
        ax1.imshow(median_result.reshape((8,12)), cmap=plt.cm.hsv, vmin=0, vmax=1) #plt.cm.hsv
        plt.yticks(np.arange(0, 8, 1), labels=[i for i in well_labels])
        ax1.set_ylim(7.5,-0.5)
        plt.xticks(np.arange(0, 12, 1),labels=np.arange(1, h+1, 1))
        ax1.xaxis.tick_top()
        for r in range(w):
            ax1.hlines(r+0.5, -0.5, h-0.5, color='white')
        for c in range(h):
            ax1.vlines(c+0.5, -0.5, w-0.5, color='white')
        ax1.set_title('Median hue of well', fontsize=20)

        # Plot results in HSV format
        ax6 = fig.add_subplot(gs[3, :])
        hue_result = median_result[np.isfinite(median_result)]
        ax6.scatter(hue_result, np.arange(0, 1, 1/hue_result.shape[0]), c=hue_result, cmap='hsv', s=50, vmin=0, vmax=1)
        ax6.set_xlim(0,0.175)
        ax6.set_title('Well hue', fontsize=20)
        ax6.set_xlabel('HSV', fontsize=16)
        ax6.yaxis.set_ticklabels([])
        ax6.yaxis.set_visible(False)
        ax6.yaxis.set_ticks([])
    
    if save_plot == True:
        if plateID == None:
            save_name = f"{fname.split('.')[0]}.png"
        else:
            save_name = f"{plateID}.png"
        plt.savefig(save_name, dpi=300)
    
    return result

