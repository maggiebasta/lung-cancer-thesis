import math
import matplotlib.pyplot as plt
import pydicom


def visualize_contours(raw_path, patient_df):
    """
    Given a datafraem generated from the get_patient_df() function above,
    plots the images with contours overlayed

    :param patient_df: patient summary DatFrame from get_patient_df() function
    return: None
    """
    nrows = int(math.ceil(len(patient_df)/3))
    ncols = 3

    fig, axs = plt.subplots(nrows, ncols, figsize=(20, nrows*7))
    for ax, row in zip(axs.reshape(-1), patient_df.iterrows()):
        path = row[1]['path']
        pos = row[1]['position']
        rois = row[1]['ROIs']

        ds = pydicom.dcmread(path)
        ax.imshow(
            ds.pixel_array,
            vmin=0, vmax=2048,
            cmap='gray',
            extent=[0, 512, 512, 0]
        )
        for i, roi in enumerate(rois):
            xs, ys = list(zip(*roi))
            ax.scatter(xs, ys, s=.75- .1*(i+1), alpha=1- .05*(i+1), label=f'radiologist {i+1}')
        ax.set_title(f"Z-Position: {pos[-1]}")
        ax.legend(markerscale=6)
    return