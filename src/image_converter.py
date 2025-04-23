import bioformats
import javabridge
import numpy as np
import tifffile
import os
import imageio.v2 as imageio
from skimage import exposure

def auto_color_balance(image):
    """Performs automatic contrast stretching on an image."""
    p2, p98 = np.percentile(image, (2, 98))  # Contrast stretching
    return exposure.rescale_intensity(image, in_range=(p2, p98))

def convert_lsm_to_tif(lsm_path, output_folder):
    # Start Java VM for bioformats
    javabridge.start_vm(class_path=bioformats.JARS)

    try:
        # Read the image using Bio-Formats
        with bioformats.ImageReader(lsm_path) as reader:
            num_channels = reader.rdr.getSizeC()
            image_shape = (reader.rdr.getSizeY(), reader.rdr.getSizeX(), num_channels)
            
            # Read all channels and normalize
            merged_image = np.zeros(image_shape, dtype=np.float32)
            for c in range(num_channels):
                channel_image = reader.read(series=0, c=c, rescale=True)
                channel_image = auto_color_balance(channel_image)  # Adjust color
                merged_image[:, :, c] = channel_image  # Merge channels

            # Normalize to 8-bit RGB
            merged_image = (255 * (merged_image - merged_image.min()) / (merged_image.max() - merged_image.min())).astype(np.uint8)
        
        # Save as TIFF
        output_path = os.path.join(output_folder, os.path.basename(lsm_path).replace('.lsm', '.tif'))
        imageio.imwrite(output_path, merged_image)

        print(f"Saved: {output_path}")
    
    finally:
        # Stop the Java VM
        javabridge.kill_vm()

# Example usage
input_folder = "path/to/lsm_files"
output_folder = "path/to/output"

for file in os.listdir(input_folder):
    if file.endswith(".lsm"):
        convert_lsm_to_tif(os.path.join(input_folder, file), output_folder)
