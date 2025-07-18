import numpy as np

def enhance_contrast(image, clip_percentile=(2, 98)):
    """Enhance contrast similar to MATLAB's imadjusta."""
    img = image.astype(np.float32)
    img -= np.min(img)
    img /= np.max(img)

    low, high = np.percentile(img, clip_percentile)
    img = np.clip((img - low) / (high - low), 0, 1)
    return img