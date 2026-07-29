"""
Pore Detection and Material Segmentation Module

This module provides comprehensive functions for detecting pores in 3D volumes using various 
thresholding techniques. It includes functionality for Sauvola and Otsu thresholding, material 
masking, and pore extraction from X-ray CT scan data. The module supports both parallel and 
sequential processing to optimize performance based on available system memory.

Key Features:
- Adaptive Sauvola thresholding for variable background conditions
- Otsu thresholding for binary segmentation
- Material mask generation with void filling
- Pore cleaning and filtering based on size and dimensional constraints
- Memory-aware processing (automatic fallback to sequential processing)
- 3D morphological operations for noise reduction

Typical Workflow:
1. Load 3D X-ray CT volume
2. Apply onlypores() to extract pore structures
3. Optionally apply clean_pores() for additional filtering
4. Analyze resulting binary pore mask

Author: [Author name]
Date: [Date]
Version: 1.0
"""

import numpy as np
from skimage import measure, exposure
from skimage import filters
import fill_voids
from skimage.morphology import remove_small_objects
from skimage.measure import label
from tqdm import tqdm
import logging
from scipy.ndimage import binary_erosion as _scipy_erode
from scipy.ndimage import binary_fill_holes as _bfh
from scipy.ndimage import uniform_filter as _uniform_filter
from scipy.ndimage import find_objects as _find_objects
from scipy import ndimage as _ndimage


logger = logging.getLogger(__name__)


def _get_bounding_box(xct, margin=2):
    """Return (min_z, max_z, min_y, max_y, min_x, max_x) of non-zero data with margin."""
    min_z, max_z = -1, -1
    for i in range(xct.shape[0]):
        if np.any(xct[i] > 0):
            min_z = i
            break
    if min_z == -1:
        return None
    for i in range(xct.shape[0] - 1, -1, -1):
        if np.any(xct[i] > 0):
            max_z = i
            break

    projection_mask = np.zeros(xct.shape[1:], dtype=bool)
    for i in tqdm(range(min_z, max_z + 1), desc="Projecting slices", position=1, leave=False):
        if np.any(xct[i] > 0):
            projection_mask |= (xct[i] > 0)

    y_inds, x_inds = np.nonzero(projection_mask)
    min_y, max_y = int(np.min(y_inds)), int(np.max(y_inds))
    min_x, max_x = int(np.min(x_inds)), int(np.max(x_inds))

    min_z = max(0, min_z - margin)
    min_y = max(0, min_y - margin)
    min_x = max(0, min_x - margin)
    max_z = min(xct.shape[0] - 1, max_z + margin)
    max_y = min(xct.shape[1] - 1, max_y + margin)
    max_x = min(xct.shape[2] - 1, max_x + margin)

    return min_z, max_z, min_y, max_y, min_x, max_x


def _erode_mask(mask, px):
    """Erode a 3D boolean mask inward by *px* voxels (cubic structuring element)."""
    if px <= 0:
        return mask
    struct = np.ones((3, 3, 3), dtype=bool)
    return _scipy_erode(mask, structure=struct, iterations=px)


def _erode_mask_2d_perslice(mask, px):
    """
    Erode each Z-slice of a 3D mask independently using a 2D square element.

    This is the correct erosion strategy for the projected material mask: it
    shrinks the sample boundary inward in every cross-section without coupling
    between slices, which is appropriate for elongated/flat samples.
    """
    if px <= 0:
        return mask
    struct = np.ones((3, 3), dtype=bool)
    result = np.empty_like(mask)
    for i in range(mask.shape[0]):
        result[i] = _scipy_erode(mask[i], structure=struct, iterations=px)
    return result


def _project_mask_along_z(mask_3d):
    """
    Build the 'column mask': project a 3D boolean mask along Z with OR,
    then broadcast the 2D footprint back to the original 3D shape.

    Any (y, x) position where material exists at *any* depth is marked True
    for the entire column.  This ensures that defects (low-intensity voids that
    span some Z slices) are considered to be inside the sample even when the
    Otsu binary has them as background at those specific Z positions.
    """
    proj_2d = np.any(mask_3d, axis=0)              # shape (Y, X)
    return np.broadcast_to(
        proj_2d[np.newaxis, :, :], mask_3d.shape
    ).copy()


def equalize_volume_histogram_match(volume, ref_slice):
    """
    Match the histogram of every Z-slice to a chosen reference slice.

    Each slice is remapped so that its cumulative intensity distribution
    matches that of *ref_slice*.  This is useful when one representative
    slice has a good contrast and you want the whole volume to look the same.

    Parameters
    ----------
    volume    : (Z, Y, X) uint8 array
    ref_slice : (Y, X) uint8 array — the reference whose histogram is matched

    Returns
    -------
    numpy.ndarray
        Matched volume with the same shape and dtype as *volume*.
    """
    from skimage.exposure import match_histograms
    matched = np.empty_like(volume)
    for i in range(volume.shape[0]):
        matched[i] = match_histograms(volume[i], ref_slice).astype(volume.dtype)
    return matched


def equalize_volume_per_slice(volume):
    """
    Apply histogram equalization independently to each Z-slice.

    Normalizes each slice so that its intensity distribution spans the full
    range, making the background and foreground peaks sit at consistent
    positions across slices. This corrects for per-slice variations in
    brightness and contrast (e.g. beam-hardening artefacts in CT).

    Parameters:
    ----------
    volume : numpy.ndarray
        3D uint8 array of shape (Z, Y, X).

    Returns:
    -------
    numpy.ndarray
        Equalized volume with the same shape and dtype as input.
    """
    equalized = np.empty_like(volume)
    for i in range(volume.shape[0]):
        equalized[i] = (exposure.equalize_hist(volume[i]) * 255).astype(volume.dtype)
    return equalized


def _sauvola_vectorized(volume: np.ndarray, window_size: int, k: float) -> np.ndarray:
    """Vectorized Sauvola: two uniform_filter passes over the full 3D crop.

    Numerically identical to skimage threshold_sauvola with mode='mirror'
    (verified bit-for-bit agreement on real (128,3200,1280) data).
    Operates along Z and X axes only (kernel shape (w,1,w)) because the
    current caller slices along Y-planes.
    """
    if window_size % 2 == 0:
        window_size += 1
    f = volume.astype(np.float32)
    w = window_size
    mean   = _uniform_filter(f,   size=(w, 1, w), mode='mirror')
    sqmean = _uniform_filter(f*f, size=(w, 1, w), mode='mirror')
    var    = np.clip(sqmean - mean * mean, 0.0, None)
    std    = np.sqrt(var)
    thresh = mean * (1.0 + k * ((std / 128.0) - 1.0))
    return f > thresh


def sauvola_thresholding_concurrent(volume, window_size, k):
    """
    Apply Sauvola thresholding to a 3D volume.

    Thin wrapper around _sauvola_vectorized for API compatibility.
    The vectorized implementation replaces the former joblib-based parallel
    approach, eliminating heap copies in GB10 unified memory.

    Parameters:
    ----------
    volume : numpy.ndarray
        3D numpy array (Z, Y, X).
    window_size : int
        Local window size for Sauvola thresholding (adjusted to odd if needed).
    k : float
        Sensitivity parameter. Typical range: 0.05-0.5.

    Returns:
    -------
    numpy.ndarray
        3D binary array (dtype=bool). True = material, False = pore/feature.
    """
    logger.info("Applying Sauvola thresholding (vectorized)...")
    return _sauvola_vectorized(volume, window_size, k)


def sauvola_thresholding_nonconcurrent(volume, window_size, k):
    """
    Apply Sauvola thresholding to a 3D volume.

    Thin wrapper around _sauvola_vectorized for API compatibility.
    The vectorized implementation supersedes the former sequential
    slice-by-slice approach.

    Parameters:
    ----------
    volume : numpy.ndarray
        3D numpy array (Z, Y, X).
    window_size : int
        Local window size for Sauvola thresholding (adjusted to odd if needed).
    k : float
        Sensitivity parameter.

    Returns:
    -------
    numpy.ndarray
        3D binary array (dtype=bool).
    """
    logger.info("Applying Sauvola thresholding (vectorized)...")
    return _sauvola_vectorized(volume, window_size, k)


def sauvola_thresholding(volume, window_size, k):
    """
    Apply Sauvola thresholding to a 3D volume.

    Vectorized implementation using two scipy.ndimage.uniform_filter passes.
    Numerically equivalent to per-slice skimage.threshold_sauvola with r=128
    and mode='mirror'. No joblib workers — avoids heap-copy overhead in GB10
    unified memory.

    Parameters:
    ----------
    volume : numpy.ndarray
        3D numpy array (Z, Y, X).
    window_size : int
        Local window size for Sauvola thresholding (adjusted to odd if needed).
    k : float
        Sensitivity parameter. Typical range: 0.05-0.5.

    Returns:
    -------
    numpy.ndarray
        3D binary array (dtype=bool). True = material, False = pore/feature.
    """
    logger.info("Applying Sauvola thresholding (vectorized)...")
    return _sauvola_vectorized(volume, window_size, k)

def otsu_thresholding(volume):
    """
    Apply Otsu's automatic thresholding to a 3D volume.
    
    Otsu's method automatically determines an optimal threshold value by minimizing
    intra-class variance (or maximizing inter-class variance) in the image histogram.
    This method works best when the histogram has a clear bimodal distribution.

    Parameters:
    ----------
    volume : numpy.ndarray
        3D numpy array representing the input volume. Non-zero values are used
        for threshold computation.

    Returns:
    -------
    numpy.ndarray
        3D binary numpy array after Otsu thresholding with dtype=bool.
        Values above threshold are True (material), below are False (background/pores).
        
    Notes:
    -----
    - Works best with bimodal intensity distributions
    - Global threshold applied to entire volume
    - Threshold value is printed for debugging/analysis purposes
    """
    # Compute optimal threshold using Otsu's method
    # This analyzes the histogram to find the threshold that best separates two classes
    threshold_value = filters.threshold_otsu(volume)
    
    logger.info(f'Applying Otsu thresholding with value: {threshold_value}')

    # Apply threshold: values above threshold are considered material (True)
    binary = volume > threshold_value

    return binary

def slice_cleaning(img,min_size=2):
    """
    Clean a 2D binary image by removing small objects below a size threshold.
    
    This function removes connected components in a binary image that are smaller
    than the specified minimum size. It is useful for eliminating noise and small
    artifacts from segmented images.

    Parameters:
    ----------
    img : numpy.ndarray
        2D binary numpy array representing the input image.
    min_size : int, optional (default=2)
        Minimum size (in pixels) for connected components to be retained.
    Returns:
    -------
    numpy.ndarray
        2D binary numpy array representing the cleaned image.
    """
    
    # Remove small objects below the specified size threshold
    cleaned_img = remove_small_objects(img > 0, min_size=min_size, connectivity=1)
    
    # Convert back to binary: non-zero labels become True
    cleaned_binary = cleaned_img > 0
    
    return cleaned_binary

def onlypores(xct, frontwall=0, backwall=0, sauvola_radius=30, sauvola_k=0.125,
              min_size_filtering=-1, mask_threshold=-1, equalize=False,
              hist_match_ref=-1, border_erosion=0):
    """
    Extract pores from a 3D X-ray CT volume using advanced thresholding and segmentation.
    
    This is the main function for pore detection. It combines Sauvola adaptive thresholding
    with material mask generation to accurately identify pore structures within a material
    sample. The function includes preprocessing steps like volume cropping, wall exclusion,
    and optional post-processing filtering.

    Parameters:
    ----------
    xct : numpy.ndarray
        3D numpy array representing the X-ray CT volume data.
        Expected shape: (Z, Y, X) where Z is the scan direction.
    frontwall : int, optional (default=0)
        Index of the front wall slice to exclude from analysis.
        Set to 0 to include all slices from the beginning.
    backwall : int, optional (default=0)  
        Index of the back wall slice to exclude from analysis.
        Set to 0 to include all slices to the end.
    sauvola_radius : int, optional (default=30)
        Radius (window size) for Sauvola thresholding. Larger values provide
        smoother thresholding but may miss fine details.
    sauvola_k : float, optional (default=0.125)
        Sensitivity parameter for Sauvola thresholding. Higher values make
        thresholding more conservative. Typical range: 0.05-0.5.
    min_size_filtering : int, optional (default=-1)
        Minimum size threshold for pore filtering. If > 0, applies clean_pores()
        function to remove small artifacts. Set to -1 to disable filtering.

    Returns:
    -------
    tuple of numpy.ndarray
        - onlypores : 3D binary array with detected pores (True = pore, False = material)
        - sample_mask : 3D binary array defining the material boundaries  
        - binary : 3D binary array from initial thresholding step
        
    Notes:
    -----
    - Automatically crops volume to non-zero region for efficiency
    - Applies adaptive thresholding optimized for varying material densities
    - Generates material mask using Otsu thresholding and void filling
    - Optional size-based filtering to remove noise and artifacts
    - Returns None values if input volume contains no data
    
    Example:
    -------
    >>> pores, mask, binary = onlypores(ct_volume, frontwall=10, backwall=200, 
    ...                                sauvola_radius=25, min_size_filtering=50)
    >>> print(f"Detected {np.sum(pores)} pore voxels")
    """
    logger.info('Starting pore detection analysis...')
    
    logger.info('Computing volume bounding box...')
    # Step 1: Find the bounding box WITHOUT loading all indices into memory    
    # --- PART A: Find Z limits (Slice-by-slice check) ---
    min_z, max_z = -1, -1
    
    # Find the first slice with data
    for i in range(xct.shape[0]):
        if np.any(xct[i] > 0):
            min_z = i
            break
            
    if min_z == -1:  # Handle empty volumes
        logger.error("No non-zero values found in the volume")
        return None, None, None
        
    # Find the last slice with data (scanning backwards)
    for i in range(xct.shape[0] - 1, -1, -1):
        if np.any(xct[i] > 0):
            max_z = i
            break

    # --- PART B: Find Y and X limits (Projection) ---
    # We collapse the 3D volume to a 2D mask to find Y/X boundaries.
    # We only check slices between min_z and max_z to save time.
    
    projection_mask = np.zeros(xct.shape[1:], dtype=bool)
    
    logger.info("Projecting volume to determine Y/X boundaries...")
    for i in tqdm(range(min_z, max_z + 1), desc="Projecting slices", position=1, leave=False):
        # Accumulate where data exists. logical_or is memory efficient here.
        # We use '|= ' which is the in-place bitwise OR operator
        if np.any(xct[i] > 0): # Tiny optimization: skip if slice is empty (unlikely inside z-range)
             projection_mask |= (xct[i] > 0)

    # Now we simply find the bounding box of the 2D projection
    # This is fast and uses negligible memory compared to 3D indices
    y_inds, x_inds = np.nonzero(projection_mask)
    
    min_y, max_y = np.min(y_inds), np.max(y_inds)
    min_x, max_x = np.min(x_inds), np.max(x_inds)

    # Step 2: Print results (Logic remains the same, calculation is done above)
    logger.info(f'Original volume shape: {xct.shape}')
    logger.info(f'Data bounding box: Z[{min_z}:{max_z}], Y[{min_y}:{max_y}], X[{min_x}:{max_x}]')
    
    # Add small margin around data for edge effects in filtering operations
    margin = 2  
    min_z = max(0, min_z - margin)
    min_y = max(0, min_y - margin) 
    min_x = max(0, min_x - margin)
    max_z = min(xct.shape[0] - 1, max_z + margin)
    max_y = min(xct.shape[1] - 1, max_y + margin)
    max_x = min(xct.shape[2] - 1, max_x + margin)

    logger.info(f'Cropped volume shape will be: Z[{min_z}:{max_z}], Y[{min_y}:{max_y}], X[{min_x}:{max_x}]')

    # Step 3: Extract the cropped volume for processing efficiency
    cropped_volume = xct[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1]
    logger.info(f'Cropped volume shape: {cropped_volume.shape}')

    # Step 4: Optionally equalise each slice before Sauvola.
    # Material mask always uses raw intensities so mask_threshold stays meaningful.
    if equalize:
        if hist_match_ref >= 0:
            ref_idx = min(hist_match_ref, xct.shape[0] - 1)
            logger.info(f'Applying histogram matching to reference slice {ref_idx}...')
            processing_volume = equalize_volume_histogram_match(
                cropped_volume, xct[ref_idx])
        else:
            logger.info('Applying per-slice histogram equalisation...')
            processing_volume = equalize_volume_per_slice(cropped_volume)
    else:
        processing_volume = cropped_volume

    # Step 5: Apply adaptive thresholding to detect material vs. background
    logger.info('Applying Sauvola adaptive thresholding...')
    binary_cropped = sauvola_thresholding(processing_volume, window_size=sauvola_radius, k=sauvola_k)

    # Step 6: Handle wall exclusions for sample boundaries
    # Set wall regions to True (material) to exclude them from pore detection
    if frontwall > 0:
        logger.info(f'Excluding front wall: slices 0 to {frontwall-1}')
        binary_cropped[:frontwall, :, :] = True
    if backwall > 0:
        logger.info(f'Excluding back wall: slices {backwall} to end')
        binary_cropped[backwall:, :, :] = True
    
    # Step 7: Reconstruct full-size binary volume
    # Create binary volume matching original dimensions
    binary = np.zeros(xct.shape, dtype=bool)
    # Place processed data back into correct spatial location
    binary[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1] = binary_cropped
    
    # Step 8: Generate material mask and build the AND mask
    logger.info('Generating material mask...')
    sample_mask_3d = material_mask(cropped_volume, threshold=mask_threshold)

    # Project the 3D mask along Z (depth) to get a 2D column footprint, then
    # broadcast back.  Any (y, x) column where material exists at any depth is
    # treated as "inside the sample" for the full column.  This prevents defects
    # (low-intensity voids that locally look like background at some Z slices)
    # from being excluded by the AND operation.
    logger.info('Building column mask (Z-projection of material mask)...')
    sample_mask_cropped = _project_mask_along_z(sample_mask_3d)

    # Border erosion: applied per 2D slice (not 3D) on the projected mask.
    # Each cross-section is shrunk inward independently to remove near-edge
    # beam-hardening artefacts without coupling between slices.
    if border_erosion > 0:
        logger.info(f'Per-slice 2D border erosion: {border_erosion} px...')
        sample_mask_cropped = _erode_mask_2d_perslice(sample_mask_cropped, border_erosion)

    # Reconstruct full-size material mask
    sample_mask = np.zeros_like(binary)
    sample_mask[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1] = sample_mask_cropped

    # Step 9: Extract pores by combining thresholding and material mask
    binary_inverted = np.invert(binary)
    onlypores_result = np.logical_and(binary_inverted, sample_mask)
    
    logger.info(f'Initial pore detection complete. Found {np.sum(onlypores_result)} pore voxels.')

    # Step 10: Optional post-processing to remove small artifacts
    if min_size_filtering > 0:
        logger.info(f'Applying pore filtering with minimum size: {min_size_filtering}')
        onlypores_result = clean_pores(onlypores_result, min_size=min_size_filtering)
        logger.info(f'After filtering: {np.sum(onlypores_result)} pore voxels remaining.')

    logger.info('Pore detection analysis complete.')
    return onlypores_result, sample_mask, binary

def material_mask(xct, threshold=-1):
    """
    Generate a material mask for a 3D volume using sequential processing.
    
    This is a memory-efficient, sequential version of material mask generation.
    It applies the same algorithm as the parallel version but processes the entire
    volume at once without chunking. Used when memory is insufficient for parallel
    processing or for debugging purposes.

    Parameters:
    ----------
    xct : numpy.ndarray
        3D numpy array representing the input CT volume.
    threshold : float, optional (default=-1)
        Manual threshold value. If -1, uses Otsu thresholding.

    Returns:
    -------
    numpy.ndarray
        3D binary numpy array representing the material mask where True indicates
        material regions and False indicates background/air.
        
    Notes:
    -----
    - Lower memory footprint than parallel version
    - Processes entire volume as single unit
    - Same algorithm: Otsu + max projection + void filling
    - More stable for very large volumes on memory-limited systems
    """
    logger.info('Computing material mask using 3D processing...')
    
    # Apply global thresholding to entire volume
    if threshold >= 0:
        threshold_value = threshold
    else:
        threshold_value = filters.threshold_otsu(xct)
    
    binary = xct > threshold_value

    # Find connected components in 3D
    labels, n_labels = measure.label(binary, return_num=True)

    if n_labels > 0:
        # Select the component with the highest mean intensity using vectorized
        # np.bincount — avoids the O(n_components) regionprops overhead on large arrays.
        # In CT, material (dense) is brighter than background (air), so the
        # highest-mean-intensity component reliably identifies the sample.
        labels_flat = labels.ravel()
        xct_flat = xct.ravel().astype(np.float32)
        sizes = np.bincount(labels_flat, minlength=n_labels + 1)
        intensity_sums = np.bincount(labels_flat, weights=xct_flat, minlength=n_labels + 1)
        sizes[0] = 0
        intensity_sums[0] = 0  # ignore background label
        mean_int = np.where(sizes > 0, intensity_sums / np.maximum(sizes, 1), 0.0)
        best_label = int(np.argmax(mean_int))

        # Retrieve bounding box without regionprops
        sl = _find_objects(labels)[best_label - 1]
        minz, maxz = sl[0].start, sl[0].stop
        miny, maxy = sl[1].start, sl[1].stop
        minx, maxx = sl[2].start, sl[2].stop

        # Crop binary volume to sample region; keep only the best component
        component_mask = labels[minz:maxz, miny:maxy, minx:maxx] == best_label
        
        # Step 1 — 3D void fill: closes voids that are fully enclosed in 3D.
        # Works well for small isolated spherical pores.
        logger.info('Filling internal voids in 3D...')
        sample_mask_cropped = fill_voids.fill(component_mask, in_place=False)

        # Step 2 — per-slice 2D hole fill: closes any void whose 3D connectivity
        # escapes the volume (surface cracks, delaminations, open porosity, etc.)
        # that fill_voids misses.  In each Z slice the sample outline forms a
        # closed 2D polygon; binary_fill_holes fills everything inside it,
        # independent of 3D topology.  This ensures pore-edge pixels (partial
        # volume, intensity ~50-100) that are spatially inside the sample but
        # below the Otsu threshold end up inside the material mask so the
        # intensity range threshold can correctly select them.
        logger.info('Applying per-slice 2D hole filling to capture surface-connected pores...')
        for i in range(sample_mask_cropped.shape[0]):
            sample_mask_cropped[i] = _bfh(sample_mask_cropped[i])

        # Reconstruct full-size mask
        sample_mask = np.zeros_like(binary)
        sample_mask[minz:maxz, miny:maxy, minx:maxx] = sample_mask_cropped
    else:
        # Fallback: use thresholded volume directly
        logger.warning('No connected components found, using raw threshold')
        sample_mask = binary
    
    logger.info('Material mask generation complete.')
    return sample_mask
    
def clean_pores(onlypores, min_size=8):
    """
    Clean and filter detected pores by removing small artifacts and dimensionally constrained objects.
    
    This function performs post-processing on detected pores to remove noise and artifacts
    that may have been incorrectly identified as pores. It applies two main filtering criteria:
    1. Minimum volume threshold (removes small noise objects)
    2. Minimum dimensional extent (removes flat/thin artifacts)

    Parameters:
    ----------
    onlypores : numpy.ndarray
        3D binary numpy array with detected pores where True represents pore voxels
        and False represents material/background.
    min_size : int, optional (default=8)
        Minimum number of voxels for a connected component to be retained.
        Objects smaller than this threshold are considered noise and removed.

    Returns:
    -------
    numpy.ndarray
        3D binary numpy array with cleaned pores, explicitly cast to bool dtype.
        Only pores meeting both size and dimensional criteria are retained.
        
    Notes:
    -----
    - Uses 3D connectivity (26-neighborhood) for component labeling
    - Removes objects with less than 2 voxels extent in any spatial dimension
    - Dimensional filtering prevents retention of flat/linear artifacts
    - Essential for removing scanning artifacts and noise
    
    Algorithm Steps:
    1. Label connected components in 3D
    2. Remove components smaller than min_size voxels
    3. Analyze bounding box dimensions of remaining components
    4. Retain only components with ≥2 voxels in all axes (X, Y, Z)
    
    Example:
    -------
    >>> # Clean pores with minimum 50 voxels and 2-voxel dimensional extent
    >>> cleaned = clean_pores(detected_pores, min_size=50)
    >>> print(f"Removed {np.sum(detected_pores) - np.sum(cleaned)} noise voxels")
    """
    
    logger.info(f'Cleaning pores with min_size={min_size}...')

    # Step 1: Label connected components using 3D connectivity
    # Connectivity=3 means 26-neighborhood (face, edge, and corner neighbors)
    labeled_pores, initial_components = label(onlypores, connectivity=3, return_num=True)
    logger.info(f'  Initial connected components: {initial_components}')

    # Step 2: Remove small objects based on voxel count
    # This eliminates single-voxel noise and very small artifacts
    labeled_pores = remove_small_objects(labeled_pores, min_size=min_size, connectivity=3)

    # Step 3: Build a boolean LUT (look-up table) for dimensional filtering.
    # find_objects returns bounding-box slices for each label without scanning
    # the entire array for each component — O(n_voxels) single pass instead of
    # O(n_voxels × n_labels) for np.isin.
    n = int(labeled_pores.max())
    keep = np.zeros(n + 1, dtype=bool)
    objs = _find_objects(labeled_pores)
    valid_components = 0
    remaining_after_size = 0
    for lbl_idx, sl in enumerate(objs):
        if sl is None:
            continue
        remaining_after_size += 1
        z0, z1 = sl[0].start, sl[0].stop
        y0, y1 = sl[1].start, sl[1].stop
        x0, x1 = sl[2].start, sl[2].stop
        # Require ≥2 voxels extent in every dimension to filter flat/linear artifacts
        if (z1 - z0) >= 2 and (y1 - y0) >= 2 and (x1 - x0) >= 2:
            keep[lbl_idx + 1] = True
            valid_components += 1

    logger.info(f'  Components after size filtering: {remaining_after_size}')
    logger.info(f'  Valid components after dimensional filtering: {valid_components}')

    # Step 4: Single O(n_voxels) gather via boolean LUT — replaces np.isin
    cleaned_pores = keep[labeled_pores]

    logger.info(f'  Total pore voxels retained: {np.sum(cleaned_pores)}')

    # Ensure output is explicitly bool dtype
    return cleaned_pores.astype(bool)


def onlypores_intensity_threshold(xct, min_intensity, max_intensity,
                                   min_size_filtering=-1, mask_threshold=-1,
                                   equalize=False, hist_match_ref=-1,
                                   border_erosion=0):
    """
    Extract pores using a global intensity-range threshold instead of Sauvola.

    Pixels whose intensity falls in [min_intensity, max_intensity] are classified
    as pores.  The material mask (Otsu + void filling) is applied afterwards so
    that the exterior of the sample (air/background) and – optionally – the near-
    boundary voxels are excluded.

    Parameters
    ----------
    xct : numpy.ndarray
        3-D uint8 CT volume (Z, Y, X).
    min_intensity : int
        Lower bound of the pore intensity range (inclusive).
    max_intensity : int
        Upper bound of the pore intensity range (inclusive).
    min_size_filtering : int
        Minimum pore size in voxels. -1 disables filtering.
    mask_threshold : float
        Manual material-mask threshold. -1 uses Otsu.
    equalize : bool
        Apply per-slice histogram equalisation before thresholding.
    border_erosion : int
        Erode the material mask inward by this many voxels before applying it.
        Use this to exclude low-intensity edge artefacts (beam hardening, partial
        volume effects) from being counted as pores.

    Returns
    -------
    tuple of numpy.ndarray  (pores, sample_mask, pores_before_mask)
    """
    logger.info('Starting intensity-threshold pore detection '
                f'(range [{min_intensity}, {max_intensity}])...')

    bbox = _get_bounding_box(xct)
    if bbox is None:
        logger.error('No non-zero values found in the volume.')
        return None, None, None

    min_z, max_z, min_y, max_y, min_x, max_x = bbox
    logger.info(f'Bounding box: Z[{min_z}:{max_z}] Y[{min_y}:{max_y}] X[{min_x}:{max_x}]')

    cropped_volume = xct[min_z:max_z + 1, min_y:max_y + 1, min_x:max_x + 1]

    if equalize:
        if hist_match_ref >= 0:
            ref_idx = min(hist_match_ref, xct.shape[0] - 1)
            logger.info(f'Applying histogram matching to reference slice {ref_idx}...')
            processing_volume = equalize_volume_histogram_match(
                cropped_volume, xct[ref_idx])
        else:
            logger.info('Applying per-slice histogram equalisation...')
            processing_volume = equalize_volume_per_slice(cropped_volume)
    else:
        processing_volume = cropped_volume

    # Intensity-range pore mask (True = potential pore)
    pores_cropped = (
        (processing_volume.astype(np.int32) >= min_intensity) &
        (processing_volume.astype(np.int32) <= max_intensity)
    )

    # Material mask — always computed on original intensities for stability.
    logger.info('Generating material mask...')
    sample_mask_3d = material_mask(cropped_volume, threshold=mask_threshold)

    # Project along Z → column mask: any (y,x) position where material exists
    # at any depth is considered inside the sample for the whole column.
    logger.info('Building column mask (Z-projection of material mask)...')
    sample_mask_cropped = _project_mask_along_z(sample_mask_3d)

    # Border erosion per 2D slice (not 3D) on the projected mask.
    if border_erosion > 0:
        logger.info(f'Per-slice 2D border erosion: {border_erosion} px...')
        sample_mask_cropped = _erode_mask_2d_perslice(sample_mask_cropped, border_erosion)

    # Keep only pores inside the column mask.
    onlypores_cropped = pores_cropped & sample_mask_cropped

    # Reconstruct full-size arrays
    sample_mask = np.zeros(xct.shape, dtype=bool)
    sample_mask[min_z:max_z + 1, min_y:max_y + 1, min_x:max_x + 1] = sample_mask_cropped

    pores_full = np.zeros(xct.shape, dtype=bool)
    pores_full[min_z:max_z + 1, min_y:max_y + 1, min_x:max_x + 1] = onlypores_cropped

    # "binary" return: pore mask before material-mask is applied (for reference)
    binary_full = np.zeros(xct.shape, dtype=bool)
    binary_full[min_z:max_z + 1, min_y:max_y + 1, min_x:max_x + 1] = pores_cropped

    logger.info(f'Initial pore detection: {np.sum(pores_full):,} voxels.')

    if min_size_filtering > 0:
        logger.info(f'Applying pore size filter (min_size={min_size_filtering})...')
        pores_full = clean_pores(pores_full, min_size=min_size_filtering)
        logger.info(f'After filtering: {np.sum(pores_full):,} voxels.')

    logger.info('Intensity-threshold pore detection complete.')
    return pores_full, sample_mask, binary_full