import numpy as np

from modules.utils import is_nan_inf, apply_exclusion_zone


def top_k_motifs(matrix_profile: dict, top_k: int = 3) -> dict:
    """
    Find the top-k motifs based on matrix profile

    Parameters
    ---------
    matrix_profile: the matrix profile structure
    top_k : number of motifs

    Returns
    --------
    motifs: top-k motifs (left and right indices and distances)
    """

    motifs_idx = []
    motifs_dist = []

    # INSERT YOUR CODE
    matrix_profile_array = matrix_profile['mp'].copy()
    motif_indices = np.array(matrix_profile['mpi']).copy().astype(int)
    exclusion_zone = matrix_profile['excl_zone']

    for _ in range(top_k):
        if is_nan_inf(matrix_profile_array):
            break

        min_idx = np.argmin(matrix_profile_array)
        min_distance = matrix_profile_array[min_idx]

        motifs_dist.append(min_distance)
        motifs_idx.append((min_idx, motif_indices[min_idx]))

        apply_exclusion_zone(matrix_profile_array, min_idx, exclusion_zone, np.inf)
        apply_exclusion_zone(matrix_profile_array, motif_indices[min_idx], exclusion_zone, np.inf)


    '''
    mp = matrix_profile['mp']
    mpi = matrix_profile['mpi']
    m = matrix_profile['m']
    excl_zone = matrix_profile['excl_zone']

    # Sort indices by matrix profile values
    sorted_indices = np.argsort(mp)

    # Apply exclusion zone
    filtered_indices = []
    for idx in sorted_indices:
        if all(abs(idx - other_idx) > excl_zone for other_idx in filtered_indices):
            filtered_indices.append(idx)

    # Find top-k motifs
    for idx in filtered_indices[:top_k]:
        left_idx = idx
        right_idx = mpi[idx]
        motifs_idx.append((left_idx, right_idx))
        motifs_dist.append(mp[idx])

    return {
        "indices": motifs_idx,
        "distances": motifs_dist
    }
    '''
    '''
    profile = matrix_profile['matrix_profile'] 
    indices = matrix_profile['indices'] 
    
    for _ in range(top_k): 
      min_idx = np.argmin(profile) 
      min_dist = profile[min_idx] 
      motifs_idx.append((indices[min_idx], min_idx)) 
      motifs_dist.append(min_dist) 
      apply_exclusion_zone(profile, min_idx)
    '''
    '''
    for i in range(top_k): 
      
      motif_idx = np.argmin(matrix_profile['profile'])
      motif_dist = matrix_profile['profile'][motif_idx] 
      
      motifs_idx.append((motif_idx, matrix_profile['index'][motif_idx])) 
      motifs_dist.append(motif_dist) 
      
      apply_exclusion_zone(matrix_profile, motif_idx)
    '''

    return {
        "indices" : motifs_idx,
        "distances" : motifs_dist
        }
