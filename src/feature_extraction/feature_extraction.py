import re
import torch
import torch.nn as nn
from torch import Tensor

_restypes = ["A","R","N","D","C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",]
_restypes_with_x = _restypes + ["X"]
_restypes_with_x_and_gap = _restypes_with_x + ["-"]

restype_order_with_x = {res: i for i, res in enumerate(_restypes_with_x)}
restype_order_with_x_and_gap = {res: i for i, res in enumerate(_restypes_with_x_and_gap)}

def load_a3m_file(file_name: str) -> list[str]:
    """
    Loads an A3M (multiple sequence alignment) file and extracts the raw amino acid sequences.

    Args:
        file_name (str): Path to the A3M file.

    Returns:
        A list of strings where each string represents an individual protein sequence from the input MSA.
    """
    
    with open(file_name, 'r') as f:
        lines = f.readlines()

    description_line_indices = [i for i, l in enumerate(lines) if l.startswith('>')]
    seqs = [lines[i+1].strip() for i in description_line_indices]

    return seqs

def onehot_encode_aa_type(seq: list[str], include_gap_token: bool=False) -> Tensor:
    """
    Converts a protein sequence into one-hot encoding. X represents an unknown amino acid.

    Args:
        seq (list[str]): A string representing the amino acid sequencce using single-letter codes.
        include_gap_token (bool, optional): If True, includes an extra token ('-') in the encoding to
            represent gaps.

    Returns:
        A Tensor of shape (N_res, 22) if `include_gap_token` is True,
        or shape (N_res, 21) otherwise. Here, N_res is the length of the sequence.
    """
    restype_order = restype_order_with_x if not include_gap_token else restype_order_with_x_and_gap
    
    sequence_idx = Tensor([restype_order[a] for a in seq]).long()
    encoding = nn.functional.one_hot(sequence_idx, num_classes=len(restype_order))

    return encoding

def initial_data_from_seqs(seqs: list[str]) -> dict[Tensor, Tensor, Tensor]:
    """
    Processes raw sequences from an A3M file to extract initial feature representations.

    Args:
        seqs: A list of amino acid sequences loaded from the A3M file.
            Sequences are represented with single-letter amino acid codes.
            Lowercase letters represent deletions.

    Returns:
        A dictionary containing:
            * msa_aatype: A Tensor of one-hot encoded amino acid sequences
                of shape (N_seq, N_res, 22), where N_seq is the number of unique
                sequences (with deletions removed) and N_res is the length of 
                sequences. The dimension 22 corresponts to the 20 amino acids, an
                unknown amino acid token, and a gap token.
            * msa_deletion_count: A Tensor of shape (N_seq, N_res) where each
                element represents the number of deletions occuring before the
                corresponding residue in the MSA.
            * aa_distribution: A Tensor of shape (N_res, 22) containing the
                overall amino acid distribution at each residue position across
                the MSA.
    """

    deletion_count_matrix = []
    unique_seqs = []
    for seq in seqs:
        deletion_count_list = []
        deletion_counter = 0
        for letter in seq:
            if letter.islower():
                deletion_counter += 1
            else:
                deletion_count_list.append(deletion_counter)
                deletion_counter = 0
        seq_without_deletion = re.sub('[a-z]', '', seq)

        if seq_without_deletion in unique_seqs:
            continue

        unique_seqs.append(seq_without_deletion)
        deletion_count_matrix.append(deletion_count_list)

    unique_seqs = torch.stack([onehot_encode_aa_type(seq, include_gap_token=True) for seq in unique_seqs], dim=0).float()
    deletion_count_matrix = torch.tensor(deletion_count_matrix).float()
    aa_distribution = unique_seqs.float().mean(dim=0)

    return { 'msa_aatype': unique_seqs, 'msa_deletion_count': deletion_count_matrix, 'aa_distribution': aa_distribution}

def select_cluster_centers(features: dict, max_msa_clusters: int=512, seed: int=42) -> dict:
    """
    Selects representative sequences as cluster centers from the MSA to
    reduce redundancy.

    Args:
        features (dict): A dictionary containing feature representations of the MSA.
        max_msa_clusters (int): The maximum number of cluster centers to select.
        seed (int): An optional integer seed for the random number generator.

    Modifies:
        The 'features' dictionary in-place by:
            * Updating the 'msa_aatype' and 'msa_deletion_count' features to contain
                data for the cluster centers only.
            * Adding 'extra_msa_aatype' and 'extra_msa_deletion_count' features
                to hold the data for the remaining (non-center) sequences.
    """

    N_seq, N_res = features['msa_aatype'].shape[:2]
    MSA_FEATURE_NAMES = ['msa_aatype', 'msa_deletion_count']
    max_msa_clusters = min(max_msa_clusters, N_seq)

    gen = None
    if seed is not None:
        gen = torch.Generator(features['msa_aatype'].device)
        gen.manual_seed(seed)

    shuffled = torch.randperm(N_seq - 1, generator=gen) + 1
    shuffled = torch.cat((torch.tensor([0]), shuffled), dim=0)

    for key in MSA_FEATURE_NAMES:
        extra_key = f'extra_{key}'
        value = features[key]
        features[extra_key] = value[shuffled[max_msa_clusters:]]
        features[key] = value[shuffled[:max_msa_clusters]]

    return features

def mask_cluster_centers(features: dict, mask_probability: float=0.15, seed: int=42) -> dict:
    """
    Introduces random masking in the cluster center sequences for data augmentation.

    This function modifies the 'msa_aatype' feature within the 'features' dictionary to improve
    model robustness in the presence of noisy or missing input data. Masking is inspired by
    the AlphaFold architecture.

    Args:
        features (dict): A dictionary containing feature representations of the MSA. It is assumed
            that cluster centers have already been selected.
        mask_probability (float): The probability of masking out an individual amino acid
            in a cluster center sequence.
        seed (int): An optional integer seed for the random number generator.

    Modifies:
        The 'features' dictionary in-place by:
            * Updating the 'msa_aatype' feature with masked-out tokens as well as possible 
              replacements based on defined probabilities. 
            * Creating a copy of the original 'msa_aatype' feature with the key 'true_msa_aatype'. 
    """

    N_clust, N_res = features['msa_aatype'].shape[:2]
    N_aa_categories = 23 # 20 aa, 1 unknown, 1 gap, 1 masked
    odds = {
        'uniform_replacement': 0.1,
        'replacement_from_distribution': 0.1,
        'no_replacement': 0.1,
        'masked_out': 0.7
    }
    gen = None
    if seed is not None:
        gen = torch.Generator(features['msa_aatype'].device)
        gen.manual_seed(seed)
        torch.manual_seed(seed)

    # (22, )
    uniform_replacement = torch.tensor([1/20]*20+[0, 0]) * odds['uniform_replacement']
    # (N_res, 22)
    replacement_from_distribution = features['aa_distribution'] * odds['replacement_from_distribution']
    # (N_clust, N_res, 22)
    no_replacement = features['msa_aatype'] * odds['no_replacement']
    # (N_clust, N_res, 1)
    masked_out = torch.ones((N_clust, N_res, 1)) * odds['masked_out']

    uniform_replacement = uniform_replacement[None, None, ...].broadcast_to(no_replacement.shape)
    replacement_from_distribution = replacement_from_distribution[None, ...].broadcast_to(no_replacement.shape)

    categories_without_mask_token = uniform_replacement + replacement_from_distribution + no_replacement
    categories_with_mask_token = torch.cat((categories_without_mask_token, masked_out), dim=-1)
    categories_with_mask_token = categories_with_mask_token.reshape(-1, N_aa_categories)
    
    replace_with = torch.distributions.Categorical(categories_with_mask_token).sample()
    replace_with = nn.functional.one_hot(replace_with, num_classes=N_aa_categories)
    replace_with = replace_with.reshape(N_clust, N_res, N_aa_categories)
    replace_with = replace_with.float()

    replace_mask = torch.rand((N_clust, N_res), generator=gen) < mask_probability

    features['true_msa_aatype'] = features['msa_aatype'].clone()
    aatype_padding = torch.zeros((N_clust, N_res, 1))
    features['msa_aatype'] = torch.cat((features['msa_aatype'], aatype_padding), dim=-1)
    features['msa_aatype'][replace_mask] = replace_with[replace_mask]

    return features

def cluster_assignment(features: dict) -> dict:
    """
    Assigns sequences in the extra MSA to their closest cluster centers based on Hamming Distance.

    Args:
        features (dict): A dictionary containing feature representations of the MSA.
            It is assumed that cluster centers have already been selected.

    Returns:
        The updated 'features' dictionary with the following additions:
            * cluster_assignment: A tensor of shape (N_extra,) containing the indices
                of the assigned cluster centers for each eactra sequence.
            * cluster_assignment_counts: A tensor of shape (N_clust,) where each element indicates
                the number of extra sequeances assigned to a cluster center
                (excluding the cluster center itself.)
    """

    N_clust, N_res = features['msa_aatype'].shape[:2]
    N_extra = features['extra_msa_aatype'].shape[0]

    msa_aatype = features['msa_aatype'][...,:21]
    extra_msa_aatype = features['extra_msa_aatype'][...,:21]
    agreement = torch.einsum('cra,era->ce', msa_aatype, extra_msa_aatype)
    assignment = torch.argmax(agreement,dim=0)
    features['cluster_assignment'] = assignment

    assignment_counts = torch.bincount(assignment, minlength=N_clust)
    features['cluster_assignment_counts'] = assignment_counts

    return features

def cluster_average(feature: Tensor, extra_feature: Tensor, cluster_assignment: Tensor, cluster_assignment_count: Tensor) -> Tensor:
    """
    Calculates the average representation of each cluster center by aggregating features
    from the assigned extra sequences.

    Args:
        feature (tensor): A tensor containing feature representations for the cluster centers.
            Shape: (N_clust, N_res, *)
        extra_feature (tensor): A tensor containing feature representations for extra sequence.
            Shape: (N_extra, N_res, *). The trailing dimensions (*) must be smaller
            or equal to those of the 'feature' tensor.
        cluster_assignment (tensor): A tensor indicating the cluster assignment of each extra sequence.
            Shape: (N_extra, )
        cluster_assignment_count (tensor): A tensor containing the number of extra sequences assigned
            to each cluster center.
            Shape: (N_clust, )

    Returns:
        A tensor containing the average feature representation for each cluster.
        Shape: (N_clust, N_res, *)
    """

    N_clust, N_res = feature.shape[:2]
    N_extra = extra_feature.shape[0]

    unsqueezed_extra_shape = (N_extra,) + (1,) * (extra_feature.dim()-1)
    unsqueezed_cluster_shape = (N_clust,) + (1,) * (feature.dim()-1)
    cluster_assignment = cluster_assignment.view(unsqueezed_extra_shape).broadcast_to(extra_feature.shape)
    cluster_assignment_count = cluster_assignment_count.view(unsqueezed_cluster_shape).broadcast_to(feature.shape)

    cluster_sum = torch.scatter_add(feature, dim=0, index=cluster_assignment, src=extra_feature)
    cluster_average = cluster_sum / (cluster_assignment_count + 1)

    return cluster_average

def summarize_clusters(features: dict) -> dict:
    """
    Calculates cluster summaries by applying cluster averaging to the MSA amino acid
    representations and deletion counts.

    Args:
        features (dict): A dictionary containing feature representations of the MSA.

    Modifies:
        The 'features' dictionary in-place by adding the following:
            * cluster_deletion_mean: Average deletion counts for each cluster center,
                scaled for numerical stability.
            * cluster_profile: Average amino acid representation for each cluster center.
    """

    N_clust, N_res = features['msa_aatype'].shape[:2]
    N_extra = features['extra_msa_aatype'].shape[0]

    cluster_deletion_mean = cluster_average(
        features['msa_deletion_count'],
        features['extra_msa_deletion_count'],
        features['cluster_assignment'],
        features['cluster_assignment_counts']
    )

    cluster_deletion_mean = 2/torch.pi * torch.arctan(cluster_deletion_mean/3)
    extra_msa_aatype = features['extra_msa_aatype']
    pad = torch.zeros(extra_msa_aatype.shape[:-1]+(1,), dtype=extra_msa_aatype.dtype, device=extra_msa_aatype.device)
    extra_msa_aatype_padded = torch.cat((extra_msa_aatype, pad), dim=-1)

    cluster_profile = cluster_average(
        features['msa_aatype'],
        extra_msa_aatype_padded,
        features['cluster_assignment'],
        features['cluster_assignment_counts']
    )

    features['cluster_deletion_mean'] = cluster_deletion_mean
    features['cluster_profile'] = cluster_profile

    return features

def crop_extra_msa(features: dict, max_extra_msa_count: int=5120, seed: int=42) -> dict:
    """
    Reduces the number of extra sequences in the MSA to a fixed size for the computational efficiency.

    Args:
        features (dict): A dictionary containing feature representations of the MSA.
        max_extra_msa_count (int): The maximum number of extra sequences to retain.
        seed (int): An optional integer seed for the random number generator.

    Modifies:
        The 'feature' dictionary in-place by cropping the following keys to include
        only the first 'max_extra_msa_count' sequences:
            * Any key starting with 'extra_'
    """

    N_extra = features['extra_msa_aatype'].shape[0]
    gen = None
    if seed is not None:
        gen = torch.Generator(features['extra_msa_aatype'].device)
        gen.manual_seed(seed)

    max_extra_msa_count = min(max_extra_msa_count, N_extra)

    sequence_idx = torch.randperm(N_extra, generator=gen)
    sequence_idx = sequence_idx[:max_extra_msa_count]

    for key in features.keys():
        if key.startswith("extra_"):
            features[key] = features[key][sequence_idx]

    return features

def calculate_msa_feat(features: dict) -> Tensor:
    """
    Prepares the final MSA feature representation for protein structure prediction.

    Args:
        features (dict): A dictionary containing feature representations of the MSA.

    Returns:
        A tensor of shape (N_clust, N_res, 49) representing the final MSA features,
        formed by concatenating processed cluster information and deletion-related values. 
    """
    
    N_clust, N_res = features['msa_aatype'].shape[:2]

    cluster_msa = features['msa_aatype']
    msa_deletion_count = features['msa_deletion_count']
    cluster_deletion_mean = features['cluster_deletion_mean'].unsqueeze(-1)
    cluster_profile = features['cluster_profile']

    # cluster_has_deletion = (features['msa_deletion_count'] > 0).float().unsqueeze(-1)
    cluster_has_deletion = msa_deletion_count.masked_fill(msa_deletion_count != 0, 1).unsqueeze(-1)
    
    cluster_deletion_value = 2/torch.pi * torch.arctan(msa_deletion_count / 3)
    cluster_deletion_value = cluster_deletion_value.unsqueeze(-1)

    msa_feat = torch.cat((cluster_msa, cluster_has_deletion, cluster_deletion_value, cluster_profile, cluster_deletion_mean), dim=-1)

    return msa_feat

def calculate_extra_msa_feat(features: dict) -> Tensor:
    """
    Prepares the extra MSA feature representation for protein structure prediction. 
    This function is similar to 'calculate_msa_feat' but operates on  extra MSA sequences
    and includes padding of extra_msa_aatype to match the shape of msa_aatype. 

    Args:
        features (dict): A dictionary containing feature representations of the MSA.

    Returns:
        A tensor of shape (N_extra, N_res, 25) representing the final extra MSA features.
    """

    N_extra, N_res = features['extra_msa_aatype'].shape[:2]

    extra_msa_aatype = features['extra_msa_aatype']
    extra_msa_deletion_count = features['extra_msa_deletion_count']

    extra_msa_has_deletion = extra_msa_deletion_count.masked_fill(extra_msa_deletion_count != 0, 1).unsqueeze(-1)
    extra_msa_deletion_value = 2/torch.pi * torch.arctan(extra_msa_deletion_count / 3)
    extra_msa_deletion_value = extra_msa_deletion_value.unsqueeze(-1)

    padding = torch.zeros((N_extra, N_res, 1))
    extra_msa = torch.cat([extra_msa_aatype, padding], dim=-1)

    extra_msa_feat = torch.cat([extra_msa, extra_msa_has_deletion, extra_msa_deletion_value], dim=-1)

    return extra_msa_feat

def create_features_from_a3m(file_name: str, seed: int=42) -> dict:
    """
    Creates feature representations for an MSA from its A3M file.

    This function orchestrates a sequence of transformations on the raw MSA sequences to 
    produce features suitable for protein structure prediction.

    Args:
        file_name (str): Path to the A3M file containing the MSA sequences.

    Returns:
        A dictionary containing the following feature representations for the MSA:
           * msa_feat: A tensor containing the final MSA feature representation.
           * extra_msa_feat: A tensor containing the final extra MSA feature representation.
           * target_feat: A tensor containing a one-hot encoded representation of the 
                          target protein sequence (excluding gaps and masked tokens).
           * residue_index: A tensor containing the residue indices (0, 1, ..., N_res-1). 
    """

    if seed is not None:
        select_clusters_seed = seed
        mask_clusters_seed = seed+1
        crop_extra_seed = seed+2

    seqs = load_a3m_file(file_name)
    features = initial_data_from_seqs(seqs)

    transforms = [
        lambda x: select_cluster_centers(x, seed=select_clusters_seed),
        lambda x: mask_cluster_centers(x, seed=mask_clusters_seed),
        cluster_assignment,
        summarize_clusters,
        lambda x: crop_extra_msa(x, seed=crop_extra_seed)
    ]

    for transform in transforms:
        features = transform(features)

    msa_feat = calculate_msa_feat(features)
    extra_msa_feat = calculate_extra_msa_feat(features)

    target_feat = onehot_encode_aa_type(seqs[0], include_gap_token=False).float()
    residue_index = torch.arange(len(seqs[0]))

    return {
        'msa_feat': msa_feat,
        'extra_msa_feat': extra_msa_feat,
        'target_feat': target_feat,
        'residue_index': residue_index
    }