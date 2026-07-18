import torch
import torch.nn as nn
from torch import Tensor

from src.structure_module.ipa import InvariantPointAttention
from src.geometry.geometry import compute_all_atom_coordinates, assemble_4x4_transform, quat_to_3x3_rotation
from src.geometry.geometry import residue_constants


class StructureModuleTransition(nn.Module):
    """
    Implements the transition in the Structure Module (lines 8 and 9 from Algorithm 20).
    """

    def __init__(self, c_s: int):
        """
        Initializes StructureModuleTransition.

        Args:
            c_s (int): Number of channels for the single representation.
        """
        super().__init__()
        self.c_s = c_s

        self.linear_1 = nn.Linear(c_s, c_s)
        self.linear_2 = nn.Linear(c_s, c_s)
        self.linear_3 = nn.Linear(c_s, c_s)
        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm = nn.LayerNorm(c_s)
        self.relu = nn.ReLU()

    def forward(self, s: Tensor) -> Tensor:
        """
        Implements the forward pass for the transition as
        s -> linear -> relu -> linear -> relu -> linear + s -> layer_norm

        Args:
            s (torch.tensor): Single representation of shape (*, N_res, c_s).

        Returns:
            torch.tensor: Output single representation of shape (*, N_res, c_s).
        """

        s = s + self.linear_3(self.relu(self.linear_2(self.relu(self.linear_1(s)))))
        s = self.layer_norm(self.dropout(s))

        return s


class BackboneUpdate(nn.Module):
    """
    Implements the backbone update, according to Algorithm 23.
    """

    def __init__(self, c_s: int):
        """
        Args:
            c_s (int): Number of channels for the single representation.
        """
        super().__init__()
        self.linear = nn.Linear(c_s, 6)

    def forward(self, s: Tensor) -> Tensor:
        """
        Computes the forward pass for Algorithm 23.

        Args:
            s (torch.tensor): Single representation of shape (*, N_res, c_s).

        Returns:
            torch.tensor: Backbone transforms of shape (*, N_res, 4, 4).
        """

        group = self.linear(s)
        quat = torch.cat((torch.ones(group.shape[:-1]+(1,), device=group.device), group[..., :3]), dim=-1)
        quat = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
        t = group[..., 3:]

        # Explicit formula from Algorithm 22:
        # a, b, c, d = torch.unbind(quat, dim=-1)
        # R = [
        #     [a**2+b**2-c**2-d**2, 2*b*c-2*a*d, 2*b*d+2*a*c],
        #     [2*b*c+2*a*d, a**2-b**2+c**2-d**2, 2*c*d-2*a*b],
        #     [2*b*d-2*a*c, 2*c*d+2*a*b, a**2-b**2-c**2+d**2]
        # ]
        # R = [torch.stack(vals, dim=-1) for vals in R]
        # R = torch.stack(R, dim=-2)

        R = quat_to_3x3_rotation(quat)
        T = assemble_4x4_transform(R,  t)

        return T


class AngleResNetLayer(nn.Module):
    """
    Implements a layer of the AngleResNet for the Structure Module, 
    which is line 12 or line 13 from Algorithm 20.
    """
    
    def __init__(self, c: int):
        """
        Initializes AngleResNetLayer.

        Args:
            c (int): Embedding dimension for the AngleResNet.
        """
        super().__init__()

        self.linear_1 = nn.Linear(c, c)
        self.linear_2 = nn.Linear(c, c)
        self.relu = nn.ReLU()

    def forward(self, a: Tensor) -> Tensor:
        """
        Computes the forward pass as 
        a -> relu -> linear -> relu -> linear + a

        Args:
            a (torch.tensor): Embedding of shape (*, N_res, c).

        Returns:
            torch.tensor: Output embedding of shape (*, N_res, c).
        """

        a = a + self.linear_2(self.relu(self.linear_1(self.relu(a))))
        
        return a


class AngleResNet(nn.Module):
    """
    Implements the AngleResNet from the Structure Module (lines 11-14 in Algorithm 20).
    """
    
    def __init__(self, c_s: int, c: int, n_torsion_angles: int=7):
        """
        Initializes the AngleResNet.

        Args:
            c_s (int): Number of channels for the single representation.
            c (int): Embedding dimension of the AngleResNet.
            n_torsion_angles (int, optional): Number of torsion angles to be predicted. Defaults to 7.
        """
        super().__init__()
        self.n_torsion_angles = n_torsion_angles

        self.linear_in = nn.Linear(c_s, c)
        self.linear_initial = nn.Linear(c_s, c)
        self.layers = nn.ModuleList([AngleResNetLayer(c) for _ in range(2)])
        self.linear_out = nn.Linear(c, 2*n_torsion_angles)
        self.relu = nn.ReLU()

    def forward(self, s: Tensor, s_initial: Tensor) -> Tensor:
        """
        Implements the forward pass through the AngleResNet according to Algorithm 20.
        In contrast to the supplement, s and s_initial are passed through a ReLU
        function before the first linear layers.

        Args:
            s (torch.tensor): Single representation of shape (*, N_res, c_s).
            s_initial (torch.tensor): Initial single representation of shape (*, N_res, c_s).

        Returns:
            torch.tensor: Torsion angles of shape (*, N_res, 2*n_torsion_angles).
        """

        s = self.relu(s)
        s_initial = self.relu(s_initial)
        a = self.linear_in(s) + self.linear_initial(s_initial)
        for layer in self.layers:
            a = layer(a)
        alpha = self.linear_out(self.relu(a))
        alpha_shape = alpha.shape[:-1] + (self.n_torsion_angles, 2)
        alpha = alpha.view(alpha_shape)

        return alpha


class StructureModule(nn.Module):
    """
    Implements the Structure Module according to Algorithm 20.
    """

    def __init__(self, c_s: int, c_z: int, n_layer: int=8, c: int=128):
        """
        Args:
            c_s (int): Number of channels for the single representation.
            c_z (int): Number of channels for the pair representation.
            n_layer (int, optional): Number of layers for the whole module. Defaults to 8.
            c (int, optional): Embedding dimension for the AngleResNet. Defaults to 128.
        """
        super().__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.n_layer = n_layer

        self.layer_norm_s = nn.LayerNorm(c_s)
        self.layer_norm_z = nn.LayerNorm(c_z)
        self.linear_in = nn.Linear(c_s, c_s)

        self.layer_norm_ipa = nn.LayerNorm(c_s)
        self.dropout_s = nn.Dropout(0.1)
        self.ipa = InvariantPointAttention(c_s, c_z)
        self.transition = StructureModuleTransition(c_s)
        self.bb_update = BackboneUpdate(c_s)
        self.angle_resnet = AngleResNet(c_s, c)

    def process_outputs(self, T: Tensor, alpha: Tensor, F: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Computes the final atom positions, the atom mask and the pseudo beta positions
        from the backbone transforms, torsion angles and amino acid labels.

        Args:
            T (torch.tensor): Backbone transforms of shape (*, N_res, 4, 4). Units 
                are measured in nanometers (this affects only the translation). 
            alpha (torch.tensor): Torsion angles of shape (*, N_res, n_torsion_angles, 2).
            F (torch.tensor): Labels for the amino acids of shape (*, N_res). Labels are encoded
                as 0 -> Alanine, 1 -> Arginine, ..., 19 -> Valine. 

        Returns:
            tuple: A tuple consisting of the following values:
                - final_positions: Tensor of shape (*, N_res, 37, 3). The 3D positions of 
                    all atoms, measured in Angstrom.
                - position_mask: Boolean tensor of shape (*, N_res, 37). Masks the side-chain 
                    atoms that aren't present in the amino acids.
                - pseudo_beta_positions: Tensor of shape (*, N_res, 3). 3D positions in Angstrom
                    of C-beta (for all amino acids except glycine) or C-alpha (for glycine).
        """

        scaled_T = T.clone()    
        scaled_T[..., :3, 3] *= 10
        final_positions, position_mask = compute_all_atom_coordinates(scaled_T, alpha, F)

        c_beta_ind = residue_constants.atom_types.index('CB')
        c_alpha_ind = residue_constants.atom_types.index('CA')
        glycine_ind = residue_constants.restypes.index('G')
        pseudo_beta_positions = final_positions[..., c_beta_ind, :]
        alpha_positions = final_positions[..., c_alpha_ind, :]
        pseudo_beta_positions[F==glycine_ind] = alpha_positions[F==glycine_ind]

        return final_positions, position_mask, pseudo_beta_positions

    def forward(self, s: Tensor, z: Tensor, F: Tensor) -> dict:
        """
        Forward pass for the Structure Module.

        Args:
            s (torch.tensor): Single representation of shape (*, N_res, c_s).
            z (torch.tensor): Pair representation of shape (*, N_res, c_z).
            F (torch.tensor): Labels for the amino acids of shape (*, N_res).

        Returns:
            dict: Output dictionary with the following entries:
                - angles: Torsion angles of shape (*, N_layers, N_res, n_torsion_angles, 2). 
                - frames: Backbone frames of shape (*, N_layers, N_res, 4, 4).  
                - final_positions: Heavy atom positions in Angstrom of shape (*, N_res, 37, 3).
                - position_mask: Boolean tensor of shape (*, N_res, 37), masking atoms that are
                    not present in the amino acids.
                - pseudo_beta_positions: C-beta-positions (non-glycine) or C-alpha-positions
                    (glycine) for each residue, of shape (*, N_res, 3).
        """
        N_res = z.shape[-2]
        batch_dim = s.shape[:-2]
        outputs = {'angles': [], 'frames': []}
        device = s.device
        dtype = s.dtype

        s_initial = self.layer_norm_s(s)
        z = self.layer_norm_z(z)
        s = self.linear_in(s_initial)
        T = torch.eye(4, device=device, dtype=dtype).broadcast_to(batch_dim+(N_res, 4, 4))

        for _ in range(self.n_layer):
            s += self.ipa(s, z, T)
            s = self.layer_norm_ipa(self.dropout_s(s))
            s = self.transition(s)
            T = T @ self.bb_update(s)

            alpha = self.angle_resnet(s, s_initial)
            outputs['angles'].append(alpha)
            outputs['frames'].append(T)

        outputs['angles'] = torch.stack(outputs['angles'], dim=-4)
        outputs['frames'] = torch.stack(outputs['frames'], dim=-4)

        final_positions, position_mask, pseudo_beta_positions = self.process_outputs(T, alpha, F)
        outputs['final_positions'] = final_positions
        outputs['position_mask'] = position_mask
        outputs['pseudo_beta_positions'] = pseudo_beta_positions

        return outputs