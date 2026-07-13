from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections.abc import Callable
from typing import Any

import torch
import math

CaseFn = Callable[[], Any]

CASES: dict[str, CaseFn] = {}

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
TESTS_DIR = ROOT / "tests"
ATTENTION_TESTS_DIR = TESTS_DIR / "attention"
FEATURE_EMBEDDING_TESTS_DIR = TESTS_DIR / "feature_embedding"
EVOFORMER_TESTS_DIR = TESTS_DIR / "evoformer"
GEOMETRY_TESTS_DIR = TESTS_DIR / "geometry"
STRUCTURE_MODULE_TESTS_DIR = TESTS_DIR / "structure_module"
MODEL_TESTS_DIR = TESTS_DIR / "model"
FEATURE_EXTRACTION_TESTS_DIR = TESTS_DIR / "feature_extraction"

for path in reversed(
	[
		str(ROOT),
		str(SRC_DIR),
		str(ATTENTION_TESTS_DIR),
		str(FEATURE_EMBEDDING_TESTS_DIR),
		str(EVOFORMER_TESTS_DIR),
		str(GEOMETRY_TESTS_DIR),
		str(STRUCTURE_MODULE_TESTS_DIR),
		str(MODEL_TESTS_DIR),
		str(FEATURE_EXTRACTION_TESTS_DIR),
	],
):
	if path not in sys.path:
		sys.path.insert(0, path)


def _ensure_feature_extraction_controls() -> Path:
	return FEATURE_EXTRACTION_TESTS_DIR


def register_case(name: str) -> Callable[[CaseFn], CaseFn]:
	def decorator(func: CaseFn) -> CaseFn:
		CASES[name] = func
		return func

	return decorator


def run_case(name: str) -> Any:
	try:
		case = CASES[name]
	except KeyError as error:
		available = ", ".join(sorted(CASES)) or "<none>"
		raise SystemExit(f"Unknown case '{name}'. Available cases: {available}") from error

	return case()


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run a named test case.")
	parser.add_argument("case", help="Name of the case to execute")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	result = run_case(args.case)
	if result is not None:
		print(result)


@register_case("attention")
def run_attention_case() -> str:
	from src.attention.mha import MultiHeadAttention
	from attention_checks import (
		N_head,
		attn_dim,
		c,
		c_in,
		test_module_forward,
		test_module_method,
		test_module_shape,
	)

	control_folder = str(ATTENTION_TESTS_DIR)

	test_module_shape(
		MultiHeadAttention(c_in, c, N_head, attn_dim, gated=True),
		"mha_init",
		control_folder,
	)
	test_module_shape(
		MultiHeadAttention(c_in, c, N_head, attn_dim, gated=True, use_bias_for_embeddings=True),
		"mha_bias_init",
		control_folder,
	)

	mha = MultiHeadAttention(c_in, c, N_head, attn_dim, gated=True)
	test_module_method(
		mha,
		"mha_prep_qkv",
		("q", "k", "v"),
		("q_prep", "k_prep", "v_prep"),
		control_folder,
		mha.prepare_qkv,
	)

	mha_ungated = MultiHeadAttention(c_in, c, N_head, attn_dim, gated=False)
	test_module_forward(mha_ungated, "mha_ungated_forward", "x", "out", control_folder)

	mha_gated = MultiHeadAttention(c_in, c, N_head, attn_dim, gated=True)
	test_module_forward(mha_gated, "mha_gated_forward", "x", "out", control_folder)
	test_module_forward(
		MultiHeadAttention(c_in, c, N_head, attn_dim, gated=True, use_bias_for_embeddings=True),
		"mha_gated_bias_forward",
		("x", "bias"),
		"out",
		control_folder,
	)

	mha_global = MultiHeadAttention(c_in, c, N_head, attn_dim, gated=False, is_global=True)
	test_module_shape(mha_global, "mha_global_init", control_folder)
	test_module_method(
		mha_global,
		"mha_global_prep_qkv",
		("q_global", "k_global", "v_global"),
		("q", "k", "v"),
		control_folder,
		mha_global.prepare_qkv_global,
	)
	test_module_forward(mha_global, "mha_global_forward", "x", "out", control_folder)

	mha_masked = MultiHeadAttention(c_in, c, N_head, attn_dim, use_bias_for_embeddings=True)
	test_module_method(
		mha_masked,
		"attention_mask",
		("x", "fake_attention_mask"),
		"out",
		control_folder,
		lambda x, attention_mask: mha_masked(x, attention_mask=attention_mask),
	)

	return "attention case completed"


@register_case("feature_embedding")
def run_feature_embedding_case() -> str:
	from embedding_checks import (
		c,
		c_e,
		c_m,
		c_z,
		f_e,
		msa_feat_dim,
		tf_dim,
		test_module_forward,
		test_module_method,
		test_module_shape,
	)
	from src.feature_embedding.extra_msa_stack import ExtraMsaBlock, ExtraMsaEmbedder, ExtraMsaStack, MSAColumnGlobalAttention
	from src.feature_embedding.input_embedder import InputEmbedder
	from src.feature_embedding.recycling_embedder import RecyclingEmbedder

	control_folder = str(FEATURE_EMBEDDING_TESTS_DIR)

	input_embedder = InputEmbedder(c_m, c_z, tf_dim, msa_feat_dim=msa_feat_dim, vbins=32)
	test_module_shape(input_embedder, "input_embedder", control_folder)
	test_module_method(input_embedder, "input_embedder_relpos", "residue_index", "z_out", control_folder, lambda x: input_embedder.relpos(x))
	test_module_forward(input_embedder, "input_embedder", "batch", ("m_out", "z_out"), control_folder)

	recycling_embedder = RecyclingEmbedder(c_m, c_z)
	test_module_shape(recycling_embedder, "recycling_embedder", control_folder)
	test_module_forward(recycling_embedder, "recycling_embedder", ("m", "z", "x"), ("m_out", "z_out"), control_folder)

	extra_msa_embedder = ExtraMsaEmbedder(f_e, c_e)
	test_module_shape(extra_msa_embedder, "extra_msa_embedder", control_folder)
	test_module_forward(extra_msa_embedder, "extra_msa_embedder", "batch", "e_out", control_folder)

	msa_global_col_att = MSAColumnGlobalAttention(c_m, c_z, c, 7)
	test_module_shape(msa_global_col_att, "msa_global_col_att", control_folder)
	test_module_forward(msa_global_col_att, "msa_global_col_att", "m", "m_out", control_folder)

	extra_msa_block = ExtraMsaBlock(c_e, c_z)
	test_module_shape(extra_msa_block, "extra_msa_block", control_folder)
	test_module_forward(extra_msa_block, "extra_msa_block", ("e", "z"), "m_out", control_folder)

	extra_msa_stack = ExtraMsaStack(c_e, c_z, num_blocks=3)
	test_module_shape(extra_msa_stack, "extra_msa_stack", control_folder)
	test_module_forward(extra_msa_stack, "extra_msa_stack", ("e", "z"), "m_out", control_folder)

	return "feature_embedding case completed"


@register_case("evoformer")
def run_evoformer_case() -> str:
	from evoformer_checks import (
		N_head,
		c,
		c_m,
		c_z,
		test_module,
		test_module_shape,
	)
	from src.evoformer.evoformer import EvoformerBlock, EvoformerStack
	from src.evoformer.msa_stack import MSAColumnAttention, MSARowAttentionWithPairBias, MSATransition, OuterProductMean
	from src.evoformer.pair_stack import PairStack, PairTransition, TriangleAttention, TriangleMultiplication

	control_folder = str(EVOFORMER_TESTS_DIR)

	msa_row_att = MSARowAttentionWithPairBias(c_m, c_z, c, N_head)
	test_module_shape(msa_row_att, "msa_row_att", control_folder)
	test_module(msa_row_att, "msa_row_att", ("m", "z"), "out", control_folder)

	msa_col_att = MSAColumnAttention(c_m, c, N_head)
	test_module_shape(msa_col_att, "msa_col_att", control_folder)
	test_module(msa_col_att, "msa_col_att", "m", "out", control_folder)

	msa_trans = MSATransition(c_m, 3)
	test_module_shape(msa_trans, "msa_transition", control_folder)
	test_module(msa_trans, "msa_transition", "m", "out", control_folder)

	opm = OuterProductMean(c_m, c_z, c)
	test_module_shape(opm, "outer_product_mean", control_folder)
	test_module(opm, "outer_product_mean", "m", "z_out", control_folder)

	tri_mul_in = TriangleMultiplication(c_z, "incoming", c)
	tri_mul_out = TriangleMultiplication(c_z, "outgoing", c)
	test_module_shape(tri_mul_in, "tri_mul_in", control_folder)
	test_module_shape(tri_mul_out, "tri_mul_out", control_folder)
	test_module(tri_mul_in, "tri_mul_in", "z", "z_out", control_folder)
	test_module(tri_mul_out, "tri_mul_out", "z", "z_out", control_folder)

	tri_att_start = TriangleAttention(c_z, "starting_node", c, N_head)
	tri_att_end = TriangleAttention(c_z, "ending_node", c, N_head)
	test_module_shape(tri_att_start, "tri_att_start", control_folder)
	test_module_shape(tri_att_end, "tri_att_end", control_folder)
	test_module(tri_att_start, "tri_att_start", "z", "z_out", control_folder)
	test_module(tri_att_end, "tri_att_end", "z", "z_out", control_folder)

	pair_trans = PairTransition(c_z, 3)
	test_module_shape(pair_trans, "pair_transition", control_folder)
	test_module(pair_trans, "pair_transition", "z", "z_out", control_folder)

	pair_stack = PairStack(c_z)
	test_module_shape(pair_stack, "pair_stack", control_folder)
	test_module(pair_stack, "pair_stack", "z", "z_out", control_folder)

	evo_block = EvoformerBlock(c_m, c_z)
	test_module_shape(evo_block, "evo_block", control_folder)
	test_module(evo_block, "evo_block", ("m", "z"), ("m_out", "z_out"), control_folder)

	evoformer = EvoformerStack(c_m, c_z, num_blocks=3, c_s=5)
	test_module_shape(evoformer, "evoformer", control_folder)
	test_module(evoformer, "evoformer", ("m", "z"), ("m_out", "z_out", "s_out"), control_folder)

	return "evoformer case completed"


@register_case("feature_extraction")
def run_feature_extraction_case() -> str:
	from src.feature_extraction.feature_extraction import (
		calculate_extra_msa_feat,
		calculate_msa_feat,
		cluster_assignment,
		cluster_average,
		crop_extra_msa,
		create_features_from_a3m,
		initial_data_from_seqs,
		load_a3m_file,
		mask_cluster_centers,
		onehot_encode_aa_type,
		select_cluster_centers,
		summarize_clusters,
	)

	control_folder = _ensure_feature_extraction_controls()
	control_folder_str = str(control_folder)
	base_folder = str(FEATURE_EXTRACTION_TESTS_DIR)

	seqs = load_a3m_file(f"{base_folder}/alignment_tautomerase.a3m")
	first_expected = [
		"PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK",
		"PVVTIELWEGRTPEQKRELVRAVSSAISRVLGCPEEAVHVILHEVPKANWGIGGRLASE",
		"PVVTIEMWEGRTPEQKKALVEAVTSAVAGAIGCPPEAVEVIIHEVPKVNWGIGGQIASE",
		"PIIQVQMLKGRSPELKKQLISEITDTISRTLGSPPEAVRVILTEVPEENWGVGGVPINE",
		"PFVQIHMLEGRTPEQKKAVIEKVTQALVQAVGVPASAVRVLIQEVPKEHWGIGGVSARE",
	]
	assert len(seqs) == 8361 and seqs[:5] == first_expected

	test_seq = "ARNDCQEGHILKMFPSTWYV"
	enc1 = onehot_encode_aa_type(test_seq, include_gap_token=False)
	enc2 = onehot_encode_aa_type(test_seq, include_gap_token=True)
	enc3 = onehot_encode_aa_type(test_seq + "-", include_gap_token=True)
	assert torch.allclose(enc1, torch.nn.functional.one_hot(torch.arange(20), num_classes=21))
	assert torch.allclose(enc2, torch.nn.functional.one_hot(torch.arange(20), num_classes=22))
	enc3_exp = torch.nn.functional.one_hot(torch.cat((torch.arange(20), torch.tensor([21]))), num_classes=22)
	assert torch.allclose(enc3, enc3_exp)

	features = initial_data_from_seqs(seqs)
	expected_features = torch.load(f"{control_folder_str}/initial_data.pt")
	for key, param in features.items():
		assert torch.allclose(param, expected_features[key]), f"Error in computation of feature {key}."

	inp = torch.load(f"{control_folder_str}/initial_data.pt")
	features = select_cluster_centers(inp, seed=0)
	expected_features = torch.load(f"{control_folder_str}/clusters_selected.pt")
	for key, param in features.items():
		assert torch.allclose(param, expected_features[key]), f"Error in computation of feature {key}."

	inp = torch.load(f"{control_folder_str}/clusters_selected.pt")
	features = mask_cluster_centers(inp, seed=1)
	expected_features = torch.load(f"{control_folder_str}/clusters_masked.pt")
	for key, param in features.items():
		assert torch.allclose(param, expected_features[key]), f"Error in computation of feature {key}."

	inp = torch.load(f"{control_folder_str}/clusters_masked.pt")
	features = cluster_assignment(inp)
	expected_features = torch.load(f"{control_folder_str}/clusters_assigned.pt")
	for key, param in features.items():
		assert torch.allclose(param, expected_features[key]), f"Error in computation of feature {key}."

	assignment = torch.tensor([7, 1, 1, 8, 3, 4, 7, 1, 4, 4, 9, 8, 4, 8, 1, 5, 8, 8, 8, 5])
	assignment_count = torch.tensor([0, 4, 0, 1, 4, 2, 0, 2, 6, 1])
	N_clust = 10
	N_res = 3
	N_extra = 20
	dim1 = 5
	dim2 = 7
	ft1_shape = (N_clust, N_res, dim1)
	eft1_shape = (N_extra, N_res, dim1)
	ft2_shape = (N_clust, N_res, dim1, dim2)
	eft2_shape = (N_extra, N_res, dim1, dim2)
	ft1 = torch.linspace(-2, 2, torch.tensor(ft1_shape).prod().item()).reshape(ft1_shape)
	eft1 = torch.linspace(-2, 2, torch.tensor(eft1_shape).prod().item()).reshape(eft1_shape)
	ft2 = torch.linspace(-2, 2, torch.tensor(ft2_shape).prod().item()).reshape(ft2_shape)
	eft2 = torch.linspace(-2, 2, torch.tensor(eft2_shape).prod().item()).reshape(eft2_shape)
	res1 = cluster_average(ft1, eft1, assignment, assignment_count)
	res2 = cluster_average(ft2, eft2, assignment, assignment_count)
	expected_res1 = torch.load(f"{control_folder_str}/cluster_average_res1.pt")
	expected_res2 = torch.load(f"{control_folder_str}/cluster_average_res2.pt")
	assert torch.allclose(res1, expected_res1)
	assert torch.allclose(res2, expected_res2)

	inp = torch.load(f"{control_folder_str}/clusters_assigned.pt")
	features = summarize_clusters(inp)
	expected_features = torch.load(f"{control_folder_str}/clusters_summarized.pt")
	for key, param in features.items():
		assert torch.allclose(param, expected_features[key]), f"Error in computation of feature {key}."

	inp = torch.load(f"{control_folder_str}/clusters_summarized.pt")
	features = crop_extra_msa(inp, seed=2)
	expected_features = torch.load(f"{control_folder_str}/extra_msa_cropped.pt")
	for key, param in features.items():
		assert torch.allclose(param, expected_features[key]), f"Error in computation of feature {key}."

	inp = torch.load(f"{control_folder_str}/extra_msa_cropped.pt")
	msa_feat = calculate_msa_feat(inp)
	expected_feat = torch.load(f"{control_folder_str}/msa_feat.pt")
	assert torch.allclose(msa_feat, expected_feat)

	inp = torch.load(f"{control_folder_str}/extra_msa_cropped.pt")
	extra_feat = calculate_extra_msa_feat(inp)
	expected_feat = torch.load(f"{control_folder_str}/extra_msa_feat.pt")
	assert torch.allclose(extra_feat, expected_feat)

	batch = create_features_from_a3m(f"{base_folder}/alignment_tautomerase.a3m", seed=0)
	expected_batch = torch.load(f"{control_folder_str}/full_batch.pt")
	for key, param in batch.items():
		assert torch.allclose(param, expected_batch[key]), f"Error in computation of feature {key}."
	assert batch['target_feat'].dtype == torch.float32, f"Target feat isn't a float, but {batch['target_feat'].dtype}."

	return "feature_extraction case completed"


@register_case("structure_module")
def run_structure_module_case() -> str:
	from structure_module_checks import (
		N_head,
		c,
		c_s,
		c_z,
		n_layer,
		n_pv,
		n_qp,
		test_module_forward,
		test_module_method,
		test_module_shape,
	)
	from structure_module.ipa import InvariantPointAttention
	from structure_module.structure_module import AngleResNet, AngleResNetLayer, BackboneUpdate, StructureModule, StructureModuleTransition

	control_folder = str(STRUCTURE_MODULE_TESTS_DIR)

	ipa = InvariantPointAttention(c_s, c_z, n_qp, n_pv, N_head, c)
	test_module_shape(ipa, "ipa", control_folder)
	test_module_method(ipa, "ipa_prep", "s", ("q", "k", "v", "qp", "kp", "vp"), control_folder, lambda x: ipa.prepare_qkv(x))
	test_module_method(ipa, "ipa_att_scores", ("q", "k", "qp", "kp", "z", "T"), "att", control_folder, lambda *x: ipa.compute_attention_scores(*x))
	test_module_method(ipa, "ipa_att_outputs", ("att_scores", "z", "v", "vp", "T"), ("v_out", "vp_out", "vp_outnorm", "pairwise_out"), control_folder, lambda *x: ipa.compute_outputs(*x))
	test_module_forward(ipa, "ipa", ("s", "z", "T"), "out", control_folder)

	transition = StructureModuleTransition(c_s)
	test_module_shape(transition, "sm_transition", control_folder)
	test_module_forward(transition, "sm_transition", "s", "s_out", control_folder)

	bb_update = BackboneUpdate(c_s)
	test_module_shape(bb_update, "bb_update", control_folder)
	test_module_forward(bb_update, "bb_update", "s", "T_out", control_folder)

	resnet_layer = AngleResNetLayer(c)
	test_module_shape(resnet_layer, "resnet_layer", control_folder)
	test_module_forward(resnet_layer, "resnet_layer", "a", "a_out", control_folder)

	angle_resnet = AngleResNet(c_s, c)
	test_module_shape(angle_resnet, "angle_resnet", control_folder)
	test_module_forward(angle_resnet, "angle_resnet", ("s", "s_initial"), "alpha", control_folder)

	sm = StructureModule(c_s, c_z, n_layer, c)
	test_module_shape(sm, "structure_module", control_folder)

	def process_outputs_check(*x):
		return sm.process_outputs(*x)

	test_module_method(sm, "sm_process_outputs", ("T", "alpha", "F"), ("pos", "pos_mask", "pseudo_beta"), control_folder, process_outputs_check, include_batched=False)

	def forward_check(*args):
		output = sm(*args)
		return output['angles'], output['frames'], output['final_positions'], output['position_mask'], output['pseudo_beta_positions']

	test_module_method(sm, "structure_module", ("s", "z", "F"), ("angles", "frames", "final_positions", "position_mask", "pseudo_beta_positions"), control_folder, forward_check)

	return "structure_module case completed"


@register_case("model")
def run_model_case() -> str:
	from feature_extraction.feature_extraction import create_features_from_a3m
	from model_checks import (
		c_e,
		c_m,
		c_s,
		c_z,
		f_e,
		num_blocks_evoformer,
		num_blocks_extra_msa,
		tf_dim,
		test_module_forward,
		test_module_method,
		test_module_shape,
	)
	from src.model.model import Model

	control_folder = str(MODEL_TESTS_DIR)
	model = Model(c_m, c_z, c_e, f_e, tf_dim, c_s, num_blocks_extra_msa, num_blocks_evoformer)
	test_module_shape(model, "model", control_folder)

	feature_file = TESTS_DIR / "feature_extraction" / "alignment_tautomerase.a3m"
	single_batches = [create_features_from_a3m(str(feature_file), seed=seed) for seed in range(4)]
	batch = {key: torch.stack([single_batch[key] for single_batch in single_batches], dim=-1) for key in single_batches[0]}

	expected_shapes = {
		"msa_feat": (512, 59, 49, 4),
		"extra_msa_feat": (5120, 59, 25, 4),
		"target_feat": (59, 21, 4),
		"residue_index": (59, 4),
	}

	shapes = {key: value.shape for key, value in batch.items()}
	assert set(expected_shapes.keys()) == set(shapes.keys())
	for key, shape in shapes.items():
		assert expected_shapes[key] == shape, f"Shape mismatch for {key}: {shape} vs {expected_shapes[key]}"

	def test_method(*args):
		outputs = model(*args)
		return outputs['final_positions'], outputs['position_mask'], outputs['angles'], outputs['frames']

	test_module_method(model, "model", "batch", ("final_positions", "position_mask", "angles", "frames"), control_folder, test_method)

	return "model case completed"


@register_case("geometry")
def run_geometry_case() -> str:
	from bunny_renderer import BunnyRenderer
	from geometry.geometry import (
		assemble_4x4_transform,
		calculate_chi_transforms,
		calculate_non_chi_transforms,
		compute_all_atom_coordinates,
		compute_global_transforms,
		conjugate_quat,
		create_3x3_rotation,
		create_4x4_transform,
		invert_4x4_transform,
		makeRotX,
		precalculate_rigid_transforms,
		quat_from_axis,
		quat_mul,
		quat_to_3x3_rotation,
		quat_vector_mul,
		warp_3d_point,
	)
	from geometry.residue_constants import atom_local_positions, atom_mask, atom_types, chi_angles_chain, chi_angles_mask, restype_order

	control_folder = str(GEOMETRY_TESTS_DIR)

	phi = torch.tensor(math.pi / 4)
	n = torch.tensor([0.2, 0.5, -0.3])
	n = n / torch.linalg.vector_norm(n)
	phi_batch = phi.broadcast_to(2, 5)
	n_batch = n.broadcast_to(2, 5, 3)
	q = quat_from_axis(phi, n)
	q_batch = quat_from_axis(phi_batch, n_batch)
	q_exp = torch.load(f"{control_folder}/quat_from_axis_check.pt")
	assert torch.allclose(q, q_exp, atol=1e-5)
	assert torch.allclose(q_batch, q_exp.broadcast_to((2, 5, 4)), atol=1e-5)
	p = torch.tensor([0.3, -0.4, 0.1, 0.8])
	p_batch = p.broadcast_to(2, 5, 4)
	pq = quat_mul(p, q)
	pq_batch = quat_mul(p_batch, q_batch)
	pq_exp = torch.load(f"{control_folder}/quat_mul_check.pt")
	assert torch.allclose(pq, pq_exp, atol=1e-5)
	assert torch.allclose(pq_batch, pq_exp.broadcast_to((2, 5, 4)), atol=1e-5)

	q = torch.tensor([0.3, -0.4, 0.1, 0.8])
	q = q / torch.linalg.vector_norm(q)
	q_copy = q.clone()
	v = torch.tensor([4.0, 1.0, 2.0])
	q_conj = conjugate_quat(q)
	q_conj_batch = conjugate_quat(q.broadcast_to((3, 4, 4)))
	q_conj_exp = torch.load(f"{control_folder}/quat_conjugate_check.pt")
	q_conj_batch_exp = q_conj_exp.broadcast_to(3, 4, 4)
	assert torch.allclose(q_copy, q, atol=1e-5)
	assert torch.allclose(q_conj, q_conj_exp, atol=1e-5)
	assert torch.allclose(q_conj_batch, q_conj_batch_exp, atol=1e-5)
	qv = quat_vector_mul(q, v)
	qv_batch = quat_vector_mul(q.broadcast_to(3, 4, 4), v.broadcast_to(3, 4, 3))
	qv_exp = torch.load(f"{control_folder}/quat_vector_check.pt")
	qv_batch_exp = qv_exp.broadcast_to(3, 4, 3)
	assert torch.allclose(qv, qv_exp, atol=1e-5)
	assert torch.allclose(qv_batch, qv_batch_exp, atol=1e-5)

	ex = torch.tensor([-0.6610, -0.7191, 0.1293])
	ey = torch.tensor([-0.2309, 1.7710, -1.3062])
	ex_batch = ex.broadcast_to(5, 3)
	ey_batch = ey.broadcast_to(5, 3)
	R = create_3x3_rotation(ex, ey)
	R_batch = create_3x3_rotation(ex_batch, ey_batch)
	R_exp = torch.tensor([[-0.6709, -0.6218, 0.4041], [-0.7299, 0.4572, -0.5082], [0.1312, -0.6359, -0.7605]])
	R_exp_batch = R_exp.broadcast_to((5, 3, 3))
	assert torch.allclose(R, R_exp, atol=1e-3)
	assert torch.allclose(R_batch, R_exp_batch, atol=1e-3)

	R = torch.tensor([[0.7071, -0.7071, 0.0000], [0.7071, 0.7071, -0.0000], [0.0000, 0.0000, 1.0000]])
	t = torch.tensor([2.0, 1.0, -1.0])
	R_batch = R.broadcast_to((2, 1, 3, 3))
	t_batch = t.broadcast_to((2, 1, 3))
	T = assemble_4x4_transform(R, t)
	T_batch = assemble_4x4_transform(R_batch, t_batch)
	T_exp = torch.load(f"{control_folder}/assemble_4x4_check.pt")
	T_batch_exp = T_exp.broadcast_to((2, 1, 4, 4))
	assert torch.allclose(T, T_exp, atol=1e-5)
	assert torch.allclose(T_batch, T_batch_exp, atol=1e-5)

	x = torch.tensor([-1.0, 0.0, 3.0])
	y = R @ x + t
	y_4x4 = None
	T = assemble_4x4_transform(R, t)
	y_4x4 = (T @ torch.cat((x, torch.tensor([1.0]))))[..., :3]
	assert torch.allclose(y, y_4x4)

	T = torch.load(f"{control_folder}/assemble_4x4_check.pt")
	x = torch.tensor([-1.0, 0.0, 3.0])
	batch_shape = (2, 1)
	T_batch = T.broadcast_to(batch_shape + T.shape)
	x_batch = x.broadcast_to(batch_shape + x.shape)
	x_warped = warp_3d_point(T, x)
	x_warped_batch = warp_3d_point(T_batch, x_batch)
	x_exp = torch.load(f"{control_folder}/warp_3d_point.pt")
	assert torch.allclose(x_exp, x_warped, atol=1e-5)
	assert torch.allclose(x_warped_batch, x_exp.broadcast_to(batch_shape + x_exp.shape), atol=1e-5)

	ex = torch.tensor([1.2, 0.3, 0.5])
	ey = torch.tensor([1.6, -2.2, 0.3])
	t = torch.tensor([0.4, 0.2, 0.85])
	T = create_4x4_transform(ex, ey, t)
	T_batch = create_4x4_transform(ex.broadcast_to(2, 4, 3), ey.broadcast_to(2, 4, 3), t.broadcast_to(2, 4, 3))
	T_exp = torch.load(f"{control_folder}/create_4x4_T.pt")
	assert torch.allclose(T, T_exp, atol=1e-5)
	assert torch.allclose(T_batch, T_exp.broadcast_to(2, 4, 4, 4), atol=1e-5)

	T = torch.load(f"{control_folder}/create_4x4_T.pt")
	T_batch = T.broadcast_to((5, 1, 4, 4))
	T_inv = invert_4x4_transform(T)
	T_inv_batch = invert_4x4_transform(T_batch)
	T_inv_exp = torch.load(f"{control_folder}/invert_4x4_T.pt")
	assert torch.allclose(T_inv, T_inv_exp, atol=1e-5)
	assert torch.allclose(T_inv_batch, T_inv_exp.broadcast_to(5, 1, 4, 4), atol=1e-5)

	phi = torch.tensor([math.cos(0.5), math.sin(0.5)])
	phi_batch = phi.broadcast_to(5, 3, 2)
	T = makeRotX(phi)
	T_batch = makeRotX(phi_batch)
	T_exp = torch.load(f"{control_folder}/makeRotX_T.pt")
	assert torch.allclose(T, T_exp, atol=1e-5)
	assert torch.allclose(T_batch, T_exp.broadcast_to(5, 3, 4, 4), atol=1e-5)

	transforms = calculate_non_chi_transforms()
	transforms_exp = torch.load(f"{control_folder}/non_chi_transforms.pt")
	assert torch.allclose(transforms, transforms_exp, atol=1e-5)

	chi_transforms = calculate_chi_transforms()
	chi_transforms_exp = torch.load(f"{control_folder}/chi_transforms.pt")
	assert torch.allclose(chi_transforms, chi_transforms_exp, atol=1e-5)
	all_transforms = torch.load(f"{control_folder}/all_transforms.pt")
	assert torch.allclose(precalculate_rigid_transforms(), all_transforms, atol=1e-5)

	N_res = 5
	T = torch.linspace(-4, 4, N_res * 4 * 4).reshape(N_res, 4, 4)
	alpha = torch.linspace(-3, 3, N_res * 7 * 2).reshape(N_res, 7, 2)
	F = torch.tensor([4, 0, 18, 2, 0], dtype=torch.int64)
	global_transforms = compute_global_transforms(T, alpha, F)
	global_transforms_exp = torch.load(f"{control_folder}/global_transforms.pt")
	assert torch.allclose(global_transforms, global_transforms_exp, atol=1e-5)

	atom_positions, atom_mask = compute_all_atom_coordinates(T, alpha, F)
	atom_positions_exp = torch.load(f"{control_folder}/global_atom_positions.pt")
	atom_mask_exp = torch.load(f"{control_folder}/global_atom_mask.pt")
	assert torch.allclose(atom_positions, atom_positions_exp, atol=1e-5)
	assert torch.allclose(atom_mask, atom_mask_exp, atol=1e-5)

	return "geometry case completed"


if __name__ == "__main__":
	main()
