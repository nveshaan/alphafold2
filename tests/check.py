from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections.abc import Callable
from typing import Any

import torch


CaseFn = Callable[[], Any]

CASES: dict[str, CaseFn] = {}

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
TESTS_DIR = ROOT / "tests"
ATTENTION_TESTS_DIR = TESTS_DIR / "attention"
FEATURE_EMBEDDING_TESTS_DIR = TESTS_DIR / "feature_embedding"
EVOFORMER_TESTS_DIR = TESTS_DIR / "evoformer"
FEATURE_EXTRACTION_TESTS_DIR = TESTS_DIR / "feature_extraction"

for path in reversed(
	[
		str(ROOT),
		str(SRC_DIR),
		str(ATTENTION_TESTS_DIR),
		str(FEATURE_EMBEDDING_TESTS_DIR),
		str(EVOFORMER_TESTS_DIR),
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


if __name__ == "__main__":
	main()
