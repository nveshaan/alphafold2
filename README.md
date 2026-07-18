# AlphaFold2
A minimal, trainable, pytorch re-implementation of alphafold2, compatible with openfold weights. Read more about it at [Reimplementing AlphaFold2 (Part1)](https://nveshaan.github.io/blog/alphafold2-part1/) and [Reimplementing AlphaFold2 (Part2)](https://nveshaan.github.io/blog/alphafold2-part2/).

![tautomerase](assets/movie1.gif)
*Tautomerase structure prediction using openfold weights.*

## Project Structure
```bash
.
├── data/
├── scripts
│   ├── download_openfold_params.sh
│   └── openfold_demo.py
├── src
│   ├── attention
│   │   └── mha.py
│   ├── evoformer
│   │   ├── dropout.py
│   │   ├── evoformer.py
│   │   ├── msa_stack.py
│   │   └── pair_stack.py
│   ├── feature_embedding
│   │   ├── extra_msa_stack.py
│   │   ├── input_embedder.py
│   │   └── recycling_embedder.py
│   ├── feature_extraction
│   │   └── feature_extraction.py
│   ├── geometry
│   │   ├── geometry.py
│   │   └── residue_constants.py
│   ├── model
│   │   ├── model.py
│   │   └── utils.py
│   └── structure_module
│       ├── ipa.py
│       └── structure_module.py
└── tests/
```

## Setup
```bash
git clone https://github.com/nveshaan/alphafold2.git
cd alphafold2
uv sync

brew install awscli
./scripts/download_openfold_params.sh ./data
```

## OpenFold Demo
Requires `ChimeraX` installed.
```py
python -m scripts.openfold_demo
```

## Algorithm Map
A mapping from the Codebase to AlphaFold2 Supplement [Paper](/assets/af2_paper.pdf).
| Algorithm | Description | Location |
| :--- | :--- | :--- |
| 1 | MSABlockDeletion | |
| 2 | Inference | `model/model.py: Model` |
| 3 | InputEmbedder | `feature_embedding/input_embedder.py: InputEmbedder` |
| 4 | relpos | `feature_embedding/input_embedder.py: InputEmbedder` |
| 5 | one_hot | |
| 6 | EvoformerStack | `evoformer/evoformer.py: EvoformerStack` |
| 7 | MSARowAttentionWithPairBias | `evoformer/msa_stack.py: MSARowAttentionWithPairBias` |
| 8 | MSAColumnAttention | `evoformer/msa_stack.py: MSAColumnAttention` |
| 9 | MSATransition | `evoformer/msa_stack.py: MSATransition` |
| 10 | OuterProductMean | `evoformer/msa_stack.py: OuterProductMean` |
| 11 | TriangleMultiplicationOutgoing | `evoformer/pair_stack.py: TriangleMultiplication` |
| 12 | TriangleMultiplicationIncoming | `evoformer/pair_stack.py: TriangleMultiplication` |
| 13 | TriangleAttentionStartingNode | `evoformer/pair_stack.py: TriangleAttention` |
| 14 | TriangleAttentionEndingNode | `evoformer/pair_stack.py: TriangleAttention` |
| 15 | PairTransition | `evoformer/pair_stack.py: PairTransition` |
| 16 | TemplatePairStack | |
| 17 | TemplatePointwiseAttention | |
| 18 | ExtraMsaStack |`feature_embedding/extra_msa_stack.py: ExtraMsaStack` |
| 19 | MSAColumnGlobalAttention | `feature_embedding/extra_msa_stack.py: MSAColumnGlobalAttention` |
| 20 | StructureModule | `structure_module/structure_module.py: StructureModule` |
| 21 | rigidFrom3Points | |
| 22 | InvariantPointAttention | `structure_module/ipa.py: InvariantPointAttention` |
| 23 | BackboneUpdate | `structure_module/structure_module.py: BackboneUpdate` |
| 24 | computeAllAtomCoordinates | `geometry/geometry.py: computeAllAtomCoordinates` |
| 25 | makeRotX | `geometry/geometry.py: makeRotX` |
| 26 | renameSymmetricGroundTruthAtoms | |
| 27 | torsionAngleLoss | |
| 28 | computeFAPE | |
| 29 | predictPerResidueLDDT | |
| 30 | RecyclingInference | |
| 31 | RecyclingTraining | |
| 32 | RecyclingEmbedder | `feature_embedding/recycling_embedder.py: RecyclingEmbedder` |

## Acknowledgements

 - Mandon, K. (2024). AlphaFold Decoded: Implementing AlphaFold 2 from Scratch in PyTorch. YouTube Course Playlist. https://youtube.com/playlist?list=PLJ0WcPQS7xJVJr6ceIPFSkAGAgrkmw1c9&si=xAo2FNEvpYATEea6
 - Jumper, J., Evans, R., Pritzel, A., et al. (2021). Highly accurate protein structure prediction with AlphaFold. Nature, 596, 583–589. https://doi.org/10.1038/s41586-021-03819-2
 - Ahdritz, G., Bouatta, N., Kofman, S., et al. (2024). OpenFold: Retraining AlphaFold2 yields new insights into its learning mechanisms and economy of scale. Nature Methods. https://doi.org/10.1038/s41592-024-02272-x

## License

This project is distributed under the MIT License. See the `LICENSE` file for details.