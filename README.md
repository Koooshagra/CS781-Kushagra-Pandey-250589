# CS781 Capstone: In-House Legal Intelligence System

Kushagra Pandey · IIT Kanpur · Codabench: `k_250589`

The project trains a legal language model from scratch and uses legal checkpoints in six downstream tasks. Each phase directory contains the winning Kaggle notebook. Its experiments subfolder contains the other available notebook versions. A single notebook can export several submissions. The notebooks generate their own Python modules in the notebook runtime.

## Official submissions

| Phase | Task | Selected run | Codabench ID | Score | Rank |
| --- | --- | --- | ---: | ---: | ---: |
| 1 | Legal LM pretraining | v2 | 963282 | 0.8329 | 52 |
| 2 | Legal NER | v2 all fixes | 973112 | 0.5667 | 35 |
| 3 | Rhetorical role prediction | v3 LoRMA Plus | 993617 | 0.5006 | 63 |
| 4 | Court judgment prediction | v3 | 1006288 | 0.7455 | 52 |
| 5 | Legal statute identification | v3 | 1016706 | 0.3544 | 90 |
| 6 | Abstractive summarization | v4 LoRMA | 1031013 | 0.8488 | 26 |
| 7 | Prior case retrieval | v4 | 1040870 | 0.3969 | 112 |

## Kaggle setup

1. Import the notebook for the phase you want to run. Use a fresh Kaggle GPU session with Internet enabled and attach the inputs below. Phase 5 is configured to distribute jobs across two T4 GPUs. Phase 6 explicitly requires two GPUs.
2. Grant the notebook access to a Kaggle secret named `HF` for Hugging Face reads. Use a token with access to the official course files and any required private model artifacts. Phase 6 uses `HF_WRITE` for its upload preflight and upload step. For that notebook, set `hf_repo_id` to a repository you control before running the preflight. For other phases, use your own writable destination or skip upload-only cells when reproducing local output. Some launch cells also call upload helpers. Skip or disable those calls for local runs. In Phase 7, set `UPLOAD_TO_HF = False` for local export.
3. Attach each required checkpoint as a Kaggle input or point the notebook's checkpoint variable to a local copy of the **same file**. Inspect the Kaggle Input pane to confirm the mounted path and filename. The v3 and v4 dependencies are distinct from the Phase 2 winning v2 submission. Use the original checkpoint files for reproduction.
4. Run setup, data download and preprocessing, module generation, training, local validation, and submission export in notebook order. The official data references are embedded in the notebooks. Use the full-run settings for competition output and check the exported archive for the expected `prediction.json`. The saved cell outputs were cleared from these copies. The original notebooks retain the validation logs.

[Official course files](https://huggingface.co/Exploration-Lab/CS-781-Capstone) · [Shared v2 byte-level BPE tokenizer](https://huggingface.co/DumbLad/legal-bpe-tokenizer-phase1-v2)

## Run a phase

### Phase 1 · Legal LM pretraining

Open [phase1-v2.ipynb](code/phase1/phase1-v2.ipynb). Prepare the corpus and tokenizer inputs, then run the cells that generate `llama_scratch.py`, `train_llm.py`, and the evaluation script. Run the full v2 training configuration and generate predictions from the selected checkpoint. The full-run export is under `/kaggle/working/final_out/`. The notebook uses a Llama-style decoder with grouped-query attention, RMSNorm, and SwiGLU. The exact decoding configuration for Codabench submission 963282 is pending confirmation.

[Phase 1 v2 artifacts](https://huggingface.co/DumbLad/legal-llm-phase1v2)

### Phase 2 · Legal NER

Open [phase2-v2.ipynb](code/phase2/phase2-v2.ipynb). Attach `phase2v2/model.pt` and run tokenizer setup and the full `phase2_finetune.py` launcher. It uses rank-16 LoRA, five epochs, overlapping 1,024-token windows, and an outside-label weight of 0.3. The notebook includes offset and span checks before export. Collect the archive in `/kaggle/working/out_final/`. The nearby Codabench alternative was `submission_Loraviterbibidir.zip` (ID 972987, score 0.5663).

[Phase 2 v2 artifact folder](https://huggingface.co/DumbLad/legal-ner-phase2-lora-v1/tree/main/phase2v2_bidir_w1024_lora_r16_o0.3_e5)

### Phase 3 · Rhetorical role prediction

Open [phase3-v3.ipynb](code/phase3/phase3-v3.ipynb). Attach `phase2v3/final_model (1).pt`. Run preprocessing and the v3 adapter launcher, then use the export named `v3_lorma_plus.zip`. The selected configuration uses document context, a bidirectional GRU, position features, class weighting, and rank-32 LoRMA Plus. The distillation cells reproduce additional variants.

[Phase 3 v3 artifacts](https://huggingface.co/DumbLad/legal-rr-phase3-v3)

### Phase 4 · Court judgment prediction

Open [phase4-v3.ipynb](code/phase4/phase4-v3.ipynb). Attach `phase2v3/final_model (1).pt` and use the full configuration. The notebook deduplicates the data, builds a held-out split, chunks judgments from the end, and fine-tunes the upper half of the backbone with the lower half frozen. Run held-out threshold selection and export to `/kaggle/working/cs781_phase4/submission.zip`.

[Phase 4 v3 artifacts](https://huggingface.co/DumbLad/cs781-phase4-v3-cjpe)

### Phase 5 · Legal statute identification

Open [phase5-v3.ipynb](code/phase5/phase5-v3.ipynb). Attach `phase2v3/final_model (1).pt`, build the token cache, and run the full candidate launcher. The notebook compares chunked multilabel models and chooses the export using validation scores. The saved run selected `lsi3_N1_mla_lorma_lwan` with per-label thresholds. Collect `/kaggle/working/lsi_v3/submission.zip`.

[Phase 5 v3 artifacts](https://huggingface.co/DumbLad/legal-lsi-phase5-v3)

### Phase 6 · Abstractive summarization

Open [phase6-v4.ipynb](code/phase6/phase6-v4.ipynb). Attach `phase2v4/final_model(2).pt`. After dependency setup and upload preflight, run preparation, the time-bounded rank-32 LoRMA+ training job, and the generation and length-selection cells. The notebook uses sentence-aware source windows and writes the prediction archive under `/kaggle/working/phase6_v4_submission/`.

[Phase 6 v4 prediction artifacts](https://huggingface.co/DumbLad/legal-phase6-v4)

### Phase 7 · Prior case retrieval

Open [phase7-v4.ipynb](code/phase7/phase7-v4.ipynb). Attach `phase2v4/final_model(2).pt` and set the top-level `CHECKPOINT` to that file. The full-run launcher passes this path to the workers and requires it to exist. Use `SMOKE_MODE = False`, materialize the workers, then run passage retrieval, sparse and dense candidate scoring, and LambdaMART reranking. The notebook writes the selected `prediction.json` at the run root and the individual ranking variants in method subfolders. Use the selected export for the v4 submission.

[Phase 7 v4 artifacts](https://huggingface.co/DumbLad/phase7-v4-pcr)

## Provenance to complete

Add the exact Hugging Face repository, file path, and revision for each checkpoint that replaces a Kaggle input. If a checkpoint lives only on Kaggle, link its dataset page and specify the exact filename. The repository links above identify artifact locations. The exact checkpoint filenames and revisions still need to be confirmed where the notebook loads a Kaggle mount. Use the Codabench IDs and original run outputs to fill any unresolved mapping in the experiment index below.

## Experiment notebooks

The version links below cover all 27 supplied notebook versions supplied for the project. Use the Kaggle setup above, then run the setup, configuration, and launcher cells in the linked experiment notebook. Each version has its own checkpoint path and output settings. Use its launcher and export cells to reproduce a specific variant.

| Phase | Notebook version | Architecture, approach, or variants |
| --- | --- | --- |
| 1 | [v1](code/phase1/experiments/phase1-v1.ipynb) | Initial 142M decoder pretraining baseline |
| 1 | [v2](code/phase1/phase1-v2.ipynb) | Smaller GQA decoder, custom byte-level BPE and decoding sweep |
| 1 | [v3](code/phase1/experiments/phase1-v3.ipynb) | Multi-head latent attention and revised token budget |
| 1 | [v4](code/phase1/experiments/phase1-v4.ipynb) | Deeper, narrower GQA with rotary embeddings, query/key normalization, parallel branches, and decoding variants |
| 2 | [v1](code/phase2/experiments/phase2-v1.ipynb) | Partially tuned NER baseline |
| 2 | [v2](code/phase2/phase2-v2.ipynb) | Bidirectional tagging, Viterbi decoding, LoRA, and offset fixes |
| 2 | [v3](code/phase2/experiments/phase2-v3.ipynb) | LoRA, LoRMA Plus and LoRMA Pi comparisons |
| 2 | [v4](code/phase2/experiments/phase2-v4.ipynb) | Adapter variants and a distilled student |
| 2 | [v5](code/phase2/experiments/phase2-v5.ipynb) | Follow-up LoRA training and validation configurations |
| 3 | [v1](code/phase3/experiments/phase3-v1.ipynb) | Sentence/window/document context, PEFT sweep, and student controls |
| 3 | [v2](code/phase3/experiments/phase3-v2.ipynb) | Document-context models, bidirectional GRU, PEFT variants, and distillation |
| 3 | [v3](code/phase3/phase3-v3.ipynb) | Document-level LoRA, AdaLoRA, LoRMA Plus and Pi, and distilled students |
| 4 | [v1](code/phase4/experiments/phase4-v1.ipynb) | Adapter, full, partial-freezing, and distilled classifier runs |
| 4 | [v2](code/phase4/experiments/phase4-v2.ipynb) | Long-document classifier and layer-freezing comparisons |
| 4 | [v3](code/phase4/phase4-v3.ipynb) | End-anchored windows and upper-half fine-tuning |
| 4 | [v4](code/phase4/experiments/phase4-v4.ipynb) | Later judgment classifier using the newer backbone |
| 5 | [v1](code/phase5/experiments/phase5-v1.ipynb) | LoRA-based multilabel statute classifier |
| 5 | [v2](code/phase5/experiments/phase5-v2.ipynb) | Chunked LoRMA classifiers and student runs |
| 5 | [v3](code/phase5/phase5-v3.ipynb) | MLA, LoRMA, label-wise attention, and threshold comparisons |
| 6 | [v1](code/phase6/experiments/phase6-v1.ipynb) | Houlsby, Pfeiffer, LoRA, AdaLoRA, and LoRMA adapter comparison |
| 6 | [v2](code/phase6/experiments/phase6-v2.ipynb) | LoRMA Plus, aligned targets, cached generation, and sentence reduction |
| 6 | [v3](code/phase6/experiments/phase6-v3.ipynb) | Long-document LoRA teacher and distilled student. Attach Phase 2 v3 final_model (1).pt and select two GPUs. Run setup, preparation, teacher and student training, and generation. Export submission_lora.zip and submission_student.zip. |
| 6 | [v4](code/phase6/phase6-v4.ipynb) | Time-bounded LoRMA Plus with sentence-aware windows and a tuned output cap |
| 7 | [v1](code/phase7/experiments/phase7-v1.ipynb) | LoRA, upper-half fine-tuning, distilled students, LoRMA Plus and IA³. Attach Phase 2 v1 final_model.pt. Run worker setup and the training waves, then select the method-specific export. |
| 7 | [v2](code/phase7/experiments/phase7-v2.ipynb) | LoRMA passage retrieval |
| 7 | [v3](code/phase7/experiments/phase7-v3.ipynb) | BM25, dense LoRMA retrieval and reranking alternatives |
| 7 | [v4](code/phase7/phase7-v4.ipynb) | Hybrid passage shortlist and LightGBM LambdaMART reranking |

### Selected non-winning Codabench submissions

| Phase | ID | Export or configuration | Score |
| --- | --- | --- | ---: |
| 2 | 972987 | Loraviterbibidir | 0.5663 |
| 2 | 975690 | v3 LoRA | 0.5648 |
| 2 | 975688 | v3 LoRMA Pi | 0.5427 |
| 2 | 975658 | v3 LoRMA Plus | 0.5584 |
| 2 | 977301 | Distilled 50M student | 0.5262 |
| 2 | 978100 | v5 | 0.5624 |
| 3 | 993630 | v3 LoRA | 0.4848 |
| 3 | 993316 | v3 AdaLoRA | 0.4772 |
| 3 | 993320 | v3 LoRMA Pi | 0.4535 |
| 3 | 992347 | Distilled student | 0.4416 |
| 4 | 1005472 | v2 | 0.7321 |
| 4 | 1008651 | v4 | 0.6787 |
| 5 | 1016238 | v1 | 0.3306 |
| 5 | 1016237 | v2 | 0.3302 |
| 6 | 1030756 | Student | 0.8473 |
| 6 | 1030755 | LoRA | 0.8470 |
| 6 | 1028690 | Early AdaLoRA | 0.7822 |
| 7 | 1040624 | v1 LoRA | 0.3048 |
| 7 | 1040625 | v1 LoRMA | 0.2995 |
| 7 | 1040626 | v1 distilled LoRA student | 0.3006 |
| 7 | 1040627 | v1 distilled partial student | 0.2983 |
| 7 | 1040628 | v1 upper-half fine-tuning | 0.3030 |
| 7 | 1040629 | v1 IA³ | 0.2953 |
| 7 | 1040680 | v2 LoRMA | 0.3420 |
| 7 | 1040635 | v3 LoRMA | 0.3320 |
| 7 | 1040634 | v3 BM25 | 0.2835 |
| 7 | 1040636 | v3 reranker | 0.2056 |
| 7 | 1040871 | v4 hybrid | 0.3768 |

[Phase 7 v1 artifacts](https://huggingface.co/DumbLad/phase7-v1-pcr)
