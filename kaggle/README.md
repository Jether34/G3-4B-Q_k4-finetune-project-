# Kaggle first-run pilot

This notebook trains a 49M-parameter defensive GPT from scratch. It uses a private, attributed corpus and a separate held-out set of expert-reviewed defensive QA. It is a narrow research run, not a capable general cybersecurity assistant. Do not treat its checkpoint as deployment-ready.

## Submit the private first run

1. To include your style data, prepare it locally and review `data/derived/copilot_style_private.jsonl` for remaining private information.
2. Install Kaggle's CLI in a local environment and authenticate using your Kaggle API token. Never add `kaggle.json` or a token to this repository.
3. Get your Kaggle username from your Kaggle profile. Do not print or share the API key.
4. Package the style data: `python cyber_50m/package_kaggle_style_data.py --owner YOUR_KAGGLE_USERNAME`.
5. Package the corpus, code, config, and optional tone file using Kaggle account owner `pycache`: `python cyber_50m/package_kaggle_style_data.py --owner pycache --slug defensive-cyber-50m-pilot-data`.
6. Create the dataset privately (Kaggle CLI defaults to private): `kaggle datasets create -p .\data\kaggle-upload`.
7. Submit the notebook: `kaggle kernels push -p .\kaggle`.

The notebook reads the prepared corpus, task evaluator, and training script from a private Kaggle dataset. It trains for 3,000 steps on one GPU, saves the best checkpoint and metrics to Kaggle output, then generates answers on the held-out QA split. The private dataset also includes the redacted Human-only tone file. Raw archives, raw conversations, processed corpora, credentials, and model outputs are excluded from Git.

## Limits

The 100-step run confirms the data and training pipeline can execute. It will overfit this small corpus and does not demonstrate coverage of 90% of cybersecurity tasks. The network-flow archives are not language-model input. More licensed text and expert-reviewed defensive prompt/answer examples are needed before a meaningful training run.
