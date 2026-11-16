# Kaggle first-run pilot

This notebook trains a 49M-parameter defensive GPT from scratch. Source preparation, pretraining, SFT, and evaluations run in separate cells. It uses a private, attributed corpus and a separate held-out set of expert-reviewed defensive QA. It is a narrow research experiment, not a capable general cybersecurity assistant. Do not treat its checkpoint as deployment-ready.

## Submit the private first run

1. To include your style data, prepare it locally and review `data/derived/copilot_style_private.jsonl` for remaining private information.
2. Install Kaggle's CLI in a local environment and authenticate using your Kaggle API token. Never add `kaggle.json` or a token to this repository.
3. Get your Kaggle username from your Kaggle profile. Do not print or share the API key.
4. Package the style data: `python cyber_50m/package_kaggle_style_data.py --owner YOUR_KAGGLE_USERNAME`.
5. Package the corpus, code, config, and optional tone file using your Kaggle username: `python cyber_50m/package_kaggle_style_data.py --owner YOUR_KAGGLE_USERNAME --slug defensive-cyber-50m-pilot-data`.
6. Create the private dataset once: `kaggle datasets create -p .\data\kaggle-upload`; for later code/data changes, publish a new private version with `kaggle datasets version -p .\data\kaggle-upload -m "Update training pipeline"`.
7. Submit the notebook: `kaggle kernels push -p .\kaggle`.

The notebook reads the prepared corpus, task evaluator, and training scripts from a private Kaggle dataset. It downloads licensed Fenrir and defensive CTI sources at runtime and skips the optional Cosmopedia download that produced OpenStax descriptor retries. It derives update counts from actual chunk counts, targets up to two shuffled passes per phase, caps each at 30,000 updates, validates every 500 updates, and stops early on a validation plateau. Progress is printed every 50 updates, and each stage has its own cell. It then generates answers on the held-out QA split. The private dataset also includes the redacted Human-only tone file. Raw archives, raw conversations, processed corpora, credentials, and model outputs are excluded from Git.

## Limits

The previous 8,000-step run reduced validation loss, but held-out generations remained weak. The expanded run published no result artifacts and its log showed no optimizer-step output. The workflow now makes each stage observable and budgets updates from the actual data size, but neither change guarantees model quality. This does not demonstrate coverage of 90% of cybersecurity tasks. The network-flow archives are not language-model input. More licensed text and expert-reviewed defensive prompt/answer examples are needed before a meaningful capability claim.
