# Traditional Dress Recognition & Trend Analytics Dashboard

A real-world ML project for classifying South Asian traditional clothing
(saree, lehenga, sherwani, kurta, etc.) and predicting which traditional
garments trend best, using two real datasets — no synthetic data, no
Western fashion data.

## Why this project

Most public "fashion AI" datasets and tutorials are built around Western
clothing (t-shirts, jeans, jackets) and fail badly on ethnic wear, because
the visual structure is completely different (drape vs. cut, layered
garments, embroidery-heavy texture). This project fixes that gap for
South Asian traditional wear and produces something genuinely useful:
auto-tagging for e-commerce sellers, cataloging tools for boutiques, and a
trend-prediction tool that tells a seller/designer which kind of garment
is likely to sell well.

## Two real datasets

| Purpose | Dataset | What it gives you |
|---|---|---|
| Image classification | **IndoFashion** (CVPR-W 2021) — 106K images, 15 classes | Request access: https://github.com/IndoFashion/IndoFashion (fill the form, they email you a download script) |
| Trend / popularity prediction | **Myntra Sales Dataset** (Kaggle) — ~52K real product listings with Price, MRP, Discount%, Rating, Number of Ratings, Brand | https://www.kaggle.com/datasets/skmewati/myntra-sales-dataset |

Optional, once the core works:
- **Myntra Fashion Product Dataset** (Kaggle, has both images + descriptions of kurtas/palazzos/dupattas) — good if you want to add a text/description angle later (your team's hashtag/BERT idea, applied to product descriptions instead of unavailable Instagram captions).

## Project structure

```
traditional-dress-dashboard/
├── data/
│   ├── indofashion/        # put downloaded IndoFashion images + train.json/val.json/test.json here
│   └── myntra/              # put myntra_dataset_ByScraping.csv here
├── src/
│   ├── prepare_indofashion.py   # turns IndoFashion json into ImageFolder structure
│   ├── prepare_myntra.py        # filters to traditional wear + engineers popularity labels
│   ├── train_classifier.py      # trains & compares 4 CNN models on IndoFashion
│   ├── train_popularity.py      # trains & compares 5 tabular models on Myntra data
│   └── utils.py
├── models/                  # trained models + metrics land here (auto-created)
├── reports/                 # confusion matrices, leaderboard JSON (auto-created)
├── app/
│   └── streamlit_app.py     # the dashboard
└── requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

GPU note: IndoFashion has 106K images. Training 4 CNNs on it needs a GPU —
use Google Colab (free T4 GPU) or Kaggle Notebooks (free GPU, and the
dataset can be added directly as a Kaggle input). Training on CPU will
work but will be slow; reduce `--epochs` and use a subset for quick
iteration first.

## Run order

```bash
# 1. Prepare data
python src/prepare_indofashion.py --raw_dir data/indofashion --out_dir data/indofashion/processed
python src/prepare_myntra.py --csv data/myntra/myntra_dataset_ByScraping.csv --out_dir data/myntra/processed

# 2. Train & compare image classifiers (run on Colab/Kaggle GPU for full dataset)
python src/train_classifier.py --data_dir data/indofashion/processed --epochs 8

# 3. Train & compare popularity/trend models
python src/train_popularity.py --csv data/myntra/processed/traditional_wear_clean.csv

# 4. Launch the dashboard
streamlit run app/streamlit_app.py
```

The dashboard reads whatever is in `models/` and `reports/`. If you haven't
trained yet, it runs in **demo mode** and tells you exactly that — it will
never silently show fabricated numbers as if they were real results.

## What goes in the dashboard

1. **Dress Classifier** — upload a photo, every trained CNN votes on the
   garment category, see confidence per model side by side.
2. **Model Leaderboard** — accuracy/F1/precision/recall per model,
   confusion matrix viewer, training time comparison.
3. **Trend & Popularity Explorer** — real charts from the Myntra data:
   which traditional categories sell most, price vs. rating, brand
   leaderboard.
4. **Popularity Predictor** — pick a category, price, discount, brand
   tier → predicts a Low/Medium/High popularity tier and shows which
   features drove that prediction (feature importance).

## Next steps once this works

- Add the description-text angle (TF-IDF or DistilBERT on product titles)
  as a second signal into the popularity model — this is where your
  team's original NLP plan plugs back in.
- Deploy: Streamlit Community Cloud (free) or Hugging Face Spaces.
- Swap in BanglaDressNet / Pakistani fashion images as an extra
  "regional" tab once you've inspected their actual label quality.
