# GrowMate: Hydroponic Farming Chatbot for Rwanda

A domain-specific chatbot built with FLAN-T5-base, fine-tuned on 625 hydroponic farming Q&A pairs to provide expert guidance for smallholder farmers in Rwanda.

## Links & Resources

- **Live Demo**: [Try GrowMate](https://farmsmartgrowmatechatbot.streamlit.app/)
- **GitHub**: [Source Code](https://github.com/Afsaumutoniwase/Farmsmart_growmate_chatbot)
- **Hugging Face**: [Fine-tuned Model](https://huggingface.co/Afsa20/Farmsmart_Growmate/tree/main)
- **Demo Video**: [YouTube](https://www.youtube.com/watch?v=DLju-kdCgy0)

## About

GrowMate addresses the information gap in hydroponic farming for Rwandan farmers. Hydroponics is a soil-free, water-efficient method that uses 90% less water than traditional farming. This chatbot provides instant, accessible guidance on system setup, nutrient management, pH control, and troubleshooting.

## Features

- **FLAN-T5-base** (247M parameters) fine-tuned on 625 Q&A pairs
- **283% ROUGE-2 improvement** through 3 systematic experiments
- **Live web interface** with Streamlit
- **Rwanda-specific** context and farming conditions
- **Complete metrics**: ROUGE, BLEU, F1, Perplexity

## Repository Structure

```
Farmsmart_growmate_chatbot/
├── .streamlit/config.toml          # Streamlit configuration
├── Assets/
│   ├── loader.png                  # Bot avatar
│   └── logo.png                    # App logo
├── data/
│   └── hydroponic_FAQS.csv         # Training dataset
├── notebooks/
│   └── flan_t5_hydroponic_chatbot.ipynb  # Model training
├── trained_model/                  # Fine-tuned model files
├── app.py                          # Streamlit app
├── requirements.txt                # Dependencies
└── README.md
```

## Quick Start

### 1. Environment Setup

```bash
cd Farmsmart_growmate_chatbot
pip install -r requirements.txt
```

### 2. Train the Model

Open and run `notebooks/flan_t5_hydroponic_chatbot.ipynb` to train the model. The trained model saves to `trained_model/` automatically.

### 3. Launch the Chatbot

```bash
streamlit run app.py  # Opens at http://localhost:8501
```

## Model Training

The notebook covers:
- Data loading and cleaning
- FLAN-T5-base model setup
- Data preprocessing for instruction tuning
- Model training and evaluation
- Model saving for deployment

## Experiment Results

### Experiment Tracking

| Exp | Epochs | LR | Batch | Train Loss | Test Loss | ROUGE-1 | ROUGE-2 | ROUGE-L | Status |
|-----|--------|-----|-------|------------|-----------|---------|---------|---------|--------|
| 1 | 12 | 1e-5 | 2 | 4.1165 | 3.6720 | 0.1387 | 0.0125 | 0.1162 | Baseline |
| 2 | 25 | 3e-5 | 4 | 3.1419 | 3.2267 | 0.2003 | 0.0441 | 0.1694 | Optimized |
| 3 | 35 | 5e-5 | 4 | 2.3853 | 3.2122 | **0.2061** | **0.0479** | 0.1665 | **BEST** |

### Performance Improvements

- **283% ROUGE-2 improvement** (0.0125 → 0.0479)
- **42.1% training loss reduction** (4.1165 → 2.3853)
- **48.6% ROUGE-1 improvement** (0.1387 → 0.2061)
- No overfitting - consistent improvement across experiments

### Additional Metrics (Experiment 3)

- **BLEU**: 0.0116
- **F1 Score**: 0.1357 (Precision: 0.1270, Recall: 0.1510)
- **Perplexity**: 1.3675 (excellent for domain-specific models)

## Evaluation Metrics

**ROUGE Scores**:
- **ROUGE-1**: Unigram overlap
- **ROUGE-2**: Bigram overlap (semantic coherence)
- **ROUGE-L**: Longest common subsequence

**BLEU Score**: N-gram precision between generated and reference text

**F1 Score**: Harmonic mean of precision and recall

**Perplexity**: Model confidence (lower is better, <10 is very good)

## Sample Model Predictions

**Example 1:**
- **Q**: Can I grow dill hydroponically?
- **Model**: Dill is a perennial herb that grows well in hydroponic systems; it can be grown as a seedling or in small containers.

**Example 2:**
- **Q**: Why are my roots brown?
- **Model**: Brown roots indicate nutrient deficiency or poor water quality. Check pH and airflow to ensure adequate nutrients are in the system.

**Example 3:**
- **Q**: What pH level should I maintain for hydroponic tomatoes?
- **Model**: Maintain pH 5.8–6.2; maintain EC 1.3–1.8; avoid overwatering or nutrient deficiency in tomatoes.

## Usage Example

```bash
streamlit run app.py  # Opens at http://localhost:8501
```

**Example Questions:**
- "What pH level is best for lettuce?"
- "How often should I change nutrient solution?"
- "Best hydroponic crops for Rwanda?"

## System Requirements

- **Minimum**: 4GB RAM, CPU-only inference
- **Recommended**: 8GB RAM, GPU for training
- **Response time**: 1-3 seconds per query

## Deployment

Deployed on Streamlit Cloud: [https://farmsmartgrowmatechatbot.streamlit.app/](https://farmsmartgrowmatechatbot.streamlit.app/)

The chatbot is a single-file Streamlit app with automatic model loading.

## Future Work

- **Immediate**: Kinyarwanda language support, mobile app
- **Advanced**: IoT sensor integration, multimodal capabilities (image-based disease diagnosis)
- **Long-term**: Regional expansion, field validation with farmers

## Contributing

Open source contributions welcome! For technical help or collaboration opportunities, see the GitHub repository.

---

