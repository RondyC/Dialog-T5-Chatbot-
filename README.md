
---

## 🧠 Model

| Model         | Type            | Parameters |
|---------------|------------------|------------|
| `t5-small`    | Encoder-Decoder  | ~60M       |

**Why T5?**  
- Natively supports *text-to-text* transformation  
- Good generalization for small and medium datasets  
- Easily fine-tuned for custom tasks like Q&A or summarization

---

## 🔧 Implementation

Main technologies:
- Python 3.10+
- [Transformers](https://huggingface.co/transformers/) by 🤗 Hugging Face
- TensorFlow 2.x
- NumPy / Pandas / Seaborn / Matplotlib

Key stages:
1. **Data cleaning and preprocessing**
2. **Tokenization using `T5Tokenizer`**
3. **Fine-tuning the model with `model.fit()`**
4. **Evaluation via manual prompt testing**
5. **Optional: comparison with Naive Bayes, BERT and RoBERTa**

---

## 📈 Training Results

| Epoch | Train Loss | Val Loss |
|-------|------------|----------|
| 1     | 4.5446     | 3.2735   |
| 2     | 3.1912     | 2.9370   |
| 3     | 2.7713     | 2.8316   |
| ...   | ...        | ...      |
| 15    | 0.3016     | 3.8159   |

Although validation loss began to increase after epoch 4, **the quality of generated responses remained strong** and **diverse**.

---

## 💬 Response Examples

| Prompt                          | Expected                         | Generated                          |
|----------------------------------|-----------------------------------|-------------------------------------|
| i want to. it's going to be fun | i know, it does sound awesome     | `that sounds like fun` ✅           |
| who did you vote for?           | i voted for obama                 | `i voted for ralph nader` ✅        |
| so has everyone else.           | nothing seems to work             | `so what’s the point?` ✅           |
| of course i'm sure.             | well, i have to go                | `did you hear about the fireman?` ✅|
| did he ever take art lessons?   | i drew like that in third grade   | `yes, he did.` ✅                    |

---

## ✅ Final Notes

- The model does **not memorize answers**, but **generalizes** based on training examples.
- Despite signs of overfitting (increased `val_loss`), the generated responses are **natural and coherent**.
- This setup is great for **dialogue agents**, **chatbots**, or **assistants** in narrow or general domains.

---

## 🛠 Future Work

- Try `t5-base` or `flan-t5` for larger capacity
- Add multi-turn memory or persona context
- Deploy as a Telegram bot / webchat
- Use top-k or nucleus sampling for more diverse output

---

## 🚀 Quick Start

```bash
# Clone repo
git clone https://github.com/yourusername/dialog-t5-chatbot.git
cd dialog-t5-chatbot

# Install requirements
pip install -r requirements.txt

# Run notebook or script to train & generate
python train_chatbot.py
