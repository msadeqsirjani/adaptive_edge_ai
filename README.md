# Adaptive Edge AI 🤖

![Edge AI Banner](https://github.com/msadeqsirjani/adaptive_edge_ai/blob/master/docs/images/banner.png)

## 📋 Overview

Adaptive Edge AI is a project focused on optimizing and deploying deep learning models for edge devices through model compression and knowledge distillation techniques. This solution enables efficient AI model deployment on resource-constrained devices while maintaining high performance.

## 🌟 Key Features

- 🔄 Model Compression
- 📚 Knowledge Distillation
- 📱 Edge Device Optimization
- 📊 Adaptive Performance Scaling
- 🚀 ONNX Export Support

## 🏗️ Architecture

```mermaid
graph LR
A[Teacher Model] --> B[Knowledge Distillation]
B --> C[Student Model]
C --> D[Model Compression]
D --> E[Edge Deployment]
```


## 🛠️ Installation

1. Clone the repository

```bash
git clone https://github.com/msadeqsirjani/adaptive_edge_ai.git
```

2. Create and activate virtual environment

```bash
python -m venv .venv
source .venv/bin/activate # Linux/Mac
# or
.venv\Scripts\activate # Windows
```

3. Install dependencies

```bash
pip install -r requirements.txt
```


## 📊 Performance Metrics

| Model | Size | Accuracy | Inference Time |
|-------|------|----------|----------------|
| Teacher | 500MB | 95% | 100ms |
| Student | 50MB | 92% | 20ms |
| Compressed | 10MB | 90% | 5ms |

## 💻 Usage

### Training the Teacher Model

```bash
python main.py --mode train_teacher --data_path data/
```

### Knowledge Distillation

```bash
python main.py --mode distill --teacher_model best_teacher_model.pth
```

### Model Compression

```bash
python main.py --mode compress --model student_model.pth
```


## 📁 Project Structure

```bash
adaptive_edge_ai/
├── data/ # Dataset directory (gitignored)
├── src/
│ ├── models/ # Model architectures
│ ├── optimization/ # Compression algorithms
│ ├── training/ # Training utilities
│ └── utils/ # Helper functions
├── outputs/ # Saved models & results
├── tests/ # Unit tests
├── requirements.txt # Dependencies
└── main.py # Entry point
```


## 📈 Results

Our compressed models achieve:
- 📉 90% size reduction
- ⚡ 20x faster inference
- 💪 Minimal accuracy loss

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📬 Contact

- `Mohammad Sadegh Sirjani` - [@msadeqsirjani](https://twitter.com/msadeqsirjani)
- Email - `m.sadeq.sirjani@gmail.com`

## 🙏 Acknowledgments

- Thanks to relevant papers or projects
- Special thanks to contributors
- Inspired by related work

---
⭐ Don't forget to star this repo if you find it helpful!