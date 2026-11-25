# Adversarial Attacks on an Automated Cheque-Amount Digit Reader

**Course:** Responsible and Safe AI (RSA)  
**Student Name:** Shrey Lakhataria  
**ID Number:** 23CS039  
**Practical Set:** Set A  

---

## Scenario & Problem Statement

A cooperative bank is introducing automated cheque clearing in which a computer-vision model reads the handwritten digits of the cheque amount and account number. The fraud-prevention cell has warned that an attacker could add tiny, almost invisible pixel perturbations (for example, through a specially printed background pattern) so that the model reads a `"1"` as `"7"` or a `"3"` as `"8"`, altering the cleared amount without visible tampering.

Before the system goes live, the bank's AI Security team evaluates the digit reader's vulnerability to adversarial evasion attacks (**FGSM** and **PGD**) and hardens the model using **adversarial training**.

---

## Practical Objectives (Tasks 1–6)

1. **Baseline CNN Training:** Load MNIST (60,000 train / 10,000 test, 28×28 grey-scale, pixel values in `[0, 1]`) and train a 2-Conv + 2-FC CNN on CPU (`torch.manual_seed(42)`, Adam, `lr = 0.001`, batch size `64`, `2` epochs). Report clean test accuracy.
2. **FGSM Attack:** Implement the Fast Gradient Sign Method and evaluate the baseline CNN for $\varepsilon \in \{0.05, 0.10, 0.20, 0.30\}$.
3. **PGD Attack:** Implement Projected Gradient Descent ($L_\infty$, 10 iterations, step size $\alpha = \varepsilon / 4$, random start) for the same $\varepsilon$ values and compare it against FGSM.
4. **Visual Inspection:** Display and save 5 clean digits alongside their FGSM ($\varepsilon = 0.20$) adversarial counterparts with predicted labels (`clean_vs_adversarial.png`) and identify mis-read digits.
5. **Adversarial Training (Defence):** Fine-tune the baseline model for 2 epochs on a 50:50 mix of clean and FGSM ($\varepsilon = 0.20$) images, then re-evaluate clean, FGSM (0.20), and PGD (0.20) accuracy.
6. **Visualization & Comparison:** Plot test accuracy vs $\varepsilon$ (`accuracy_vs_epsilon.png`) and tabulate baseline vs adversarially-trained performance.

---

## Repository Structure

```text
Shrey_RSA/
├── Adversarial_Attacks_Cheque_Digit_Reader.ipynb   # Google Colab / Jupyter notebook (cell-by-cell)
├── cheque_digit_adversarial.py                     # Standalone Python script
├── RSA_Practical_Report_23CS039_SetA.pdf           # Complete submission report (Aim, Algorithm, Code, Output, Q1-Q4, Conclusion)
├── requirements.txt                                # Python dependencies
├── .gitignore                                      # Git ignore rules
└── README.md                                       # Project documentation
```

---

## Tools & Environment

- **Language:** Python 3
- **Libraries:** PyTorch, Torchvision, NumPy, Matplotlib
- **Dataset:** MNIST Handwritten Digits (28×28 grey-scale, normalized to `[0, 1]`)
- **Execution Guidelines:** `torch.manual_seed(42)`, CPU execution, 2 epochs per training phase

---

## How to Run in Google Colab

1. Upload `Adversarial_Attacks_Cheque_Digit_Reader.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Keep the runtime on **CPU** (as required by the practical guidelines).
3. Run all cells sequentially (`Runtime` → `Run all`).
4. The notebook automatically downloads MNIST via `torchvision.datasets.MNIST`, trains the baseline CNN, runs FGSM and PGD attacks, applies adversarial training, and saves both `clean_vs_adversarial.png` and `accuracy_vs_epsilon.png`.

### Running as a Python Script

```bash
pip install -r requirements.txt
python cheque_digit_adversarial.py
```

---


