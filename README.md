# **3-SAT Genetic Algorithm + Wisdom of Artificial Crowds Solver**

This repository contains the full implementation, experiments, results, and visualization tools for a research project exploring **Genetic Algorithms (GA)** and **Wisdom of Artificial Crowds (WoC)** for solving **3-SAT**, a classic NP-Complete optimization problem.

The project systematically compares a baseline GA against WoC variants across small, medium, and large 3-SAT instances and includes extensive logs, graphs, and a Streamlit visualizer used during experimentation.

## **Repository Contents**

### **Source Code**
- `cse545_final_project.py` — Main implementation of the Genetic Algorithm and WoC variants  
- `visualizer.py` — Streamlit dashboard for viewing experiment logs and progress  
- `3sat_instance_small.json`  
- `3sat_instance_medium.json`  
- `3sat_instance_large.json`  

### **Experiment Logs (JSON)**
Stored in `experiment_logs/`:

- Baseline GA (`Baseline_small`, `Baseline_medium`, `Baseline_large`)
- WoC with various subpopulation counts (K=3, 5, 10)
- Weighted vs. unweighted wisdom aggregation
- Mutation-rate experiments (0.01, 0.05)
- Large-instance stress tests

### **GUI Output Screenshots**
Located in `GUI Screenshots/`, categorized by experiment type:
- Fitness curves  
- Solved vs. unsolved run distributions  
- Solution generation plots  

### **Reports & Presentation**
- `545 Final Project Presentation.pdf`
- `Solving 3-SAT Using Genetic Algorithms and Wisdom of Artificial Crowds.doc`


## **Running the Solver**

### **Run the main GA / WoC solver**
```bash
python cse545_final_project.py
```
### **Run the Streamlit visualizer**
```bash
streamlit run visualizer.py
```
Make sure Streamlit is installed:
```bash
pip install streamlit
```


## **🧠 Project Overview**

### **Goal**
Evaluate whether **Wisdom of Artificial Crowds** improves Genetic Algorithm performance on small, medium, and large 3-SAT instances.

### **Methods Implemented**
- Baseline GA with:
  - Tournament selection  
  - Two-point crossover  
  - Bit-flip mutation  
- WoC-GA:
  - Multiple subpopulations (K = 3, 5, 10)
  - Wisdom aggregation of elites
  - Weighted and unweighted variants
  - Multiple mutation-rate sensitivity tests

### **Key Questions**
- Does WoC discover higher-quality solutions than a single GA?
- How does subpopulation count affect diversity and performance?
- Does weighting elite contributions help or hurt stability?
- How do the methods scale on larger SAT instances?


## **📊 Results Summary**

- **Small instance:**  
  Both GA and WoC solve easily; WoC converges faster.

- **Medium instance:**  
  WoC outperforms GA consistently, especially K=5.

- **Large instance:**  
  Neither approach fully solves the instance, but WoC finds significantly better partial assignments.

- **Unweighted vs Weighted Wisdom:**  
  - Unweighted → more stable, preserves diversity  
  - Weighted → higher peaks but higher variance  

Screenshots and logs for all experiments are included.


## **🛠️ Dependencies**

Install required packages:
```
pip install numpy streamlit matplotlib
```



## **📜 Citation / Academic Use**

If referencing this project in coursework or research:

> "Pauig, Cristina Isabel. *Solving 3-SAT Using Genetic Algorithms and Wisdom of Artificial Crowds.* 2025."


## **📧 Contact**

If you have questions or want to build on this work, feel free to reach out.


