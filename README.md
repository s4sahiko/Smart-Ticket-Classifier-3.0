# Smart-Ticket-Classifier 3.0

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Repo Size](https://img.shields.io/github/repo-size/s4sahiko/Ai-Ticket-Classifier)
![Last Commit](https://img.shields.io/github/last-commit/s4sahiko/Ai-Ticket-Classifier)
![Dashboard Preview](assests/dashboard.png)


---

## Overview  
**Smart-Ticket-Classifier** is an AI-powered system that automatically categorises customer support tickets using Natural Language Processing (NLP) and Transformer models.  
It helps support teams reduce manual triage, improve response times, and optimize workflows through intelligent automation.  
Integrated with Slack for Real Time Communication and Backend Support.

###  Key Features  
- Automated classification of support tickets  
- Transformer-based NLP model for high accuracy  
- Real-time or batch prediction support  
- Interactive GUI dashboard  
- Knowledge-base embeddings and analytics
- Shows Real time Solutions for the tickets
- Content gap detection and performance reports
- **Analytics Part for the improvement**
- **Real time Screen Optimization**
- **Integrated Slack for Real time Backend support**

---
# What's New?

1. ## Analytics Part for the improvement
     ![Solution Preview](assests/analytics.png)
2. ## Real time Screen Optimization
   ![Filter Preview](assests/rtso.png)
3. ## Integrated Slack for Real time Backend support
    ![Filter Preview](assests/slack.png)
   
---

## ⚙️ Installation  

1. **Clone the Repository**

       git clone https://github.com/s4sahiko/Smart-Ticket-Classifier.git
       cd Smart-Ticket-Classifier
   
2. **Create and Activate a Virtual Environment**

       python3 -m venv venv
       source venv/bin/activate     # For Linux/Mac
       venv\Scripts\activate        # For Windows

3. **Install Dependencies**
    
       pip install -r requirements.txt
    
4. **Prepare Dataset**
Use the provided **cleaned_ticket_data.csv**,
Or replace it with your own dataset and update file paths in the scripts.

5. **(Optional) Train the Model**

       python trainer.py

7. **Run the Application By running**

       python app.py

9. **Open the other terminal and Run to open the GUI Dashboard**

       streamlit run gui_dashboard.py

**Uplaod the ticket and Enjoy the saved time!**

---
Training
---

**Fine-tune the Transformer model using**
    
    python trainer.py

You can customize hyperparameters (learning rate, epochs, etc.) inside the script.

---
Classification
---

**After training, classify tickets in real time**
    
    python app.py

It reads ticket text and outputs predicted categories.

---
Dashboard & Reporting
---

**Generate performance reports and content gap analysis**
    
    python reporting_dashboard.py
    
Optionally, use the **knowledge_base_with_embeddings.csv** file to enhance classification accuracy.

---
Example Workflow
---

1. A new support ticket is received.

2. data_connector.py processes and cleans it.

3. categorizer.py uses the trained model to predict the category.

4. The result appears in the GUI or CLI With **Real Time Solution Recommendation**.

5. Analytics and reports are generated automatically (**Improved**).

---
Tech Stack
---

**Language**: Python

**Libraries**: Transformers, Pandas, NumPy, Matplotlib etc.

**Frameworks**: PyTorch

**Visualization**: Streamlit

**Data**: CSV-based datasets and embeddings

---
Why Use This Project
---

**Efficiency** — Automates ticket routing

**Scalability** — Handles large datasets easily

**Accuracy** — Uses Transformer-based contextual understanding

**Insights** — Detects knowledge gaps and tracks model performance

---
Contact & Support
---

For questions, bug reports, or suggestions — open an issue on the GitHub repository.

👤 Author: **INFOAI4(Sahil,Rohit K,Sirisha,V.Raghu)**

    
 
