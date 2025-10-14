# codealpha_tasks

🎯 Overview
The Netflix Customer Churn Prediction System is a full-stack machine learning application designed to predict whether a customer will churn or stay based on their subscription behavior, engagement patterns, and account activity. The system analyzes 20+ engineered features to provide accurate predictions and personalized retention recommendations.

Key Highlights:
✅ 96.40% Prediction Accuracy

✅ 0.9521 ROC-AUC Score

✅ 20+ Engineered Features

✅ 5,000+ Training Records

✅ Modern Responsive UI

✅ Real-time Predictions

✨ Features
🤖 Machine Learning
High-Performance Model: XGBoost classifier with optimized hyperparameters

Advanced Feature Engineering: 20+ derived features including engagement scores, payment reliability, and activity patterns

Comprehensive Analysis: Subscription behavior, content consumption, device usage, and support interactions

Risk Classification: Automatic categorization into Low, Medium, and High risk levels

🌐 Web Application
Interactive Form: User-friendly input interface with validation

Real-time Predictions: Instant churn probability calculation

Probability Visualization: Animated progress bars showing churn vs. retention likelihood

Retention Recommendations: Personalized strategies based on customer behavior

Responsive Design: Mobile-friendly interface with modern UI/UX

🎨 UI/UX Design
Soothing Color Palette: Light, calming gradient backgrounds

Gradient Animations: Modern animated predict button with shimmer effects

Smooth Transitions: Professional animations throughout the application

Accessibility: Focus states, proper contrast, and intuitive navigation

🎬 Demo
Prediction Flow
Enter customer details (subscription type, usage patterns, engagement metrics)

Click the animated predict button

View real-time churn probability with visual indicators

Receive personalized retention recommendations

🛠️ Tech Stack
Backend
Python 3.8+

Flask 2.0+ - Web framework

XGBoost - Machine learning model

Scikit-Learn - ML utilities and preprocessing

Pandas & NumPy - Data manipulation

Frontend
HTML5 - Structure

CSS3 - Styling with gradient animations

Machine Learning
XGBoost Classifier (200 estimators, 0.1 learning rate)

StandardScaler - Feature normalization

Feature Engineering - 20+ derived metrics

📦 Installation
Prerequisites
Python 3.8 or higher

🚀 Usage
Making Predictions
Enter Account Information

Subscription Type (Basic/Standard/Premium)

Account Age (months)

Content Watched (hours)

Provide Usage Patterns

Devices Used (1-5+)

Profiles Created (1-5)

Days Since Last Login

Share Engagement Metrics

Genre Diversity (1-15)

Download Count

Peak Viewing Time

Include Support Data

Support Tickets

Payment Issues

Click Predict and view results with:

Churn probability percentage

Retention probability

Risk level classification

Personalized recommendations

📊 Model Performance
Metrics
Metric	Score
Accuracy	96.40%
ROC-AUC	0.9521
Precision	95.8%
Recall	94.2%
F1-Score	95.0%
Feature Importance
Top features contributing to predictions:

Account Health Score

Engagement Score

Payment Reliability

Activity Score

Days Since Last Login

Content Watch Hours per Month

Device Usage Patterns

Support Ticket Rate

Training Details
Dataset Size: 5,000 customer records

Features: 20+ engineered features

Model: XGBoost (200 estimators)

Training Time: ~5 minutes

Cross-Validation: 5-fold CV

📁 Project Structure
text
netflix-churn-prediction/
│
├── app.py                      # Flask application
├── train_model.py              # Model training script
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── models/                     # Saved models
│   ├── netflix_churn_model.pkl
│   ├── netflix_scaler.pkl
│   ├── feature_columns.pkl
│   └── model_info.pkl
│
├── data/                       # Dataset
│   └── netflix_customer_churn.csv
│
├── templates/                  # HTML templates
│   ├── index.html             # Home page
│   ├── result.html            # Results page
│   └── about.html             # About page
│
└── static/                     # Static files
    └── style.css              # Stylesheet

Application Settings
Edit app.py to modify:

👨‍💻 Author
Lakshya Khatri

🙏 Acknowledgments
Dataset inspired by Netflix customer behavior patterns

XGBoost library for high-performance ML

Flask framework for seamless web deployment

UI design inspired by modern web trends

📈 Future Enhancements
 Add more ML models (Random Forest, Neural Networks)

 Implement A/B testing for retention strategies

 Add user authentication and dashboard

 Deploy to cloud platform (AWS/Heroku)

 Integrate real-time data streaming

 Add multi-language support

 Create mobile application

Made with ❤️ by Lakshya Khatri
