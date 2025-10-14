from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'netflix-churn-prediction-2025'

model = None
scaler = None
feature_columns = None
label_encoders = None
model_info = {'model_name': 'Not Loaded', 'accuracy': 0, 'n_features': 0, 'n_samples': 0}
startup_error = None

print("Initializing app and loading metadata (no heavy model load)...")
try:
    try:
        with open('models/feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
    except Exception:
        try:
            with open('models/feature_columns.pkl', 'rb') as f:
                feature_columns = pickle.load(f)
        except Exception:
            feature_columns = None
    
    try:
        with open('models/label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
    except Exception:
        label_encoders = None
    
    try:
        with open('models/model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
    except Exception:
        pass
    
    print('✓ Metadata loaded (feature list / encoders / model_info if available)')
except Exception as e:
    startup_error = str(e)
    print('! Warning loading metadata:', startup_error)

def engineer_features(data):
    """Apply feature engineering"""
    df = data.copy()
    df['TotalRevenue'] = df['MonthlyRevenue'] * df['AccountAge']
    df['RevenuePerMonth'] = df['MonthlyRevenue'] / (df['AccountAge'] + 1)
    df['RevenueCategory'] = pd.cut(df['MonthlyRevenue'], bins=[0, 10, 16, 25], labels=[0, 1, 2])
    df['WatchHoursPerMonth'] = df['ContentWatchedHours'] / (df['AccountAge'] + 1)
    df['IsActiveUser'] = (df['WatchHoursPerMonth'] > 10).astype(int)
    df['EngagementScore'] = (df['ContentWatchedHours'] * df['GenreDiversity']) / (df['AccountAge'] + 1)
    df['DevicesPerProfile'] = df['DevicesUsed'] / (df['ProfilesCreated'] + 1)
    df['IsMultiDevice'] = (df['DevicesUsed'] > 1).astype(int)
    df['IsMultiProfile'] = (df['ProfilesCreated'] > 1).astype(int)
    df['ActivityScore'] = 1 / (df['DaysSinceLastLogin'] + 1)
    df['IsRecentUser'] = (df['DaysSinceLastLogin'] <= 7).astype(int)
    df['IsInactive'] = (df['DaysSinceLastLogin'] > 30).astype(int)
    df['SupportRate'] = df['SupportTickets'] / (df['AccountAge'] + 1)
    df['HasSupportIssues'] = (df['SupportTickets'] > 0).astype(int)
    df['HasPaymentIssues'] = (df['PaymentIssues'] > 0).astype(int)
    df['PaymentReliability'] = 1 / (df['PaymentIssues'] + 1)
    df['DownloadRate'] = df['DownloadCount'] / (df['AccountAge'] + 1)
    df['GenreDiversityScore'] = df['GenreDiversity'] / 15
    df['ContentEngagement'] = df['ContentWatchedHours'] * df['GenreDiversity']
    max_engagement = df['EngagementScore'].max()
    if max_engagement > 0:
        df['AccountHealthScore'] = (
            df['ActivityScore'] * 0.3 + 
            df['PaymentReliability'] * 0.3 + 
            df['EngagementScore'] / max_engagement * 0.4
        )
    else:
        df['AccountHealthScore'] = (
            df['ActivityScore'] * 0.3 + 
            df['PaymentReliability'] * 0.3
        )
    return df

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html', model_info=model_info)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle predictions"""
    global model, scaler, feature_columns
    
    try:
        if model is None or scaler is None:
            try:
                try:
                    with open('models/netflix_churn_model.pkl', 'rb') as f:
                        model = pickle.load(f)
                    with open('models/netflix_scaler.pkl', 'rb') as f:
                        scaler = pickle.load(f)
                except Exception:
                    with open('models/demo_model.pkl', 'rb') as f:
                        model = pickle.load(f)
                    with open('models/demo_scaler.pkl', 'rb') as f:
                        scaler = pickle.load(f)
            except Exception as e:
                err = (
                    "Failed to load model or scaler. This often means required Python packages "
                    "(scikit-learn, xgboost, scipy, etc.) are not installed in this environment, "
                    f"or the model file is incompatible. Error: {e}"
                )
                return render_template('index.html', error=err, model_info=model_info, form_data=request.form)
        
        # Get actual values from form - removed default values for devices and profiles
        try:
            subscription_type = int(request.form.get('subscription_type', 1))
            account_age = int(request.form.get('account_age', 12))
            content_watched = float(request.form.get('content_watched', 50))
            
            # Get devices_used and profiles without hardcoded defaults
            devices_used_str = request.form.get('devices_used')
            profiles_str = request.form.get('profiles') or request.form.get('profiles_created')
            
            devices_used = int(devices_used_str) if devices_used_str else 1
            profiles_created = int(profiles_str) if profiles_str else 1
            
            support_tickets = int(request.form.get('support_tickets', 0))
            payment_issues = int(request.form.get('payment_issues', 0))
            days_since_login = int(request.form.get('days_since_login', 3))
            genre_diversity = int(request.form.get('genre_diversity', 8))
            download_count = int(request.form.get('download_count', 5))
            peak_viewing = int(request.form.get('peak_viewing', 1))
        except ValueError as e:
            return render_template('index.html',
                                 error=f"Invalid input: {str(e)}",
                                 model_info=model_info,
                                 form_data=request.form)
        
        monthly_revenue = {1: 9.99, 2: 15.49, 3: 19.99}.get(subscription_type, 15.49)
        
        input_data = pd.DataFrame({
            'SubscriptionType': [subscription_type],
            'MonthlyRevenue': [monthly_revenue],
            'AccountAge': [account_age],
            'ContentWatchedHours': [content_watched],
            'DevicesUsed': [devices_used],
            'ProfilesCreated': [profiles_created],
            'SupportTickets': [support_tickets],
            'PaymentIssues': [payment_issues],
            'DaysSinceLastLogin': [days_since_login],
            'GenreDiversity': [genre_diversity],
            'DownloadCount': [download_count],
            'PeakViewingHours': [peak_viewing]
        })
        
        input_engineered = engineer_features(input_data)
        
        if feature_columns is None:
            try:
                with open('models/demo_feature_columns.pkl', 'rb') as f:
                    feature_columns = pickle.load(f)
            except Exception:
                feature_columns = list(input_engineered.columns)
        
        X_input = input_engineered.reindex(columns=feature_columns, fill_value=0)
        X_scaled = scaler.transform(X_input)
        
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0]
        
        churn_status = "Will Churn" if prediction == 1 else "Will Not Churn"
        churn_probability = probability[1] * 100
        retention_probability = probability[0] * 100
        
        if churn_probability < 30:
            risk_level, risk_class = "Low Risk", "low"
        elif churn_probability < 60:
            risk_level, risk_class = "Medium Risk", "medium"
        else:
            risk_level, risk_class = "High Risk", "high"
        
        recommendations = generate_recommendations(input_data, churn_probability, risk_level)
        
        result = {
            'churn': int(prediction),
            'churn_probability': float(churn_probability),
            'stay_probability': float(retention_probability),
            'churn_status': churn_status,
            'risk_level': risk_level,
            'risk_class': risk_class,
            'recommendations': recommendations,
            'customer_data': {
                'Subscription Type': ['Basic', 'Standard', 'Premium'][subscription_type-1],
                'Monthly Revenue': f"${monthly_revenue}",
                'Account Age': f"{account_age} months",
                'Content Watched': f"{content_watched} hours",
                'Devices Used': devices_used,
                'Profiles': profiles_created
            },
            'input_data': {
                'SubscriptionType': subscription_type,
                'MonthlyRevenue': monthly_revenue,
                'AccountAge': account_age,
                'ContentWatchedHours': content_watched,
                'DevicesUsed': devices_used,
                'ProfilesCreated': profiles_created,
                'SupportTickets': support_tickets,
                'PaymentIssues': payment_issues,
                'DaysSinceLastLogin': days_since_login,
            }
        }
        
        return render_template('result.html', result=result, model_info=model_info)
        
    except Exception as e:
        return render_template('index.html',
                             error=f"Prediction error: {str(e)}",
                             model_info=model_info,
                             form_data=request.form)

def generate_recommendations(customer_data, churn_prob, risk_level):
    """Generate recommendations"""
    recommendations = []
    data = customer_data.iloc[0]
    
    if data['ContentWatchedHours'] < 20:
        recommendations.append("📺 Recommend personalized content based on viewing history")
    
    if data['DevicesUsed'] == 1:
        recommendations.append("📱 Encourage multi-device usage with app download promotions")
    
    if data['ProfilesCreated'] == 1:
        recommendations.append("👥 Promote family plan benefits and additional profiles")
    
    if data['SupportTickets'] > 2:
        recommendations.append("🎯 Priority customer service and proactive support outreach")
    
    if data['PaymentIssues'] > 0:
        recommendations.append("💳 Offer flexible payment options or payment plan assistance")
    
    if data['DaysSinceLastLogin'] > 30:
        recommendations.append("⏰ Send re-engagement email with new content highlights")
    
    if data['SubscriptionType'] == 1 and churn_prob > 50:
        recommendations.append("⭐ Offer limited-time upgrade discount to Standard plan")
    
    if risk_level == "High Risk":
        recommendations.append("🎁 Provide exclusive retention offer or discount")
        recommendations.append("📞 Schedule personalized retention call")
    
    if not recommendations:
        recommendations.append("✅ Customer appears stable - continue regular engagement")
    
    return recommendations

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html', model_info=model_info)

if __name__ == '__main__':
    print("\n" + "="*80)
    print(" "*20 + "NETFLIX CHURN PREDICTION APP")
    print("="*80)
    print(f"\n🚀 Starting Flask application...")
    print(f"📊 Model: {model_info.get('model_name', 'Unknown')}")
    print(f"✅ Accuracy: {model_info.get('accuracy', 0)*100:.2f}%")
    print(f"\n🌐 Access at: http://127.0.0.1:5000")
    print("="*80 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
