import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import re

class PhishingDetector:
    """LSTM-based phishing URL detection"""
    
    def __init__(self, model_path='ml/models/lstm_model.keras', 
                 label_encoder_path='ml/models/label_encoder.pkl',
                 scaler_path='ml/models/scaler.pkl',
                 feature_names_path='ml/models/feature_names.pkl'):
        """Load trained model and preprocessing artifacts"""
        try:
            self.model = load_model(model_path)
            self.label_encoder = joblib.load(label_encoder_path)
            self.scaler = joblib.load(scaler_path)
            self.feature_names = joblib.load(feature_names_path)
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading model: {str(e)}")
            raise
    
    def extract_features(self, url):
        """Extract 13 features from URL"""
        try:
            url_str = str(url).strip()
            features = {
                'url_length': len(url_str),
                'num_dots': url_str.count('.'),
                'num_dashes': url_str.count('-'),
                'num_underscores': url_str.count('_'),
                'num_slashes': url_str.count('/'),
                'num_at': url_str.count('@'),
                'num_question': url_str.count('?'),
                'num_equals': url_str.count('='),
                'num_ampersand': url_str.count('&'),
                'num_hash': url_str.count('#'),
                'num_percent': url_str.count('%'),
                'num_digits': sum(1 for c in url_str if c.isdigit()),
                'num_special': sum(1 for c in url_str if not c.isalnum() and c != '/'),
            }
            return features
        except Exception as e:
            print(f"✗ Error extracting features: {str(e)}")
            return {key: 0 for key in ['url_length', 'num_dots', 'num_dashes', 'num_underscores', 
                                        'num_slashes', 'num_at', 'num_question', 'num_equals', 
                                        'num_ampersand', 'num_hash', 'num_percent', 'num_digits', 'num_special']}
    
    def predict(self, url, threshold=0.5):
        """
        Predict if URL is phishing or benign
        
        Args:
            url (str): URL to analyze
            threshold (float): Decision threshold (default=0.5)
        
        Returns:
            dict: Prediction results
        """
        try:
            # Extract features
            features_dict = self.extract_features(url)
            features_array = np.array([features_dict[key] for key in self.feature_names])
            
            # Normalize
            features_scaled = self.scaler.transform(features_array.reshape(1, -1))
            
            # Reshape for LSTM (1 sample, 20 timesteps, 13 features)
            X_lstm = np.repeat(features_scaled[:, np.newaxis, :], 20, axis=1)
            
            # Predict
            probability = self.model.predict(X_lstm, verbose=0)[0][0]
            
            # Use threshold to determine class
            is_phishing = probability >= threshold
            prediction_class = 1 if is_phishing else 0
            
            # Decode label
            label = self.label_encoder.inverse_transform([prediction_class])[0]
            
            # Risk assessment
            if probability >= 0.8:
                risk_level = "CRITICAL"
            elif probability >= 0.6:
                risk_level = "HIGH"
            elif probability >= 0.4:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            return {
                'url': url,
                'is_phishing': bool(is_phishing),
                'probability': float(probability),
                'prediction_class': int(prediction_class),
                'label': str(label),
                'risk_level': risk_level,
                'confidence': float(max(probability, 1 - probability)),
                'features': features_dict
            }
        
        except Exception as e:
            return {
                'url': url,
                'error': str(e),
                'is_phishing': None,
                'probability': None,
                'risk_level': 'UNKNOWN'
            }
    
    def predict_batch(self, urls, threshold=0.5):
        """
        Predict for multiple URLs
        
        Args:
            urls (list): List of URLs
            threshold (float): Decision threshold
        
        Returns:
            list: List of prediction results
        """
        return [self.predict(url, threshold) for url in urls]
    
    def explain_prediction(self, url):
        """
        Explain what features contributed to the prediction
        
        Args:
            url (str): URL to analyze
        
        Returns:
            dict: Detailed analysis
        """
        prediction = self.predict(url)
        features = prediction.get('features', {})
        
        # Identify high-risk features
        high_risk_features = {
            'num_at': 'Multiple @ symbols (phishing indicator)',
            'num_question': 'Multiple query parameters',
            'num_equals': 'Multiple = signs in URL',
            'num_percent': 'URL encoding detected',
            'url_length': 'Unusually long URL',
            'num_underscores': 'Underscores in domain',
            'num_dashes': 'Multiple hyphens (domain mimicking)'
        }
        
        risk_factors = []
        for feature, risk_description in high_risk_features.items():
            if features.get(feature, 0) > 0:
                risk_factors.append({
                    'feature': feature,
                    'value': features[feature],
                    'risk_description': risk_description
                })
        
        return {
            'url': url,
            'prediction': prediction,
            'risk_factors': sorted(risk_factors, key=lambda x: x['value'], reverse=True),
            'summary': f"URL classified as {prediction['label'].upper()} with {prediction['risk_level']} risk"
        }


def test_detector():
    """Test the detector with sample URLs"""
    detector = PhishingDetector()
    
    test_urls = [
        'https://www.google.com',
        'https://www.facebook.com',
        'https://www.github.com',
        'http://verify-your-account-click-here.xyz',
        'https://account-update-required-confirm-password.suspicious.com',
        'https://apple-id-verify.site/login?user=john@email.com&redirect=%2F',
    ]
    
    print("\n" + "="*80)
    print("PHISHING DETECTOR TEST")
    print("="*80)
    
    for url in test_urls:
        result = detector.predict(url)
        print(f"\n🔗 URL: {url}")
        print(f"   Label: {result['label'].upper()}")
        print(f"   Risk Level: {result['risk_level']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"   Probability: {result['probability']:.4f}")


if __name__ == "__main__":
    # Initialize detector
    detector = PhishingDetector()
    
    # Test with sample URLs
    test_detector()
