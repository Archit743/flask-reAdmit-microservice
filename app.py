import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re
import requests
import tempfile
import PyPDF2
import io
import os
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load the Random Forest model
rf_model = joblib.load('random_forest_model(2).pkl')

def extract_text_from_pdf_url(pdf_url):
    """
    Extract text from a PDF file at the provided URL.
    
    Args:
        pdf_url (str): URL to the PDF file
        
    Returns:
        str: Extracted text from the PDF
    """
    try:
        # Download the PDF from the URL
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()  # Raise an exception for bad responses
        
        # Create a file-like object from the content
        pdf_file = io.BytesIO(response.content)
        
        # Use PyPDF2 to read the file
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        # Extract text from all pages
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
            
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

def parse_medical_data_from_text(text):
    """
    Parse the structured medical data from the extracted text.
    This parser is designed for the specific format shown in the example.
    
    Args:
        text (str): Text extracted from the PDF
        
    Returns:
        dict: Dictionary containing parsed data
    """
    if not text:
        return None
    
    data = {}
    
    # Pattern to match key-value pairs like "'feature_name': value"
    pattern = r"'([^']+)':\s*(\d+(?:\.\d+)?)"
    
    # Find all matches in the text
    matches = re.findall(pattern, text)
    
    # Convert matches to dictionary entries
    for key, value in matches:
        # Try to convert to appropriate numeric type (int or float)
        try:
            # If value contains a decimal point, convert to float, otherwise int
            if '.' in value:
                data[key] = float(value)
            else:
                data[key] = int(value)
        except ValueError:
            # If conversion fails, keep as string
            data[key] = value
    
    print(f"Extracted {len(data)} features from PDF")
    return data

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    if rf_model is not None:
        return jsonify({"status": "healthy", "model_loaded": True})
    return jsonify({"status": "unhealthy", "model_loaded": False}), 503

@app.route('/predict', methods=['POST'])
def predict():
    # Get the PDF URL from the request
    data = request.json
    
    # Extract and print the URL
    pdf_url = data.get('pdfUrl')
    print(f"Received PDF URL: {pdf_url}")
    
    # Default data to use if extraction fails
    default_data = {
        'age': 25,
        'time_in_hospital': 2,
        'num_procedures': 1,
        'num_medications': 1,
        'number_outpatient_log1p': 1.0,
        'number_emergency_log1p': 0.0,
        'number_inpatient_log1p': 0.7,
        'number_diagnoses': 1,
        'metformin': 1,
        'repaglinide': 0,
        'nateglinide': 0,
        'chlorpropamide': 0,
        'glimepiride': 0,
        'glipizide': 1,
        'glyburide': 0,
        'pioglitazone': 0,
        'rosiglitazone': 0,
        'acarbose': 0,
        'tolazamide': 0,
        'insulin': 1,
        'glyburide-metformin': 0,
        'AfricanAmerican': 0,
        'Asian': 1,
        'Caucasian': 0,
        'Hispanic': 0,
        'Other': 0,
        'gender_1': 1,
        'admission_type_id_3': 0,
        'admission_type_id_5': 1,
        'discharge_disposition_id_2': 0,
        'discharge_disposition_id_7': 0,
        'discharge_disposition_id_10': 0,
        'discharge_disposition_id_18': 0,
        'admission_source_id_4': 0,
        'admission_source_id_7': 1,
        'admission_source_id_9': 0,
        'max_glu_serum_1.0': 1,
        'A1Cresult_1': 1,
        'level1_diag1_1.0': 1,
        'level1_diag1_2.0': 0,
        'level1_diag1_3.0': 0,
        'level1_diag1_4.0': 0,
        'level1_diag1_5.0': 0,
        'level1_diag1_6.0': 0,
        'level1_diag1_7.0': 0,
        'level1_diag1_8.0': 0
    }
    
    # Try to extract and parse data from the PDF
    use_default = True
    extraction_info = ""
    missing_features = []
    extracted_features_count = 0
    
    if pdf_url:
        # Extract text from PDF
        pdf_text = extract_text_from_pdf_url(pdf_url)
        
        if pdf_text:
            # Parse data from the text
            parsed_data = parse_medical_data_from_text(pdf_text)
            
            if parsed_data and len(parsed_data) > 0:
                extracted_features_count = len(parsed_data)
                
                # Check what features are expected by the model
                if hasattr(rf_model, 'feature_names_in_'):
                    expected_features = rf_model.feature_names_in_
                    
                    # Check which expected features are missing from parsed data
                    missing_features = [feat for feat in expected_features if feat not in parsed_data]
                    print(f"Missing features: {missing_features}")
                    
                    # Merge parsed data with default data (for missing fields)
                    prediction_data = default_data.copy()
                    for key, value in parsed_data.items():
                        prediction_data[key] = value
                    
                    extraction_info = f"Successfully extracted {extracted_features_count} features from PDF. Missing {len(missing_features)} features."
                    use_default = False
                else:
                    # If we don't know what features the model expects, use parsed data where possible
                    prediction_data = default_data.copy()
                    for key, value in parsed_data.items():
                        prediction_data[key] = value
                    extraction_info = f"Extracted {extracted_features_count} features from PDF, but model feature names unknown."
                    use_default = False
            else:
                extraction_info = "Extracted text from PDF but couldn't parse required fields."
                prediction_data = default_data
        else:
            extraction_info = "Failed to extract text from PDF."
            prediction_data = default_data
    else:
        extraction_info = "No PDF URL provided."
        prediction_data = default_data
    
    # Prepare the input for the model
    if hasattr(rf_model, 'feature_names_in_'):
        expected_features = rf_model.feature_names_in_
        # Ensure input matches the expected features
        input_values = [[prediction_data.get(feature, 0) for feature in expected_features]]
        print(f"Using expected features: {expected_features}")
    else:
        # Fallback: select the first 48 features (less reliable)
        print("Feature names not available, selecting first 48 features as fallback.")
        input_values = [list(prediction_data.values())[:rf_model.n_features_in_]]

    # Debugging: print feature counts
    print(f"Number of features expected: {rf_model.n_features_in_}")
    print(f"Number of features provided: {len(input_values[0])}")

    # Get probabilities
    probabilities = rf_model.predict_proba(input_values)[0]

    # Convert to percentage format (assuming class 1 is readmission risk)
    probability_percentages = {
        "readmissionRisk": probabilities[1] * 100,  # Convert to percentage
        "success": True,
        "receivedUrl": pdf_url,  # Include the URL in the response for confirmation
        "dataSource": "default" if use_default else "pdf_extracted",
        "extractionInfo": extraction_info,
        "extractedFeatures": extracted_features_count,
        "missingFeatures": len(missing_features) if hasattr(rf_model, 'feature_names_in_') else "unknown"
    }
    print(f"Prediction: {probability_percentages}")
    return jsonify(probability_percentages)

if __name__ == '__main__':
    app.run(debug=True, port=4000)