import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import PyPDF2
import docx
import json
import spacy
import nltk
from nltk.tokenize import word_tokenize
import pickle
import os

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize spaCy model
nlp = spacy.load("en_core_web_sm")

class MedicalRecordAnalyzer:
    def __init__(self):
        self.medical_terms = {}
        self.load_medical_terms()
        
    def load_medical_terms(self):
        # Comprehensive medical terms dictionary organized by category
        self.medical_terms = {
            'cardiovascular': {
                'hypertension', 'heart disease', 'coronary artery disease', 'arrhythmia',
                'heart failure', 'high blood pressure', 'atherosclerosis', 'stroke',
                'heart attack', 'myocardial infarction', 'angina', 'high cholesterol',
                'peripheral artery disease', 'deep vein thrombosis'
            },
            'endocrine': {
                'diabetes', 'type 1 diabetes', 'type 2 diabetes', 'hypothyroidism',
                'hyperthyroidism', 'graves disease', 'hashimoto\'s disease',
                'cushing\'s syndrome', 'addison\'s disease', 'metabolic syndrome',
                'insulin resistance', 'gestational diabetes'
            },
            'respiratory': {
                'asthma', 'chronic bronchitis', 'emphysema', 'copd',
                'pneumonia', 'tuberculosis', 'sleep apnea', 'pulmonary fibrosis',
                'lung cancer', 'bronchiectasis', 'cystic fibrosis'
            },
            'gastrointestinal': {
                'gastritis', 'ulcer', 'crohn\'s disease', 'ulcerative colitis',
                'ibs', 'celiac disease', 'hepatitis', 'cirrhosis',
                'gallstones', 'pancreatitis', 'gerd', 'fatty liver disease'
            },
            'musculoskeletal': {
                'arthritis', 'rheumatoid arthritis', 'osteoarthritis', 'osteoporosis',
                'fibromyalgia', 'gout', 'lupus', 'scoliosis',
                'carpal tunnel syndrome', 'tendinitis', 'bursitis'
            },
            'neurological': {
                'migraine', 'epilepsy', 'multiple sclerosis', 'parkinson\'s disease',
                'alzheimer\'s disease', 'dementia', 'neuropathy', 'brain tumor',
                'meningitis', 'encephalitis', 'bell\'s palsy'
            },
            'mental_health': {
                'depression', 'anxiety', 'bipolar disorder', 'schizophrenia',
                'ptsd', 'ocd', 'eating disorder', 'adhd',
                'autism', 'insomnia', 'panic disorder'
            },
            'immune': {
                'hiv', 'aids', 'rheumatoid arthritis', 'psoriasis',
                'multiple sclerosis', 'lupus', 'scleroderma',
                'celiac disease', 'type 1 diabetes'
            },
            'cancer': {
                'breast cancer', 'lung cancer', 'prostate cancer', 'colon cancer',
                'leukemia', 'lymphoma', 'melanoma', 'ovarian cancer',
                'pancreatic cancer', 'brain tumor', 'thyroid cancer'
            },
            'kidney': {
                'kidney disease', 'kidney stones', 'kidney failure', 'polycystic kidney disease',
                'glomerulonephritis', 'nephrotic syndrome', 'renal artery stenosis'
            }
        }
    
    def extract_text_from_pdf(self, pdf_file):
        text = ""
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    
    def extract_text_from_docx(self, docx_file):
        doc = docx.Document(docx_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    def extract_medical_info(self, text):
        doc = nlp(text.lower())  # Convert text to lowercase for better matching
        medical_info = {
            'conditions': [],
            'conditions_by_category': {},
            'medications': [],
            'lab_results': []
        }
        
        # Extract medical conditions by category
        for category, terms in self.medical_terms.items():
            found_conditions = []
            for term in terms:
                if term.lower() in text.lower():
                    found_conditions.append(term)
                    if term not in medical_info['conditions']:
                        medical_info['conditions'].append(term)
            
            if found_conditions:
                medical_info['conditions_by_category'][category] = found_conditions
        
        # Extract lab results (enhanced pattern matching)
        import re
        lab_patterns = [
            r'(\w+):\s*(\d+\.?\d*)\s*(mg/dL|g/dL|mmol/L|%|ng/mL|pg/mL|U/L|mmHg)',
            r'(\w+)\s+(\d+\.?\d*)\s*(mg/dL|g/dL|mmol/L|%|ng/mL|pg/mL|U/L|mmHg)',
            r'(\w+)\s+level:\s*(\d+\.?\d*)\s*(mg/dL|g/dL|mmol/L|%|ng/mL|pg/mL|U/L|mmHg)'
        ]
        
        for pattern in lab_patterns:
            lab_results = re.findall(pattern, text)
            medical_info['lab_results'].extend(lab_results)
        
        return medical_info
    
    def predict_health_risks(self, medical_info):
        # Enhanced risk prediction model
        risk_factors = {
            'heart_disease': 0.2,
            'diabetes': 0.2,
            'stroke': 0.2,
            'kidney_disease': 0.2,
            'cancer': 0.2,
            'respiratory_disease': 0.2,
            'mental_health': 0.2,
            'autoimmune_disease': 0.2
        }
        
        # Adjust risks based on conditions
        for category, conditions in medical_info['conditions_by_category'].items():
            if category == 'cardiovascular':
                risk_factors['heart_disease'] += 0.2
                risk_factors['stroke'] += 0.15
            elif category == 'endocrine':
                risk_factors['diabetes'] += 0.2
                risk_factors['heart_disease'] += 0.1
            elif category == 'respiratory':
                risk_factors['respiratory_disease'] += 0.2
            elif category == 'immune':
                risk_factors['autoimmune_disease'] += 0.2
            elif category == 'kidney':
                risk_factors['kidney_disease'] += 0.2
            elif category == 'cancer':
                risk_factors['cancer'] += 0.2
            elif category == 'mental_health':
                risk_factors['mental_health'] += 0.2
        
        # Adjust risks based on lab results
        for result in medical_info['lab_results']:
            test_name, value, unit = result
            test_name = test_name.lower()
            value = float(value)
            
            if 'glucose' in test_name and value > 100:
                risk_factors['diabetes'] += 0.1
            elif 'cholesterol' in test_name and value > 200:
                risk_factors['heart_disease'] += 0.1
            elif 'blood pressure' in test_name or 'bp' in test_name:
                try:
                    systolic = float(value)
                    if systolic > 130:
                        risk_factors['heart_disease'] += 0.1
                        risk_factors['stroke'] += 0.1
                except:
                    pass
        
        # Cap risk factors at 1.0
        for key in risk_factors:
            risk_factors[key] = min(risk_factors[key], 1.0)
        
        return risk_factors
    
    def generate_preventive_suggestions(self, risk_factors):
        suggestions = []
        
        if risk_factors['heart_disease'] > 0.3:
            suggestions.extend([
                "Schedule regular cardiovascular check-ups",
                "Maintain a heart-healthy diet low in saturated fats",
                "Exercise regularly with at least 150 minutes of moderate activity per week",
                "Monitor blood pressure and cholesterol levels",
                "Consider discussing aspirin therapy with your doctor"
            ])
        
        if risk_factors['diabetes'] > 0.3:
            suggestions.extend([
                "Monitor blood sugar levels regularly",
                "Maintain a balanced diet with controlled carbohydrate intake",
                "Exercise regularly to improve insulin sensitivity",
                "Schedule regular HbA1c tests",
                "Consider consulting with a diabetes educator"
            ])
        
        if risk_factors['stroke'] > 0.3:
            suggestions.extend([
                "Monitor blood pressure regularly",
                "Stay physically active",
                "Maintain a healthy weight",
                "Quit smoking if applicable",
                "Limit alcohol consumption"
            ])
        
        if risk_factors['kidney_disease'] > 0.3:
            suggestions.extend([
                "Stay well hydrated",
                "Monitor kidney function through regular check-ups",
                "Control blood pressure",
                "Limit salt intake",
                "Avoid nephrotoxic medications"
            ])
        
        if risk_factors['cancer'] > 0.3:
            suggestions.extend([
                "Schedule regular cancer screenings appropriate for your age and gender",
                "Maintain a healthy lifestyle with regular exercise",
                "Eat a diet rich in fruits, vegetables, and whole grains",
                "Avoid tobacco products",
                "Protect skin from excessive sun exposure"
            ])
        
        if risk_factors['respiratory_disease'] > 0.3:
            suggestions.extend([
                "Avoid exposure to smoke and air pollutants",
                "Use air purifiers in living spaces",
                "Practice breathing exercises",
                "Get annual flu vaccinations",
                "Monitor air quality in your area"
            ])
        
        if risk_factors['mental_health'] > 0.3:
            suggestions.extend([
                "Consider regular counseling or therapy sessions",
                "Practice stress-management techniques",
                "Maintain regular sleep patterns",
                "Build a strong support network",
                "Engage in regular physical activity"
            ])
        
        if risk_factors['autoimmune_disease'] > 0.3:
            suggestions.extend([
                "Schedule regular check-ups with a rheumatologist",
                "Maintain a balanced diet",
                "Get adequate rest",
                "Manage stress levels",
                "Consider vitamin D supplementation"
            ])
        
        return list(set(suggestions))  # Remove any duplicates

def main():
    st.title("AI-based Disease Prediction System")
    st.write("Upload medical records to analyze health risks and get preventive suggestions")
    
    analyzer = MedicalRecordAnalyzer()
    
    uploaded_file = st.file_uploader("Upload Medical Records", type=['pdf', 'txt', 'docx', 'json'])
    
    if uploaded_file is not None:
        # Process the uploaded file
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        try:
            if file_extension == '.pdf':
                text = analyzer.extract_text_from_pdf(uploaded_file)
            elif file_extension == '.docx':
                text = analyzer.extract_text_from_docx(uploaded_file)
            elif file_extension == '.txt':
                text = uploaded_file.getvalue().decode('utf-8')
            elif file_extension == '.json':
                text = json.loads(uploaded_file.getvalue().decode('utf-8'))
                text = str(text)  # Convert JSON to string for processing
            
            # Extract medical information
            medical_info = analyzer.extract_medical_info(text)
            
            # Display extracted information
            st.subheader("Extracted Medical Information")
            st.write("Conditions:", ", ".join(medical_info['conditions']))
            st.write("Lab Results:", medical_info['lab_results'])
            
            # Predict health risks
            risk_factors = analyzer.predict_health_risks(medical_info)
            
            # Display risk factors as a bar chart
            st.subheader("Health Risk Assessment")
            risk_df = pd.DataFrame({
                'Condition': list(risk_factors.keys()),
                'Risk Level': list(risk_factors.values())
            })
            fig = px.bar(risk_df, x='Condition', y='Risk Level',
                        title='Predicted Health Risks',
                        color='Risk Level',
                        color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig)
            
            # Generate and display preventive suggestions
            suggestions = analyzer.generate_preventive_suggestions(risk_factors)
            st.subheader("Preventive Health Suggestions")
            for suggestion in suggestions:
                st.write(f"• {suggestion}")
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main() 