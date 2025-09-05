"""
User Evaluation System
Collects and analyzes user feedback for system evaluation
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import json
import os
from typing import Dict, List, Optional  # Fixed: Added all necessary imports

class UserEvaluationSystem:
    """System for collecting user feedback and evaluation metrics"""
    
    def __init__(self):
        self.feedback_data = []
        self.feedback_file = 'data/user_feedback.json'
        self.ensure_data_directory()
        self.load_feedback()
    
    def ensure_data_directory(self):
        """Ensure data directory exists"""
        os.makedirs('data', exist_ok=True)
    
    def load_feedback(self):
        """Load existing feedback data"""
        try:
            if os.path.exists(self.feedback_file):
                with open(self.feedback_file, 'r') as f:
                    for line in f:
                        try:
                            self.feedback_data.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"Error loading feedback: {e}")
            self.feedback_data = []
    
    def render_feedback_form(self):
        """Render feedback collection form in Streamlit"""
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📝 Feedback")
        
        with st.sidebar.form("user_feedback", clear_on_submit=True):
            # Explanation Clarity (Likert Scale 1-5)
            clarity = st.slider(
                "How clear was the explanation?",
                1, 5, 3,
                help="1=Very Unclear, 5=Very Clear"
            )
            
            # Trust in Recommendation
            trust = st.slider(
                "How much do you trust this recommendation?",
                1, 5, 3,
                help="1=No Trust, 5=Complete Trust"
            )
            
            # Learning Value
            learning = st.slider(
                "Did you learn something new?",
                1, 5, 3,
                help="1=Nothing, 5=Learned a lot"
            )
            
            # Actionability
            actionable = st.slider(
                "How actionable was the advice?",
                1, 5, 3,
                help="1=Not Actionable, 5=Very Actionable"
            )
            
            # Overall Satisfaction
            satisfaction = st.slider(
                "Overall satisfaction",
                1, 5, 3,
                help="1=Very Dissatisfied, 5=Very Satisfied"
            )
            
            # Comments
            comments = st.text_area("Additional comments (optional)", height=50)
            
            # Submit button
            submitted = st.form_submit_button("Submit Feedback", use_container_width=True)
            
            if submitted:
                feedback = {
                    'timestamp': datetime.now().isoformat(),
                    'clarity': clarity,
                    'trust': trust,
                    'learning': learning,
                    'actionable': actionable,
                    'satisfaction': satisfaction,
                    'comments': comments,
                    'session_id': st.session_state.get('session_id', 'unknown')
                }
                self.save_feedback(feedback)
                st.success("✅ Thank you for your feedback!")
    
    def save_feedback(self, feedback: dict):
        """Save feedback to file"""
        try:
            self.feedback_data.append(feedback)
            
            # Save to file
            with open(self.feedback_file, 'a') as f:
                json.dump(feedback, f)
                f.write('\n')
        except Exception as e:
            print(f"Error saving feedback: {e}")
    
    def generate_evaluation_report(self) -> Dict:
        """Generate evaluation metrics report"""
        if not self.feedback_data:
            return {
                'total_responses': 0,
                'avg_clarity': 0,
                'avg_trust': 0,
                'avg_learning': 0,
                'avg_actionable': 0,
                'avg_satisfaction': 0,
                'clarity_success_rate': 0,
                'trust_success_rate': 0,
                'learning_success_rate': 0,
                'meets_clarity_target': False,
                'meets_trust_target': False,
                'sus_score': 0
            }
        
        df = pd.DataFrame(self.feedback_data)
        
        # Calculate metrics safely
        report = {}
        
        # Basic counts
        report['total_responses'] = len(df)
        
        # Average scores
        for metric in ['clarity', 'trust', 'learning', 'actionable', 'satisfaction']:
            if metric in df.columns:
                report[f'avg_{metric}'] = df[metric].mean()
            else:
                report[f'avg_{metric}'] = 0
        
        # Success rates (4-5 ratings)
        for metric in ['clarity', 'trust', 'learning']:
            if metric in df.columns:
                report[f'{metric}_success_rate'] = (df[metric] >= 4).mean() * 100
            else:
                report[f'{metric}_success_rate'] = 0
        
        # Targets
        report['meets_clarity_target'] = report.get('clarity_success_rate', 0) >= 80
        report['meets_trust_target'] = report.get('trust_success_rate', 0) >= 80
        
        # SUS score
        report['sus_score'] = self.calculate_sus_score(df)
        
        return report
    
    def calculate_sus_score(self, df: pd.DataFrame) -> float:
        """Calculate System Usability Scale score"""
        # Simplified SUS calculation based on available metrics
        positive_items = ['satisfaction', 'trust', 'actionable']
        
        score = 0
        count = 0
        
        for item in positive_items:
            if item in df.columns and not df[item].isna().all():
                # Convert 1-5 scale to SUS contribution (0-4) * 2.5
                score += (df[item].mean() - 1) * 2.5
                count += 1
        
        # Normalize to 0-100 scale
        if count > 0:
            return (score / count) * 10
        return 0
    
    def get_feedback_summary(self) -> str:
        """Get a text summary of feedback"""
        report = self.generate_evaluation_report()
        
        if report['total_responses'] == 0:
            return "No feedback collected yet."
        
        summary = f"""
        **Feedback Summary** (n={report['total_responses']})
        
        📊 **Average Ratings:**
        - Clarity: {report['avg_clarity']:.1f}/5
        - Trust: {report['avg_trust']:.1f}/5
        - Learning: {report['avg_learning']:.1f}/5
        - Actionable: {report['avg_actionable']:.1f}/5
        - Satisfaction: {report['avg_satisfaction']:.1f}/5
        
        ✅ **Success Metrics:**
        - Clarity Success Rate: {report['clarity_success_rate']:.1f}%
        - Trust Success Rate: {report['trust_success_rate']:.1f}%
        - Learning Success Rate: {report['learning_success_rate']:.1f}%
        
        🎯 **Targets:**
        - Meets Clarity Target (≥80%): {'✅' if report['meets_clarity_target'] else '❌'}
        - Meets Trust Target (≥80%): {'✅' if report['meets_trust_target'] else '❌'}
        
        📈 **System Usability Score:** {report['sus_score']:.1f}/100
        """
        
        return summary
    
    def export_feedback_data(self, filepath: str = 'data/feedback_export.csv'):
        """Export feedback data to CSV"""
        if self.feedback_data:
            df = pd.DataFrame(self.feedback_data)
            df.to_csv(filepath, index=False)
            return True
        return False