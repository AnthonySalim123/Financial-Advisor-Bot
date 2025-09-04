# CREATE this file for user testing functionality
import streamlit as st
import pandas as pd
from datetime import datetime
import json

class UserEvaluationSystem:
    """System for collecting user feedback and evaluation metrics"""
    
    def __init__(self):
        self.feedback_data = []
        self.load_feedback()
    
    def render_feedback_form(self):
        """Render feedback collection form in Streamlit"""
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📝 Feedback")
        
        with st.sidebar.form("user_feedback"):
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
            comments = st.text_area("Additional comments (optional)")
            
            # Submit button
            submitted = st.form_submit_button("Submit Feedback")
            
            if submitted:
                self.save_feedback({
                    'timestamp': datetime.now().isoformat(),
                    'clarity': clarity,
                    'trust': trust,
                    'learning': learning,
                    'actionable': actionable,
                    'satisfaction': satisfaction,
                    'comments': comments,
                    'session_id': st.session_state.get('session_id', 'unknown')
                })
                st.success("Thank you for your feedback!")
    
    def save_feedback(self, feedback: dict):
        """Save feedback to database"""
        self.feedback_data.append(feedback)
        
        # Save to file
        with open('data/user_feedback.json', 'a') as f:
            json.dump(feedback, f)
            f.write('\n')
    
    def generate_evaluation_report(self) -> Dict:
        """Generate evaluation metrics report"""
        if not self.feedback_data:
            return {}
        
        df = pd.DataFrame(self.feedback_data)
        
        report = {
            'total_responses': len(df),
            'avg_clarity': df['clarity'].mean(),
            'avg_trust': df['trust'].mean(),
            'avg_learning': df['learning'].mean(),
            'avg_actionable': df['actionable'].mean(),
            'avg_satisfaction': df['satisfaction'].mean(),
            
            # Success metrics
            'clarity_success_rate': (df['clarity'] >= 4).mean() * 100,
            'trust_success_rate': (df['trust'] >= 4).mean() * 100,
            'learning_success_rate': (df['learning'] >= 4).mean() * 100,
            
            # Target: ≥80% rate explanations 4-5
            'meets_clarity_target': (df['clarity'] >= 4).mean() >= 0.8,
            'meets_trust_target': (df['trust'] >= 4).mean() >= 0.8,
            
            # System Usability Scale
            'sus_score': self.calculate_sus_score(df)
        }
        
        return report
    
    def calculate_sus_score(self, df):
        """Calculate System Usability Scale score"""
        # Simplified SUS calculation
        positive_items = ['satisfaction', 'trust', 'actionable']
        negative_items = []  # Add if you have negative questions
        
        score = 0
        for item in positive_items:
            if item in df.columns:
                score += (df[item].mean() - 1) * 2.5
        
        return score * (100 / (len(positive_items) * 10))