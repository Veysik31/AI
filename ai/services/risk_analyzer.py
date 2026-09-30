"""
Risk Analyzer for Inventory Management
Identifies stockout risks and priority levels
"""

class RiskAnalyzer:
    def analyze(self, current_stock, predicted_demand, reorder_level, confidence):
        """
        Analyze inventory risk based on current stock and predicted demand
        
        Returns:
            dict: {
                'status': str - Human readable risk status
                'level': str - Risk level (critical, high, medium, low)
                'priority_score': int - Priority score for sorting (0-100)
            }
        """
        
        # Calculate stock coverage (days of stock remaining)
        if predicted_demand > 0:
            days_coverage = (current_stock / predicted_demand) * 30
        else:
            days_coverage = 999
        
        # Check if below reorder level
        below_reorder = current_stock < reorder_level
        
        # Determine risk level
        if current_stock == 0:
            level = 'critical'
            status = 'Out of stock'
            priority_score = 100
        elif days_coverage < 7:
            level = 'critical'
            status = 'Critical - Less than 1 week of stock'
            priority_score = 95
        elif days_coverage < 14 or below_reorder:
            level = 'high'
            status = 'High risk - Below reorder level'
            priority_score = 80
        elif days_coverage < 30:
            level = 'medium'
            status = 'Medium risk - Low stock projected'
            priority_score = 60
        else:
            level = 'low'
            status = 'Low risk - Adequate stock'
            priority_score = 30
        
        # Adjust priority based on confidence
        confidence_factor = confidence / 100
        priority_score = int(priority_score * confidence_factor)
        
        return {
            'status': status,
            'level': level,
            'priority_score': priority_score,
            'days_coverage': round(days_coverage, 1)
        }
