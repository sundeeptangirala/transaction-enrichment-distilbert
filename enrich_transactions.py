#!/usr/bin/env python3
"""
Transaction Enrichment Pipeline with DistilBERT

Enriches credit card transactions with:
- Transaction cleaning and normalization
- Merchant category classification using DistilBERT
- MCC code assignment
- Industry categorization
- Enhanced merchant information
"""

import pandas as pd
import numpy as np
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import pipeline


class TransactionCleaner:
    """Cleans and normalizes transaction descriptions"""
    
    def __init__(self):
        # Common patterns to remove from transaction descriptions
        self.patterns_to_remove = [
            r'\d{10,}',  # Long numbers (likely transaction IDs)
            r'#\d+',     # Reference numbers
            r'\*+\d+',   # Masked card numbers
            r'\b[A-Z]{2}\d+[A-Z]*\b',  # Codes
        ]
        
    def clean_description(self, description: str) -> str:
        """Clean transaction description"""
        if pd.isna(description):
            return ""
            
        cleaned = description.upper().strip()
        
        # Remove patterns
        for pattern in self.patterns_to_remove:
            cleaned = re.sub(pattern, '', cleaned)
            
        # Remove extra spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Remove special characters except spaces and dashes
        cleaned = re.sub(r'[^A-Z0-9\s\-]', '', cleaned)
        
        return cleaned
    
    def extract_merchant_name(self, description: str) -> str:
        """Extract merchant name from description"""
        cleaned = self.clean_description(description)
        
        # Split and take first meaningful part
        parts = cleaned.split()
        if len(parts) > 0:
            # Remove common prefixes
            prefixes = ['POS', 'PURCHASE', 'PAYMENT', 'DEBIT']
            merchant_parts = [p for p in parts if p not in prefixes]
            return ' '.join(merchant_parts[:3]) if merchant_parts else cleaned
        
        return cleaned


class MCCMapper:
    """Maps merchant categories to MCC codes and industries"""
    
    def __init__(self):
        # MCC code mapping (sample - expand as needed)
        self.category_to_mcc = {
            'GROCERY': {'mcc': '5411', 'industry': 'Food & Grocery'},
            'RESTAURANT': {'mcc': '5812', 'industry': 'Dining & Entertainment'},
            'GAS_STATION': {'mcc': '5541', 'industry': 'Automotive'},
            'PHARMACY': {'mcc': '5912', 'industry': 'Healthcare'},
            'RETAIL': {'mcc': '5399', 'industry': 'Retail'},
            'ONLINE_SHOPPING': {'mcc': '5968', 'industry': 'E-commerce'},
            'TRAVEL': {'mcc': '4722', 'industry': 'Travel & Hospitality'},
            'ENTERTAINMENT': {'mcc': '7832', 'industry': 'Dining & Entertainment'},
            'UTILITIES': {'mcc': '4900', 'industry': 'Utilities'},
            'INSURANCE': {'mcc': '6300', 'industry': 'Financial Services'},
            'HEALTHCARE': {'mcc': '8011', 'industry': 'Healthcare'},
            'EDUCATION': {'mcc': '8220', 'industry': 'Education'},
            'TRANSPORTATION': {'mcc': '4121', 'industry': 'Transportation'},
            'SUBSCRIPTION': {'mcc': '5968', 'industry': 'Digital Services'},
            'OTHER': {'mcc': '9999', 'industry': 'Other'},
        }
        
    def get_mcc_info(self, category: str) -> Dict:
        """Get MCC code and industry for a category"""
        return self.category_to_mcc.get(category, self.category_to_mcc['OTHER'])


class TransactionClassifier:
    """Classifies transactions using DistilBERT"""
    
    def __init__(self, model_name: str = 'distilbert-base-uncased'):
        self.device = 0 if torch.cuda.is_available() else -1
        
        # For this example, we'll use zero-shot classification
        # In production, you'd fine-tune DistilBERT on your transaction data
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=self.device
        )
        
        self.categories = [
            'GROCERY',
            'RESTAURANT', 
            'GAS_STATION',
            'PHARMACY',
            'RETAIL',
            'ONLINE_SHOPPING',
            'TRAVEL',
            'ENTERTAINMENT',
            'UTILITIES',
            'INSURANCE',
            'HEALTHCARE',
            'EDUCATION',
            'TRANSPORTATION',
            'SUBSCRIPTION'
        ]
        
    def classify_transaction(self, description: str, threshold: float = 0.3) -> str:
        """Classify transaction description into category"""
        if not description:
            return 'OTHER'
            
        try:
            result = self.classifier(
                description,
                candidate_labels=self.categories,
                multi_label=False
            )
            
            # Get top prediction
            top_category = result['labels'][0]
            top_score = result['scores'][0]
            
            # Return category if confidence is above threshold
            if top_score >= threshold:
                return top_category
            else:
                return 'OTHER'
                
        except Exception as e:
            print(f"Classification error: {e}")
            return 'OTHER'


class TransactionEnricher:
    """Main enrichment pipeline"""
    
    def __init__(self):
        self.cleaner = TransactionCleaner()
        self.classifier = TransactionClassifier()
        self.mcc_mapper = MCCMapper()
        
    def enrich_transaction(self, transaction: Dict) -> Dict:
        """Enrich a single transaction"""
        # Clean description
        raw_description = transaction.get('description', '')
        cleaned_description = self.cleaner.clean_description(raw_description)
        merchant_name = self.cleaner.extract_merchant_name(raw_description)
        
        # Classify
        category = self.classifier.classify_transaction(cleaned_description)
        
        # Get MCC info
        mcc_info = self.mcc_mapper.get_mcc_info(category)
        
        # Build enriched transaction
        enriched = transaction.copy()
        enriched.update({
            'cleaned_description': cleaned_description,
            'merchant_name': merchant_name,
            'category': category,
            'mcc_code': mcc_info['mcc'],
            'industry': mcc_info['industry'],
        })
        
        return enriched
    
    def enrich_transactions_batch(self, transactions: List[Dict]) -> List[Dict]:
        """Enrich multiple transactions"""
        enriched_transactions = []
        
        for i, txn in enumerate(transactions):
            try:
                enriched = self.enrich_transaction(txn)
                enriched_transactions.append(enriched)
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(transactions)} transactions")
                    
            except Exception as e:
                print(f"Error processing transaction {i}: {e}")
                enriched_transactions.append(txn)
                
        return enriched_transactions


def main():
    import sys
    from pathlib import Path
    
    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python enrich_transactions.py <input_file> [output_file]")
        print("\nInput file should be CSV with columns: transaction_id, date, description, amount")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('enriched_transactions.csv')
    
    if not input_file.exists():
        print(f"Error: Input file {input_file} not found")
        sys.exit(1)
    
    # Load transactions
    print(f"Loading transactions from {input_file}...")
    df = pd.read_csv(input_file)
    transactions = df.to_dict('records')
    
    # Initialize enricher
    print("Initializing transaction enricher...")
    enricher = TransactionEnricher()
    
    # Enrich transactions
    print(f"\nEnriching {len(transactions)} transactions...")
    enriched_transactions = enricher.enrich_transactions_batch(transactions)
    
    # Save results
    enriched_df = pd.DataFrame(enriched_transactions)
    enriched_df.to_csv(output_file, index=False)
    print(f"\nEnriched transactions saved to {output_file}")
    
    # Print summary
    print("\n=== ENRICHMENT SUMMARY ===")
    print(f"Total transactions: {len(enriched_df)}")
    print(f"\nCategory distribution:")
    print(enriched_df['category'].value_counts())
    print(f"\nIndustry distribution:")
    print(enriched_df['industry'].value_counts())


if __name__ == "__main__":
    main()
