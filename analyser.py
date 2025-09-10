from google.cloud import bigquery
from google.cloud.bigquery import QueryJobConfig, ScalarQueryParameter
from google.colab import auth
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- configs ---

auth.authenticate_user()
print('Authenticated')
project_id = 'hackathon-471014'
client = bigquery.Client(project=project_id)

# --- analyser the datas ---

class FinancialCrisisAnalyzer:
    def __init__(self, client):
        self.client = client
        self.banks = {
            'jpm': 'JPMorgan Chase',
            'bac': 'Bank of America (BOA)',
            'ms': 'Morgan Stanley',
            'gs': 'Goldman Sachs'
        }

    def analyze_stock_trends(self, bank_code, start_date='2006-01-01', end_date='2009-12-31'):
        """
        analyze stock trends and identify anomalies
        """
        query = f"""
        SELECT
            date_field_0 as date,
            close,
            LAG(close, 30) OVER (ORDER BY date_field_0) as prev_month_close,
            (close - LAG(close, 30) OVER (ORDER BY date_field_0)) / LAG(close, 30) OVER (ORDER BY date_field_0) as monthly_change
        FROM `banks.{bank_code}`
        WHERE date_field_0 BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY date
        """

        return self.client.query(query).to_dataframe()

    def analyze_sentiment_trends(self, filing_type, bank_code, start_year=2006, end_year=2009):
        """
        analyze sentiment trends in sec filings
        """
        filing_bank_code = 'boa' if bank_code == 'bac' else bank_code

        if filing_type == '10k':
            query = f"""
            SELECT
                year,
                AVG(sentiment_negative) as avg_negative_sentiment,
                AVG(sentiment_positive) as avg_positive_sentiment,
                COUNT(*) as total_chunks,
                SUM(CASE WHEN sentiment_negative > 0.7 THEN 1 ELSE 0 END) as high_negative_count
            FROM `hackathon.{filing_type}`
            WHERE bank = '{filing_bank_code}'
            AND year BETWEEN {start_year} AND {end_year}
            GROUP BY year
            ORDER BY year
            """
        else:
            query = f"""
            SELECT
                year,
                quarter,
                AVG(sentiment_negative) as avg_negative_sentiment,
                AVG(sentiment_positive) as avg_positive_sentiment,
                COUNT(*) as total_chunks,
                SUM(CASE WHEN sentiment_negative > 0.7 THEN 1 ELSE 0 END) as high_negative_count
            FROM `hackathon.{filing_type}`
            WHERE bank = '{filing_bank_code}'
            AND year BETWEEN {start_year} AND {end_year}
            GROUP BY year, quarter
            ORDER BY year, quarter
            """

        return self.client.query(query).to_dataframe()

    def find_risk_mentions(self, filing_type, bank_code, year=2008, quarter='Q1'):
        """
        find specific risk mentions in filings
        """
        filing_bank_code = 'boa' if bank_code == 'bac' else bank_code

        if filing_type == '10k':
            query = f"""
            SELECT
                text_chunk,
                sentiment_negative,
                section
            FROM `hackathon.{filing_type}`
            WHERE bank = '{filing_bank_code}'
            AND year = {year}
            AND sentiment_negative > 0.7
            ORDER BY sentiment_negative DESC
            LIMIT 10
            """
        else:
            query = f"""
            SELECT
                text_chunk,
                sentiment_negative,
                section
            FROM `hackathon.{filing_type}`
            WHERE bank = '{filing_bank_code}'
            AND year = {year}
            AND quarter = '{quarter}'
            AND sentiment_negative > 0.7
            ORDER BY sentiment_negative DESC
            LIMIT 10
            """

        return self.client.query(query).to_dataframe()

    # --- insigths generator using gemini 2.5 pro ml model on big query

    def generate_insights(self, bank_code, stock_analysis, sentiment_analysis):

        max_negative = sentiment_analysis['avg_negative_sentiment'].max()
        max_drop = stock_analysis['monthly_change'].min() * 100

        if 'quarter' in sentiment_analysis.columns:
            worst_period = sentiment_analysis.loc[sentiment_analysis['avg_negative_sentiment'].idxmax()]
            period_info = f"{worst_period.get('year', 'N/A')} {worst_period.get('quarter', '')}"
        else:
            worst_period = sentiment_analysis.loc[sentiment_analysis['avg_negative_sentiment'].idxmax()]
            period_info = f"{worst_period.get('year', 'N/A')}"

        stock_summary = stock_analysis.describe().to_json()
        sentiment_summary = sentiment_analysis.describe().to_json()

        prompt = f"""
        Analyze the financial crisis indicators for {self.banks[bank_code]}.

        Key metrics:
        - Maximum monthly stock drop: {max_drop:.2f}%
        - Peak negative sentiment in SEC filings: {max_negative:.3f}
        - Worst period: {period_info}

        Stock performance summary:
        {stock_summary}

        Sentiment analysis summary:
        {sentiment_summary}

        Please provide a comprehensive analysis including:
        1. Key findings from the data
        2. Early warning signs that were present
        3. Recommendations for future monitoring

        Focus on risk management failures, liquidity issues, and regulatory concerns.
        """

        query = """
                  SELECT
                     ml_generate_text_result AS generated_insights
                  FROM
                      ML.GENERATE_TEXT(
                      MODEL `hackathon-471014.crisis_analysis.anal`,
                      (SELECT @prompt_text AS prompt),
                      STRUCT(0.2 AS temperature, 8192 AS max_output_tokens, 0.8 AS top_p, 40 AS top_k)
                      )
                """

        job_config = QueryJobConfig(
            query_parameters=[
                ScalarQueryParameter("prompt_text", "STRING", prompt)
            ]
        )

        query_job = self.client.query(query, job_config=job_config)
        results = query_job.result()

        for row in results:
            return row.generated_insights

    # --- incase the model fails generates prebuilt insigths ---

    def _generate_fallback_insights(self, bank_code, max_negative, max_drop, period_info):
        
        insights = f"""
        CRISIS ANALYSIS FOR {self.banks[bank_code].upper()}

        Key Findings:
        1. Stock Performance: The stock experienced a maximum monthly drop of {max_drop:.2f}%
        2. Negative Sentiment: Peak negative sentiment in filings reached {max_negative:.3f}

        The worst period was {period_info}
        with an average negative sentiment score of {max_negative:.3f}.

        Early Warning Signs:
        - Increasing negative sentiment in SEC filings often preceded stock price declines
        - Risk disclosures related to mortgage-backed securities and credit default swaps
        - Mentions of liquidity concerns and counterparty risks

        Recommendations for Future Monitoring:
        1. Implement real-time sentiment analysis of financial disclosures
        2. Create alerts for sharp increases in negative sentiment
        3. Correlate sentiment trends with stock performance indicators
        4. Monitor specific risk factors mentioned in filings
        """

        return insights

# --- stock trend visualizer --- 

    def visualize_stock_trends(self, stock_data, bank_name):
        """
        create visualization of stock trends
        """
        plt.figure(figsize=(12, 6))
        plt.plot(stock_data['date'], stock_data['close'])
        plt.title(f'{bank_name} Stock Price (2006-2009)')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.bar(stock_data['date'], stock_data['monthly_change'] * 100)
        plt.title(f'{bank_name} Monthly Price Changes (%)')
        plt.xlabel('Date')
        plt.ylabel('Change (%)')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def visualize_sentiment_trends(self, sentiment_data, bank_name, filing_type):
        """
        create visualization of sentiment trends
        """
        if 'quarter' in sentiment_data.columns:
            sentiment_data['period'] = sentiment_data['year'].astype(str) + ' ' + sentiment_data['quarter']
        else:
            sentiment_data['period'] = sentiment_data['year'].astype(str)

        plt.figure(figsize=(12, 6))
        plt.plot(sentiment_data['period'], sentiment_data['avg_negative_sentiment'],
                marker='o', label='Negative Sentiment')
        if 'avg_positive_sentiment' in sentiment_data.columns:
            plt.plot(sentiment_data['period'], sentiment_data['avg_positive_sentiment'],
                    marker='o', label='Positive Sentiment')
        plt.title(f'{bank_name} {filing_type} Sentiment Trends')
        plt.xlabel('Period')
        plt.ylabel('Sentiment Score')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # --- crisis reporter for the selected bank ---

    def generate_crisis_report(self, bank_code):
        
        print(f"Analyzing {self.banks[bank_code]} for financial crisis indicators...")

        print("1. Analyzing stock performance...")
        stock_analysis = self.analyze_stock_trends(bank_code)
        self.visualize_stock_trends(stock_analysis, self.banks[bank_code])

        print("2. Analyzing annual report (10-K) sentiment trends...")
        k_analysis = self.analyze_sentiment_trends('10k', bank_code)
        self.visualize_sentiment_trends(k_analysis, self.banks[bank_code], '10-K')

        print("3. Analyzing quarterly report (10-Q) sentiment trends...")
        q_analysis = self.analyze_sentiment_trends('10q', bank_code)
        self.visualize_sentiment_trends(q_analysis, self.banks[bank_code], '10-Q')

        print("4. Identifying specific risk mentions...")
        risk_mentions = self.find_risk_mentions('10q', bank_code, 2008, 'Q1')
        if not risk_mentions.empty:
            print("Top risk mentions from 2008 Q1:")
            for i, row in risk_mentions.iterrows():
                print(f"{i+1}. {row['text_chunk'][:200]}... (Section: {row['section']}, Negative: {row['sentiment_negative']:.3f})")

        print("5. Generating comprehensive insights...")
        insights = self.generate_insights(bank_code, stock_analysis, q_analysis)

        return insights

# --- main functions ---

def main():
    """
    main function to drive the analysis based on user input
    """
    analyzer = FinancialCrisisAnalyzer(client)

    print("Welcome to the Financial Crisis Analyzer.")
    print("Please choose a bank to analyze:")

    bank_options = list(analyzer.banks.items())

    for i, (code, name) in enumerate(bank_options):
        print(f"{i + 1}. {name} ({code.upper()})")

    bank_name = ""
    selected_bank_code = ""
    while True:
        choice = input(f"\nPlease enter the number of your choice (1-{len(bank_options)}): ")
        selected_index = int(choice) - 1
        if 0 <= selected_index < len(bank_options):
            selected_bank_code, bank_name = bank_options[selected_index]
            break
        else:
            print(f"Invalid number. Please enter a number between 1 and {len(bank_options)}.")


    print(f"\nStarting analysis for {bank_name}...")

    insights = analyzer.generate_crisis_report(selected_bank_code)

    print(f"\n--- {bank_name} Crisis Analysis Insights ---")
    
    if isinstance(insights, dict) and 'candidates' in insights:
        clean_text = insights['candidates'][0]['content']['parts'][0]['text']
        print(clean_text)
    else:
        print(insights)

if __name__ == "__main__":
    main()