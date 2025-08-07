import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

class CO2EmissionAnalyzer:
    def __init__(self):
        self.data = None
        self.models = {}
        self.predictions = {}
        
    def load_world_bank_data(self):
        """Load CO2 emission data from World Bank API"""
        print("Loading CO2 emission data for India from World Bank...")
        
        # World Bank API for CO2 emissions (metric tons per capita) for India
        url = "https://api.worldbank.org/v2/country/IND/indicator/EN.ATM.CO2E.PC?format=json&date=1990:2023&per_page=100"
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if len(data) > 1:
                    records = []
                    for item in data[1]:
                        if item['value'] is not None:
                            records.append({
                                'year': int(item['date']),
                                'co2_per_capita': float(item['value']),
                                'country': item['country']['value']
                            })
                    
                    df = pd.DataFrame(records)
                    df = df.sort_values('year')
                    
                    # Get total CO2 emissions data
                    url_total = "https://api.worldbank.org/v2/country/IND/indicator/EN.ATM.CO2E.KT?format=json&date=1990:2023&per_page=100"
                    response_total = requests.get(url_total)
                    
                    if response_total.status_code == 200:
                        data_total = response_total.json()
                        if len(data_total) > 1:
                            for item in data_total[1]:
                                if item['value'] is not None:
                                    year = int(item['date'])
                                    if year in df['year'].values:
                                        df.loc[df['year'] == year, 'co2_total_kt'] = float(item['value'])
                    
                    self.data = df
                    print(f"Loaded {len(df)} years of data from {df['year'].min()} to {df['year'].max()}")
                    return True
        except Exception as e:
            print(f"Error loading World Bank data: {e}")
            
        return False
    
    def create_sample_data(self):
        """Create sample CO2 emission data for India if API fails"""
        print("Creating sample CO2 emission data for India...")
        
        # Historical trend data for India (approximate values)
        years = list(range(1990, 2024))
        
        # Simulate realistic CO2 emission trend for India
        base_emissions = 0.7  # Starting CO2 per capita in 1990
        growth_rate = 0.045   # Annual growth rate
        
        co2_per_capita = []
        co2_total_kt = []
        
        for i, year in enumerate(years):
            # Add some realistic variation
            variation = np.random.normal(0, 0.05)
            per_capita = base_emissions * (1 + growth_rate) ** i + variation
            
            # Ensure positive values and realistic bounds
            per_capita = max(0.5, min(per_capita, 3.0))
            co2_per_capita.append(per_capita)
            
            # Estimate total emissions (approximate population growth)
            population_factor = 1.3 + 0.01 * i  # Population growth factor
            total_kt = per_capita * 1000000 * population_factor  # Rough calculation
            co2_total_kt.append(total_kt)
        
        self.data = pd.DataFrame({
            'year': years,
            'co2_per_capita': co2_per_capita,
            'co2_total_kt': co2_total_kt,
            'country': 'India'
        })
        
        print(f"Created sample data for {len(years)} years")
        return True
    
    def load_data(self):
        """Load CO2 emission data"""
        if not self.load_world_bank_data():
            self.create_sample_data()
    
    def analyze_trends(self):
        """Analyze CO2 emission trends"""
        if self.data is None:
            print("No data loaded!")
            return
        
        print("\n=== CO2 EMISSION TREND ANALYSIS FOR INDIA ===")
        print(f"Data period: {self.data['year'].min()} - {self.data['year'].max()}")
        print(f"Total years: {len(self.data)}")
        
        # Basic statistics
        print("\n--- Basic Statistics ---")
        print(f"Average CO2 per capita: {self.data['co2_per_capita'].mean():.2f} metric tons")
        print(f"Minimum CO2 per capita: {self.data['co2_per_capita'].min():.2f} metric tons ({self.data.loc[self.data['co2_per_capita'].idxmin(), 'year']})")
        print(f"Maximum CO2 per capita: {self.data['co2_per_capita'].max():.2f} metric tons ({self.data.loc[self.data['co2_per_capita'].idxmax(), 'year']})")
        
        # Calculate year-over-year growth
        self.data['yoy_growth'] = self.data['co2_per_capita'].pct_change() * 100
        avg_growth = self.data['yoy_growth'].mean()
        print(f"Average annual growth rate: {avg_growth:.2f}%")
        
        # Recent trend (last 5 years)
        recent_data = self.data.tail(5)
        recent_growth = recent_data['yoy_growth'].mean()
        print(f"Recent 5-year average growth: {recent_growth:.2f}%")
    
    def visualize_trends(self):
        """Create visualizations of CO2 emission trends"""
        if self.data is None:
            return
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('India CO2 Emission Trends Analysis', fontsize=16, fontweight='bold')
        
        # 1. CO2 per capita trend
        axes[0, 0].plot(self.data['year'], self.data['co2_per_capita'], 
                       marker='o', linewidth=2, markersize=4, color='red')
        axes[0, 0].set_title('CO2 Emissions Per Capita')
        axes[0, 0].set_xlabel('Year')
        axes[0, 0].set_ylabel('CO2 (metric tons per capita)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Total CO2 emissions
        if 'co2_total_kt' in self.data.columns:
            axes[0, 1].plot(self.data['year'], self.data['co2_total_kt'], 
                           marker='s', linewidth=2, markersize=4, color='blue')
            axes[0, 1].set_title('Total CO2 Emissions')
            axes[0, 1].set_xlabel('Year')
            axes[0, 1].set_ylabel('CO2 (kilotons)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Year-over-year growth rate
        axes[1, 0].bar(self.data['year'][1:], self.data['yoy_growth'][1:], 
                      color='green', alpha=0.7)
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 0].set_title('Year-over-Year Growth Rate')
        axes[1, 0].set_xlabel('Year')
        axes[1, 0].set_ylabel('Growth Rate (%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Moving average
        window = 5
        self.data['moving_avg'] = self.data['co2_per_capita'].rolling(window=window).mean()
        axes[1, 1].plot(self.data['year'], self.data['co2_per_capita'], 
                       alpha=0.5, label='Actual', color='red')
        axes[1, 1].plot(self.data['year'], self.data['moving_avg'], 
                       linewidth=3, label=f'{window}-Year Moving Average', color='darkred')
        axes[1, 1].set_title('CO2 Emissions with Moving Average')
        axes[1, 1].set_xlabel('Year')
        axes[1, 1].set_ylabel('CO2 (metric tons per capita)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/workspace/co2_trends_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Trend analysis chart saved as 'co2_trends_analysis.png'")
    
    def build_models(self):
        """Build predictive models for CO2 emissions"""
        if self.data is None:
            return
        
        print("\n=== BUILDING PREDICTIVE MODELS ===")
        
        # Prepare data for modeling
        X = self.data[['year']].values
        y = self.data['co2_per_capita'].values
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 1. Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        lr_r2 = r2_score(y_test, lr_pred)
        lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
        
        self.models['linear'] = lr_model
        print(f"Linear Regression - R²: {lr_r2:.3f}, RMSE: {lr_rmse:.3f}")
        
        # 2. Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_r2 = r2_score(y_test, rf_pred)
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        
        self.models['random_forest'] = rf_model
        print(f"Random Forest - R²: {rf_r2:.3f}, RMSE: {rf_rmse:.3f}")
        
        # Choose best model
        if lr_r2 > rf_r2:
            self.best_model = lr_model
            self.best_model_name = 'Linear Regression'
            print(f"Best model: Linear Regression (R²: {lr_r2:.3f})")
        else:
            self.best_model = rf_model
            self.best_model_name = 'Random Forest'
            print(f"Best model: Random Forest (R²: {rf_r2:.3f})")
    
    def predict_future(self, years_ahead=10):
        """Predict future CO2 emissions"""
        if not hasattr(self, 'best_model'):
            print("No model trained yet!")
            return
        
        print(f"\n=== FUTURE PREDICTIONS ({years_ahead} years ahead) ===")
        
        last_year = self.data['year'].max()
        future_years = np.array([[year] for year in range(last_year + 1, last_year + years_ahead + 1)])
        
        predictions = self.best_model.predict(future_years)
        
        future_df = pd.DataFrame({
            'year': range(last_year + 1, last_year + years_ahead + 1),
            'predicted_co2_per_capita': predictions
        })
        
        self.predictions = future_df
        
        print(f"Using {self.best_model_name} model:")
        for _, row in future_df.iterrows():
            print(f"  {row['year']}: {row['predicted_co2_per_capita']:.2f} metric tons per capita")
        
        return future_df
    
    def create_prediction_chart(self):
        """Create visualization with predictions"""
        if self.predictions is None or len(self.predictions) == 0:
            return
        
        plt.figure(figsize=(12, 8))
        
        # Historical data
        plt.plot(self.data['year'], self.data['co2_per_capita'], 
                marker='o', linewidth=2, markersize=5, color='blue', label='Historical Data')
        
        # Predictions
        plt.plot(self.predictions['year'], self.predictions['predicted_co2_per_capita'], 
                marker='s', linewidth=2, markersize=5, color='red', linestyle='--', label='Predictions')
        
        # Connect last historical point with first prediction
        last_historical = self.data.iloc[-1]
        first_prediction = self.predictions.iloc[0]
        plt.plot([last_historical['year'], first_prediction['year']], 
                [last_historical['co2_per_capita'], first_prediction['predicted_co2_per_capita']], 
                color='red', linestyle='--', alpha=0.5)
        
        plt.title(f'India CO2 Emissions: Historical Data and Predictions\n(Using {self.best_model_name})', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('CO2 Emissions (metric tons per capita)', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Add vertical line to separate historical and predicted data
        plt.axvline(x=self.data['year'].max(), color='gray', linestyle=':', alpha=0.7, label='Present')
        
        plt.tight_layout()
        plt.savefig('/workspace/co2_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Prediction chart saved as 'co2_predictions.png'")
    
    def generate_report(self):
        """Generate a comprehensive report"""
        if self.data is None:
            return
        
        report = f"""
# CO2 EMISSION TREND ANALYSIS REPORT - INDIA

## Data Summary
- Analysis Period: {self.data['year'].min()} - {self.data['year'].max()}
- Total Years Analyzed: {len(self.data)}
- Data Source: World Bank / Sample Data

## Key Findings

### Current Status
- Latest CO2 per capita ({self.data['year'].max()}): {self.data['co2_per_capita'].iloc[-1]:.2f} metric tons
- Average CO2 per capita: {self.data['co2_per_capita'].mean():.2f} metric tons
- Peak emission year: {self.data.loc[self.data['co2_per_capita'].idxmax(), 'year']} ({self.data['co2_per_capita'].max():.2f} metric tons)

### Trend Analysis
- Average annual growth rate: {self.data['yoy_growth'].mean():.2f}%
- Recent 5-year growth: {self.data.tail(5)['yoy_growth'].mean():.2f}%

### Model Performance
- Best performing model: {self.best_model_name}
"""
        
        if hasattr(self, 'predictions') and len(self.predictions) > 0:
            report += f"""
### Future Projections
- Predicted CO2 per capita in {self.predictions['year'].iloc[4]}: {self.predictions['predicted_co2_per_capita'].iloc[4]:.2f} metric tons
- Predicted CO2 per capita in {self.predictions['year'].iloc[-1]}: {self.predictions['predicted_co2_per_capita'].iloc[-1]:.2f} metric tons
"""
        
        report += """
## Recommendations
1. Monitor emission trends closely
2. Implement sustainable energy policies
3. Invest in renewable energy infrastructure
4. Promote energy efficiency measures
5. Regular model updates with new data

---
Generated by CO2 Emission Trend Analyzer
"""
        
        with open('/workspace/co2_analysis_report.txt', 'w') as f:
            f.write(report)
        
        print("Analysis report saved as 'co2_analysis_report.txt'")
        print(report)

def main():
    """Main function to run the complete analysis"""
    print("🌍 CO2 EMISSION TREND ANALYZER FOR INDIA 🌍")
    print("=" * 50)
    
    analyzer = CO2EmissionAnalyzer()
    
    # Load data
    analyzer.load_data()
    
    # Analyze trends
    analyzer.analyze_trends()
    
    # Create visualizations
    analyzer.visualize_trends()
    
    # Build predictive models
    analyzer.build_models()
    
    # Make predictions
    analyzer.predict_future(years_ahead=10)
    
    # Create prediction chart
    analyzer.create_prediction_chart()
    
    # Generate report
    analyzer.generate_report()
    
    print("\n✅ Analysis complete! Check the generated files:")
    print("  📊 co2_trends_analysis.png - Trend analysis charts")
    print("  📈 co2_predictions.png - Future predictions chart")
    print("  📄 co2_analysis_report.txt - Detailed analysis report")

if __name__ == "__main__":
    main()