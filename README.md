# CO2 Emission Trend Analyzer for India 🌍

A comprehensive Python-based tool for analyzing CO2 emission trends in India with predictive modeling capabilities.

## Features

✅ **Real-time Data Collection**: Automatically fetches CO2 emission data from World Bank API
✅ **Comprehensive Analysis**: Statistical analysis of historical CO2 emission trends
✅ **Advanced Visualizations**: Multiple charts showing trends, growth rates, and moving averages
✅ **Predictive Modeling**: Machine learning models (Linear Regression, Random Forest) for future predictions
✅ **Detailed Reporting**: Generates comprehensive analysis reports
✅ **Fallback Data**: Uses sample data if API is unavailable

## Dataset Sources

- **Primary**: World Bank Open Data API
  - CO2 emissions per capita (EN.ATM.CO2E.PC)
  - Total CO2 emissions in kilotons (EN.ATM.CO2E.KT)
- **Secondary**: EDGAR (Emissions Database for Global Atmospheric Research)
- **Backup**: Generated sample data based on historical trends

## Installation

1. Clone or download the project files
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start
```bash
python co2_analyzer.py
```

### What the analyzer does:
1. **Data Collection**: Fetches latest CO2 emission data for India
2. **Trend Analysis**: Calculates growth rates, statistics, and patterns
3. **Visualization**: Creates multiple charts showing different aspects of emissions
4. **Modeling**: Builds and compares machine learning models
5. **Prediction**: Forecasts future CO2 emissions for the next 10 years
6. **Reporting**: Generates a comprehensive analysis report

## Output Files

The analyzer generates several output files:

- `co2_trends_analysis.png` - Multi-panel chart showing historical trends
- `co2_predictions.png` - Chart with historical data and future predictions
- `co2_analysis_report.txt` - Detailed text report with findings and recommendations

## Key Metrics Analyzed

- **CO2 Per Capita**: Metric tons per person
- **Total CO2 Emissions**: Kilotons annually
- **Growth Rates**: Year-over-year percentage changes
- **Moving Averages**: Smoothed trend lines
- **Future Projections**: 10-year forecasts

## Models Used

1. **Linear Regression**: For trend-based predictions
2. **Random Forest**: For complex pattern recognition
3. **Model Selection**: Automatically chooses best performing model based on R² score

## Technical Details

### Dependencies
- pandas: Data manipulation and analysis
- numpy: Numerical computations
- matplotlib/seaborn: Data visualization
- scikit-learn: Machine learning models
- requests: API data fetching
- statsmodels: Statistical analysis

### Data Processing
- Handles missing data points
- Calculates derived metrics (growth rates, moving averages)
- Validates data quality and consistency
- Provides fallback sample data if API fails

## Sample Analysis Results

The analyzer provides insights such as:
- Average CO2 per capita trends over time
- Peak emission years and values
- Annual growth rate patterns
- Future emission projections
- Policy recommendations

## Customization

You can modify the analyzer by:
- Changing the prediction timeframe (default: 10 years)
- Adding new visualization types
- Incorporating additional data sources
- Modifying model parameters
- Adding new metrics and calculations

## Contributing

Feel free to enhance the analyzer by:
- Adding more sophisticated models
- Including additional countries for comparison
- Integrating more data sources
- Improving visualizations
- Adding interactive features

## Data Sources and Citations

- World Bank Open Data: https://data.worldbank.org/
- EDGAR Database: https://edgar.jrc.ec.europa.eu/
- Our World in Data: https://ourworldindata.org/

## License

This project is open source and available under the MIT License.

---

**Note**: This analyzer is designed for educational and research purposes. For official policy decisions, please consult authoritative sources and conduct additional validation.