import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
from scipy.stats import norm


# Load data from Excel files
CapCost = pd.read_excel('Data/GovernmentTarget/CAPEX.xlsx')
OpCost = pd.read_excel('Data/GovernmentTarget/OPEX.xlsx')
TechInfo = pd.read_excel('Data/GovernmentTarget/TechInfo.xlsx')
StorCost = pd.read_excel('Data/GovernmentTarget/StorageCapex.xlsx')
CapLim = pd.read_excel('Data/GovernmentTarget/CapacityLimit.xlsx')
CapExi = pd.read_excel('Data/GovernmentTarget/ExistingCapacity.xlsx')
CapOut = pd.read_excel('Data/GovernmentTarget/CapacityOut.xlsx')
Demand = pd.read_excel('Data/GovernmentTarget/Demand.xlsx')
StorExi = pd.read_excel('Data/GovernmentTarget/ExistingStorage.xlsx')
CapacityFactors= pd.read_excel('Data/GovernmentTarget/CapacityFactors.xlsx')
StorLim = pd.read_excel('Data/GovernmentTarget/StorageLimit.xlsx')

# Create a dataframe for eta charge
EtaCh = pd.DataFrame({
    'Plant': ['Battery', 'Pumped Hydro'],
    'EtaCh': [0.9, 0.8]
})

# Create a dataframe for eta discharge
EtaDis = pd.DataFrame({
    'Plant': ['Battery', 'Pumped Hydro'],
    'EtaDis': [0.9, 0.8]
})



# Convert data to numpy arrays

CapCost = np.array(CapCost['Annualized Investment Cost [EUR/GW]'])
OpCost = np.array(OpCost['Total OPEX [€/GWh]'])
#TechInfo = np.array(TechInfo['Type'])
StorCost = np.array(StorCost['Annualized Investment Cost [EUR/GWh]'])
CapLim = np.array(CapLim['Maximum Capacity [GW]'])
CapExi = np.array(CapExi['Capacity [GW]'])
CapOut = np.array(CapOut['Maximum Capacity [GW]'])
Demand = np.array(Demand['Demand [GWh]'])
StorExi = np.array(StorExi['Capacity [GWh]'])
ProdFacOffWind = np.array(CapacityFactors['Offshore Capacity Factor'])
ProdFacOnWind = np.array(CapacityFactors['Onshore Capacity Factor'])
ProdFacSolar = np.array(CapacityFactors['Solar Capacity Factor'])
StorLim = np.array(StorLim['Maximum Capacity [GWh]'])



#Sampling Scenarios
import numpy as np

# Constants
years = [2021]
hours_per_year = len(Demand)
scenarios_per_year = 1 # Change this to generate more per year
num_years = len(years)
total_scenarios = num_years * scenarios_per_year

# Initialize final scenario arrays
Offwind_scenarios = np.zeros((hours_per_year, total_scenarios))
Onwind_scenarios = np.zeros((hours_per_year, total_scenarios))
Solar_scenarios = np.zeros((hours_per_year, total_scenarios))

for i, year in enumerate(years):
    start = i * hours_per_year

    # Step 1: Build distribution dictionary from that year's data
    OffWindDistributions = {}
    OnWindDistributions = {}
    solar_distributions = {}

    for hour in range(hours_per_year):
        idx = start + hour

        OffWindMean = ProdFacOffWind[idx]
        Offwind_std_dev = max(0.15 * OffWindMean, 0.05)
        OffWindDistributions[hour] = (OffWindMean, Offwind_std_dev)

        OnWindMean = ProdFacOnWind[idx]
        Onwind_std_dev = max(0.15 * OnWindMean, 0.05)
        OnWindDistributions[hour] = (OnWindMean, Onwind_std_dev)

        solar_mean = ProdFacSolar[idx]
        solar_std_dev = max(0.05 * solar_mean, 0.01)
        solar_distributions[hour] = (solar_mean, solar_std_dev)

    # Step 2: Generate multiple scenarios from this year
    for s in range(scenarios_per_year):
        col_index = i * scenarios_per_year + s

        for hour in range(hours_per_year):
            # Offshore
            mean, std = OffWindDistributions[hour]
            sample = np.random.normal(mean, std)
            Offwind_scenarios[hour, col_index] = np.clip(sample, 0, 1)

            # Onshore
            mean, std = OnWindDistributions[hour]
            sample = np.random.normal(mean, std)
            Onwind_scenarios[hour, col_index] = np.clip(sample, 0, 1)

            # Solar
            mean, std = solar_distributions[hour]
            sample = np.random.normal(mean, std)
            Solar_scenarios[hour, col_index] = np.clip(sample, 0, 1)
