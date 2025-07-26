import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
from scipy.stats import norm

CapCost = pd.read_excel('Data/ReferenceCase/CAPEX.xlsx')
TechInfo = pd.read_excel('Data/ReferenceCase/TechInfo.xlsx')
StorCost_DF = pd.read_excel('Data/ReferenceCase/StorageCapex.xlsx')
CapLim = pd.read_excel('Data/ReferenceCase/CapacityLimit.xlsx')
CapExidf = pd.read_excel('Data/ReferenceCase/ExistingCapacity.xlsx')
StorExidf = pd.read_excel('Data/ReferenceCase/ExistingStorage.xlsx')
CapacityFactors= pd.read_excel('Data/ReferenceCase/CapacityFactors.xlsx')
StorLim = pd.read_excel('Data/ReferenceCase/StorageLimit.xlsx')
Demand = pd.read_excel('Data/ReferenceCase/Demand.xlsx')


 # Convert data to numpy arrays
CapCost = np.array(CapCost['Annualized Investment Cost [EUR/GW]'])
StorCost = np.array(StorCost_DF['Annualized Investment Cost [EUR/GWh]'])
CapLim = np.array(CapLim['Maximum Capacity [GW]'])
CapExi = np.array(CapExidf['Capacity [GW]'])
StorExi = np.array(StorExidf['Capacity [GWh]'])
ProdFacOffWind = np.array(CapacityFactors['Offshore Capacity Factor'])
ProdFacOnWind = np.array(CapacityFactors['Onshore Capacity Factor'])
ProdFacSolar = np.array(CapacityFactors['Solar Capacity Factor'])
StorLim = np.array(StorLim['Maximum Capacity [GWh]'])
StorOpex = np.array(StorCost_DF['OPEX [EUR/GWh]'])
BatteryEfficiency = 0.93



 #Sampling Scenarios


# Constants
years = [2021]  # List of years for which scenarios are generated
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
        Offwind_std_dev = max(0.2 * OffWindMean, 0.0)
        OffWindDistributions[hour] = (OffWindMean, Offwind_std_dev)

        OnWindMean = ProdFacOnWind[idx]
        Onwind_std_dev = max(0.2 * OnWindMean, 0.0)
        OnWindDistributions[hour] = (OnWindMean, Onwind_std_dev)

        solar_mean = ProdFacSolar[idx]
        solar_std_dev = max(0.15 * solar_mean, 0.0)
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

#SetofScenarios = ['SituationToday', 'ReferenceCase', 'NetZero','HighFuel']

import pandas as pd
import numpy as np

def get_scenario_data(RunScenario):
    scenario_path = f'Data/{RunScenario}'

    # Load scenario-specific data
    opex_df = pd.read_excel(f'{scenario_path}/OPEX.xlsx')
    capout_df = pd.read_excel(f'{scenario_path}/CapacityOut.xlsx')
    demand_df = pd.read_excel(f'{scenario_path}/Demand.xlsx')

    # Convert data to numpy arrays
    OpCost = np.array(opex_df['Total OPEX [€/GWh]'])
    CapOut = np.array(capout_df['Maximum Capacity [GW]'])
    Demand = np.array(demand_df['Demand [GWh]'])
    CO2Intensity = np.array(opex_df['CO2 emissions [t/MWh]'])
    FixedOpex = np.array(opex_df['Fixed OPEX [EUR/GW]'])

    return Demand, OpCost, CapOut, CO2Intensity, FixedOpex
