import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
from scipy.stats import norm
from DataHandling import *
import matplotlib.pyplot as plt
from IDPredictor import *

# Class for external input data
class InputData():
    def __init__(self, CapCost, OpCost, TechInfo, StorCost, CapLim,CapExi,CapOut,Dem,EtaCha,EtaDis,StorExi, Offwind_scenarios, Onwind_scenarios, Solar_scenarios, StorLim,CO2Intensity):
        self.CapCost = CapCost
        self.OpCost = OpCost
        self.TechInfo = TechInfo
        self.StorCost = StorCost
        self.CapLim = CapLim
        self.CapExi = CapExi
        self.CapOut = CapOut
        self.Dem = Dem
        self.EtaCha = EtaCha
        self.EtaDis = EtaDis
        self.StorExi = StorExi
        self.Offwind_scenarios = Offwind_scenarios
        self.Onwind_scenarios = Onwind_scenarios
        self.Solar_scenarios = Solar_scenarios
        self.StorLim = StorLim
        self.CO2Intensity = CO2Intensity
        

# Class for model parameters
class Parameters():
    def __init__(self, epsilon, delta, N_Hours, N_Cap, N_Stor, N_Scen, BigM,BatteryCapacity,BatterPower):
        self.epsilon = epsilon
        self.delta = delta
        self.N_Hours = N_Hours
        self.N_Cap = N_Cap
        self.N_Stor = N_Stor
        self.N_Scen = N_Scen
        self.BigM = BigM
        self.Capacity = BatteryCapacity
        self.Power = BatterPower
        

        # Create Vectors for vector multiplication
        self.hourvector = np.ones((N_Hours,1))
        self.genvector = np.ones((N_Cap,1))
        self.storvector = np.ones((N_Stor,1))
        self.scenvector = np.ones((N_Scen,1))



# CLASS WHICH CAN HAVE ATTRIBUTES SET

class Expando(object):
    '''
        A small class which can have attributes set
    '''
    pass


# Defining the capacity problem class

class CapacityProblem():
    def __init__(self, ParametersObj, DataObj, Model_results = 1, Guroby_results = 1):
        self.P = ParametersObj # Parameters
        self.D = DataObj # Data
        self.Model_results = Model_results
        self.Guroby_results = Guroby_results
        self.var = Expando()  # Variables
        self.con = Expando()  # Constraints
        self.res = Expando()  # Results
        self._build_model() 


    def _build_variables(self):
        # Create the variables
        self.var.CapNew = self.m.addMVar((self.P.N_Cap), lb=0)  # New Capacity for each type of generator technology
        self.var.EGen = self.m.addMVar((self.P.N_Cap, self.P.N_Hours, self.P.N_Scen), lb=0)  # Energy Production for each type of generator technology for each hour and scenario
        self.var.CapStor = self.m.addMVar((self.P.N_Stor), lb=0)  # New storage capacity for each type of storage technology
        self.var.SOC = self.m.addMVar((self.P.N_Stor, self.P.N_Hours,self.P.N_Scen), lb=0)  # State of charge for each type of storage technology for each hour and scenario  
        self.var.EChar = self.m.addMVar((self.P.N_Stor, self.P.N_Hours,self.P.N_Scen), lb=0)  # Energy charged for each type of storage technology for each hour and scenario
        self.var.EDis = self.m.addMVar((self.P.N_Stor, self.P.N_Hours,self.P.N_Scen), lb=0)  # Energy discharged for each type of storage technology for each hour and scenario
        self.var.u = self.m.addMVar((self.P.N_Scen), vtype=GRB.BINARY) # Binary variable for each scenario  # Chance for each scenario


    def _build_constraints(self):
    
        G, H, S, U = self.P.N_Cap, self.P.N_Hours, self.P.N_Scen, self.P.N_Stor
        

        # Capacity is limited my maximum investable capacity for each technology
        self.con.CapLim = self.m.addConstr(self.var.CapNew <= self.D.CapLim, name='Capacitylimit')
        self.con.CapStorLim = self.m.addConstr(self.var.CapStor <= self.D.StorLim, name='Storage capacity limit')



          # Production limited by sum of new capacity, existing capacity and phased out capacity times the production factor
        
        # Generation for each hour is limited by capacity and production factors
        CapTotal = self.var.CapNew.reshape((G, 1, 1)) + self.D.CapExi.reshape((G, 1, 1)) + self.D.CapOut.reshape((G, 1, 1))

        Techs = self.D.TechInfo['Technology'].values
        OffshoreMask = (Techs == 'Wind Offshore')
        OnshoreMask = (Techs == 'Wind Onshore')
        SolarMask = (Techs == 'PV')
        ConvMask = ~(OffshoreMask | OnshoreMask | SolarMask)

        if OffshoreMask.any():
            self.con.ProdLim_Offshore = self.m.addConstr(
                self.var.EGen[OffshoreMask, :, :] <= CapTotal[OffshoreMask, :, :] * self.D.Offwind_scenarios[None, :, :],
                name='ProdLimit_Offshore'
            )
        if OnshoreMask.any():
            self.con.ProdLim_Onshore = self.m.addConstr(
                self.var.EGen[OnshoreMask, :, :] <= CapTotal[OnshoreMask, :, :] * self.D.Onwind_scenarios[None, :, :],
                name='ProdLimit_Onshore'
            )
        if SolarMask.any():
            self.con.ProdLim_Solar = self.m.addConstr(
                self.var.EGen[SolarMask, :, :] <= CapTotal[SolarMask, :, :] * self.D.Solar_scenarios[None, :, :],
                name='ProdLimit_Solar'
            )
        if ConvMask.any():
            self.con.ProdLim_Conv = self.m.addConstr(
                self.var.EGen[ConvMask, :, :] <= CapTotal[ConvMask, :, :],
                name='ProdLimit_Conventional'
            )

        # Demand and storage charge must be met for each hour and scenario by generation and storage discharge
        EGen_sum = self.var.EGen.sum(axis=0)    # shape (H, S)
        EDis_sum = self.var.EDis.sum(axis=0)    # shape (H, S)
        EChar_sum = self.var.EChar.sum(axis=0)  # shape (H, S)
        Demand_scaled = self.D.Dem[:H, None]  # shape (H, 1), broadcast to (H, S)

        self.con.Balance = self.m.addConstr(EGen_sum + EDis_sum == Demand_scaled + EChar_sum, name='EnergyBalance')

        
        # for h in range(self.P.N_Hours):
        #     for s in range(self.P.N_Scen):
        #         self.con.Balance = self.m.addConstr(gp.quicksum(self.var.EGen[:, h, s]) 
        #                                             #+ gp.quicksum(self.var.EDis[:, h, s]) - gp.quicksum(self.var.EChar[:, h, s])  
        #                                             == self.D.Dem[h], name=f'EnergyBalance_{h}_{s}') 


        # Defining RES share as a percentage of total energy demand
        for s in range(self.P.N_Scen):
            self.con.RESShare = self.m.addConstr(self.P.delta * gp.quicksum(self.D.Dem[h] for h in range(self.P.N_Hours)) - gp.quicksum(self.var.EGen[g, h,s] for h in range(self.P.N_Hours) for g in range(self.P.N_Cap) if self.D.TechInfo['Type'][g] == 'RES' )  <= self.P.BigM * (1 - self.var.u[s]), name=f'RES share S_{s}')

         # Defining how many scenarios are allowed to be violated
        self.con.ScenViol = self.m.addConstr(gp.quicksum(self.var.u[s] for s in range(self.P.N_Scen))/self.P.N_Scen >= (1- self.P.epsilon), name=f'Scenario violation')

        
        #Defining SOC, Charge and Discharge for each scenario and each hour >0
        SOC = self.var.SOC
        EChar = self.var.EChar
        EDis = self.var.EDis
        SOC_prev = SOC[:, :-1, :]  # (U, H-1, S)
        EChar_prev = EChar[:, :-1, :]
        EDis_prev = EDis[:, :-1, :]
        SOC_curr = SOC[:, 1:, :]

        # Defining SOC as results of previous hours
        self.con.SOC_Update = self.m.addConstr(SOC_curr == SOC_prev + EChar_prev - EDis_prev, name='SOC_Update')

        # Defining Initial SOC
        self.con.SOC0 = self.m.addConstr(SOC[:, 0, :] == self.D.StorExi[:, None], name='SOC_Initial')

        # Limit SOC by existing storage capacity and new storage capacity
        self.con.SOCLim = self.m.addConstr(
            SOC <= self.var.CapStor[:, None, None] + self.D.StorExi[:, None, None],
            name='StorageCapLimit'
        )

        # Limit charge by existing storage capacity and new storage capacity - SOC
        self.con.ECharLim = self.m.addConstr(
            EChar <= self.var.CapStor[:, None, None] + self.D.StorExi[:, None, None] - SOC,
            name='ECharLimit'
        )
        # Limit Discharge by SOC
        self.con.EDisLim = self.m.addConstr(
            EDis <= SOC,
            name='EDisLimit'
        )

            # Indices of Lignite and Hard Coal
        lignite_index = list(self.D.TechInfo['Technology']).index('Lignite')
        hard_coal_index = list(self.D.TechInfo['Technology']).index('Hard Coal')

        # Ramp rates for Hard Coal and Lignite
        ramp_rates = np.zeros(G)  # Initialize a vector of zeros
        ramp_rates[lignite_index] = 0.1  # 10% ramp rate for Lignite
        ramp_rates[hard_coal_index] = 0.1  # 10% ramp rate for Hard Coal

        # Capacity including new investments, existing, and out capacity
        CapTotal = self.var.CapNew + self.D.CapExi + self.D.CapOut  # Shape: (G,)

        # Mask for only Lignite and Hard Coal
        ramp_mask = np.array([False] * G)
        ramp_mask[lignite_index] = True
        ramp_mask[hard_coal_index] = True

        # Extract only the technologies with ramp constraints
        ramp_EGen = self.var.EGen[ramp_mask, :, :]  # Shape: (2, H, S)
        ramp_limits = ramp_rates[ramp_mask] * CapTotal[ramp_mask]  # Shape: (2,)

        # Shifted versions of generation for ramping constraints
        ramp_EGen_prev = ramp_EGen[:, :-1, :]  # Shape: (2, H-1, S)
        ramp_EGen_curr = ramp_EGen[:, 1:, :]   # Shape: (2, H-1, S)

        # --- Vectorized Ramping Constraints ---
        # Up-ramp constraints
        self.con.RampUp = self.m.addConstr(
            ramp_EGen_curr - ramp_EGen_prev <= ramp_limits[:, None, None], 
            name='Ramping_Up'
        )

        # Down-ramp constraints
        self.con.RampDown = self.m.addConstr(
            ramp_EGen_prev - ramp_EGen_curr <= ramp_limits[:, None, None], 
            name='Ramping_Down'
        )
        
    
    def _build_objective(self):
        # Vectorized investment cost for generators multiplied by new capacity
        gen_capex_cost = self.var.CapNew @ self.D.CapCost 

        # Vectorized operational cost for generation multiplied by generation
        op_cost = (self.var.EGen * self.D.OpCost[:, None, None]).sum()

        # Vectorized storage investment cost multiplied by new storage capacity
        stor_cost = self.var.CapStor @ self.D.StorCost  

        # Set objective, minimizing total costs
        self.m.setObjective(gen_capex_cost + op_cost/self.P.N_Scen + stor_cost, GRB.MINIMIZE)


    def _display_guropby_results(self):
        self.m.setParam('OutputFlag', self.Guroby_results)

    
    def _build_model(self):
        self.m = gp.Model('CapacityProblem')
        self._build_variables()
        self._build_constraints()
        self._build_objective()
        self._display_guropby_results()
        self.m.write("CapacityProblem.lp")
        self.m.optimize()
        self._results()
        if self.Model_results == 1:
            self._extract_results()
            

    
    def _results(self):
        self.res.obj = self.m.objVal
        self.res.CapNew = self.var.CapNew.X
        self.res.EGen = self.var.EGen.X
        self.res.CapStor = self.var.CapStor.X
        self.res.SOC = self.var.SOC.X
        self.res.EChar = self.var.EChar.X
        self.res.EDis = self.var.EDis.X
        self.res.u = self.var.u.X

          # Total number of constraints
        self.res.TotalConstraints = self.m.NumConstrs

        # Count number of active constraints
        def count_active_constraints(model, tol=1e-6):
            model.update()  # Make sure model is up to date
            active = 0
            for c in model.getConstrs():
                slack = c.getAttr("Slack")
                sense = c.Sense

                if sense == '=' and abs(slack) <= tol:
                    active += 1
                elif sense == '<' and abs(slack) <= tol:
                    active += 1
                elif sense == '>' and abs(slack) <= tol:
                    active += 1
            return active


        self.res.ActiveConstraints = count_active_constraints(self.m)



         # Calculate RES Share for each scenario
        res_shares = []
        for s in range(self.P.N_Scen):
            res_share = np.sum(self.res.EGen[:, :, s][self.D.TechInfo['Type'] == 'RES']) / np.sum(self.D.Dem)
            res_shares.append(res_share)

        # Create a DataFrame for easy export
        self.res.RESShare = pd.DataFrame({
            'Scenario': [f'Scenario_{i}' for i in range(self.P.N_Scen)],
            'RES Share': res_shares
        })
         
      
        #Create Dataframe for new capacity
        self.res.CapNew = pd.DataFrame({
            'Plant': ['Offshore', 'Onshore', 'Biomass', 'Lignite', 'Hard Coal', 'Gas', 'Hydrogren', 'Nuclear', 'Solar PV', 'Hydro', 'Other RES', 'Other Conventional'],
            'CapNew[GW]': self.res.CapNew,
            'CapExi[GW]': self.D.CapExi,
            'CapOut[GW]': self.D.CapOut
        })

        #Create Dataframe for new storage capacity
        self.res.CapStor = pd.DataFrame({
            'Plant': ['Battery', 'Pumped Hydro'],
            'CapStor[GWh]': self.res.CapStor
        })

        # Create DataFrame for scenario violations
        self.res.Violation_Scenarios = pd.DataFrame({
            'Scenario': list(range(self.P.N_Scen)),
            'Violated (1=No, 0=Yes)': self.res.u
        })
        
        
    
        # Store hourly results in one dataframe
        self.res.EGen_Scenarios = {}

        for s in range(self.P.N_Scen):
            # Extract data for scenario `s`
            e_gen_scenario = self.res.EGen[:, :, s].T  
            soc_scenario = self.res.SOC[:, :, s].T  
            echar_scenario = self.res.EChar[:, :, s].T  
            edis_scenario = self.res.EDis[:, :, s].T  

            # Extract Wind and Solar Production Factors
            Offwind_scenarios = self.D.Offwind_scenarios[:, s]  
            Onwind_scenarios = self.D.Onwind_scenarios[:, s]  
            Solar_scenarios = self.D.Solar_scenarios[:, s]  

            # Convert to DataFrames
            df_scenario = pd.DataFrame(e_gen_scenario, columns=['Offshore', 'Onshore', 'Biomass', 'Lignite', 'Hard Coal', 'Gas', 'Hydrogren', 'Nuclear', 'Solar PV', 'Hydro', 'Other RES', 'Other Conventional'])
            df_soc = pd.DataFrame(soc_scenario, columns=['Battery_SOC', 'PumpedHydro_SOC'])
            df_echar = pd.DataFrame(echar_scenario, columns=['Battery_Charge', 'PumpedHydro_Charge'])
            df_edis = pd.DataFrame(edis_scenario, columns=['Battery_Discharge', 'PumpedHydro_Discharge'])
            df_demand = pd.DataFrame({'Demand': self.D.Dem})

            # Add Wind & Solar Production Factors
            df_prod_factors = pd.DataFrame({'OffWind': Offwind_scenarios, 'OnWind': Onwind_scenarios, 'Solar': Solar_scenarios})

            # Add Hour column
            df_scenario['Hour'] = list(range(self.P.N_Hours))

            # Merge all DataFrames
            df_complete = pd.concat([df_scenario, df_soc, df_echar, df_edis, df_demand, df_prod_factors], axis=1)

            # Reorder columns
            df_complete = df_complete[['Hour', 'Offshore', 'Onshore', 'Biomass', 'Lignite', 'Hard Coal', 'Gas', 'Hydrogren', 'Nuclear', 'Solar PV', 'Hydro', 'Other RES', 'Other Conventional',
                                    'Battery_SOC', 'PumpedHydro_SOC',
                                    'Battery_Charge', 'PumpedHydro_Charge',
                                    'Battery_Discharge', 'PumpedHydro_Discharge',
                                    'Demand', 'OffWind','OnWind', 'Solar']]

            # Store in dictionary
            self.res.EGen_Scenarios[f"Scenario_{s}"] = df_complete


             # Fetch technology names dynamically from TechInfo
        generation_technologies = self.D.TechInfo['Technology'].values

        # Initialize an empty DataFrame to store results
        annual_generation = pd.DataFrame(index=generation_technologies)

        # Loop through each scenario and sum up hourly generation for each technology
        for s in range(self.P.N_Scen):
            # Get the hourly generation data for the scenario
            scenario_data = self.res.EGen[:, :, s]  # Shape: (G, H)
            # Sum over all hours to get the annual generation
            annual_gen_scenario = scenario_data.sum(axis=1)  # Shape: (G,)
            # Add to the DataFrame as a new column
            annual_generation[f'Scenario_{s}'] = annual_gen_scenario
        
        # Store in results for easy access
        self.res.AnnualGeneration = annual_generation



        # Calculate CO2 intensity for each scenario
        co2_intensity = np.zeros((self.P.N_Scen, 1))

        for s in range(self.P.N_Scen):
            # Calculate the total emissions for each technology and sum them up
            emission_sum = np.sum(self.res.EGen[:, :, s] * self.D.CO2Intensity[:, None] * 1000, axis=1)  # tCO2

            # Total energy generated in that scenario
            total_energy_generated = np.sum(self.res.EGen[:, :, s])

            # Calculate the intensity, avoid division by zero
            co2_intensity[s] = np.sum(emission_sum) / total_energy_generated if total_energy_generated > 0 else 0

        # Create DataFrame with the results
        self.res.CO2Intensity = pd.DataFrame({
            'Scenario': [f'Scenario_{i}' for i in range(self.P.N_Scen)],
            'CO2 Intensity [tCO2/MWh]': co2_intensity.flatten()
        })

            
    

    def _extract_results(self):
        # Display the objective value
        print('Objective value: ', self.m.objVal)
        # Display the new capacity for each type of generator technology
        print('New capacity for each type of generator technology: ', self.res.CapNew)
        
        #Display new storage capacity for each type of storage technology
        print('New storage capacity for each type of storage technology: ', self.res.CapStor)

        # Display RES Share for each scenario
        print('RES Share for each scenario: ', self.res.RESShare)

         # Print constraint info
        print(f"Total number of constraints: {self.res.TotalConstraints}")
        print(f"Number of active (binding) constraints: {self.res.ActiveConstraints}")









# Defining the Day Ahead Problem class

class DayAheadProblem():
    def __init__(self, ParametersObj, DataObj, GenerationCapacity, StorageCapacity,IDForecaster, Model_results = 1, Guroby_results = 1):
        self.P = ParametersObj # Parameters
        self.D = DataObj # Data
        self.Model_results = Model_results
        self.Guroby_results = Guroby_results
        self.IDForecaster = IDForecaster
        self.GenCap = GenerationCapacity
        self.StorCap = StorageCapacity
        self.var = Expando()  # Variables
        self.con = Expando()  # Constraints
        self.res = Expando()  # Results
        self._build_model() 


    def _build_variables(self):
        # Create the variables
        self.var.EGen = self.m.addMVar((self.P.N_Cap, self.P.N_Hours, self.P.N_Scen), lb=0)  # Variable for generation for each scenario, technology, and hour
        self.var.SOC = self.m.addMVar((self.P.N_Stor, self.P.N_Hours, self.P.N_Scen), lb=0)  # Variable for state of charge for each scenario, storage technology, and hour
        self.var.EChar = self.m.addMVar((self.P.N_Stor, self.P.N_Hours, self.P.N_Scen), lb=0)  # Variable for energy charge for each scenario, storage technology, and hour
        self.var.EDis = self.m.addMVar((self.P.N_Stor, self.P.N_Hours, self.P.N_Scen), lb=0)  # Variable for energy discharge for each scenario, storage technology, and hour

    def _build_constraints(self):

        G, H, S, U = self.P.N_Cap, self.P.N_Hours, self.P.N_Scen, self.P.N_Stor
    
        # Generation capacity constraint
        CapTotal = self.GenCap.reshape((G, 1, 1)) + self.D.CapExi.reshape((G, 1, 1)) + self.D.CapOut.reshape((G, 1, 1))


        Techs = self.D.TechInfo['Technology'].values
        OffshoreMask = (Techs == 'Wind Offshore')
        OnshoreMask = (Techs == 'Wind Onshore')
        SolarMask = (Techs == 'PV')
        ConvMask = ~(OffshoreMask | OnshoreMask | SolarMask)

        if OffshoreMask.any():
            self.con.ProdLim_Offshore = self.m.addConstr(
                self.var.EGen[OffshoreMask, :, :] <= CapTotal[OffshoreMask, :, :] * self.D.Offwind_scenarios[None, :, :],
                name='ProdLimit_Offshore'
            )
        if OnshoreMask.any():
            self.con.ProdLim_Onshore = self.m.addConstr(
                self.var.EGen[OnshoreMask, :, :] <= CapTotal[OnshoreMask, :, :] * self.D.Onwind_scenarios[None, :, :],
                name='ProdLimit_Onshore'
            )
        if SolarMask.any():
            self.con.ProdLim_Solar = self.m.addConstr(
                self.var.EGen[SolarMask, :, :] <= CapTotal[SolarMask, :, :] * self.D.Solar_scenarios[None, :, :],
                name='ProdLimit_Solar'
            )
        if ConvMask.any():
            self.con.ProdLim_Conv = self.m.addConstr(
                self.var.EGen[ConvMask, :, :] <= CapTotal[ConvMask, :, :],
                name='ProdLimit_Conventional'
            )

        # Energy Balance constraint
        EGen_sum = self.var.EGen.sum(axis=0)    # shape (H, S)
        EDis_sum = self.var.EDis.sum(axis=0)    # shape (H, S)
        EChar_sum = self.var.EChar.sum(axis=0)  # shape (H, S)
        Demand_scaled = self.D.Dem[:H, None]  # shape (H, 1), broadcast to (H, S)

        self.con.Balance = self.m.addConstr(EGen_sum + EDis_sum == Demand_scaled + EChar_sum, name='EnergyBalance')


        #Defining SOC, Charge and Discharge for each scenario and each hour >0
        SOC = self.var.SOC
        EChar = self.var.EChar
        EDis = self.var.EDis
        SOC_prev = SOC[:, :-1, :]  # (U, H-1, S)
        EChar_prev = EChar[:, :-1, :]
        EDis_prev = EDis[:, :-1, :]
        SOC_curr = SOC[:, 1:, :]

        # Defining SOC as results of previous hours
        self.con.SOC_Update = self.m.addConstr(SOC_curr == SOC_prev + EChar_prev - EDis_prev, name='SOC_Update')

        # Defining Initial SOC
        self.con.SOC0 = self.m.addConstr(SOC[:, 0, :] == self.D.StorExi[:, None] + self.StorCap[:,None], name='SOC_Initial')

        # Limit SOC by existing storage capacity and new storage capacity
        self.con.SOCLim = self.m.addConstr(
            SOC <= self.StorCap[:, None, None] + self.D.StorExi[:, None, None],
            name='StorageCapLimit'
        )

        # Limit charge by existing storage capacity and new storage capacity - SOC
        self.con.ECharLim = self.m.addConstr(
            EChar <= self.StorCap[:, None, None] + self.D.StorExi[:, None, None] - SOC,
            name='ECharLimit'
        )
        # Limit Discharge by SOC
        self.con.EDisLim = self.m.addConstr(
            EDis <= SOC,
            name='EDisLimit'
        )

            # Indices of Lignite and Hard Coal
        lignite_index = list(self.D.TechInfo['Technology']).index('Lignite')
        hard_coal_index = list(self.D.TechInfo['Technology']).index('Hard Coal')

        # Ramp rates for Hard Coal and Lignite
        ramp_rates = np.zeros(G)  # Initialize a vector of zeros
        ramp_rates[lignite_index] = 0.1  # 10% ramp rate for Lignite
        ramp_rates[hard_coal_index] = 0.1  # 10% ramp rate for Hard Coal

        # Capacity including new investments, existing, and out capacity
        CapTotal = self.GenCap + self.D.CapExi + self.D.CapOut  # Shape: (G,)

        # Mask for only Lignite and Hard Coal
        ramp_mask = np.array([False] * G)
        ramp_mask[lignite_index] = True
        ramp_mask[hard_coal_index] = True

        # Extract only the technologies with ramp constraints
        ramp_EGen = self.var.EGen[ramp_mask, :, :]  # Shape: (2, H, S)
        ramp_limits = ramp_rates[ramp_mask] * CapTotal[ramp_mask]  # Shape: (2,)

        # Shifted versions of generation for ramping constraints
        ramp_EGen_prev = ramp_EGen[:, :-1, :]  # Shape: (2, H-1, S)
        ramp_EGen_curr = ramp_EGen[:, 1:, :]   # Shape: (2, H-1, S)

        # --- Vectorized Ramping Constraints ---
        # Up-ramp constraints
        self.con.RampUp = self.m.addConstr(
            ramp_EGen_curr - ramp_EGen_prev <= ramp_limits[:, None, None], 
            name='Ramping_Up'
        )

        # Down-ramp constraints
        self.con.RampDown = self.m.addConstr(
            ramp_EGen_prev - ramp_EGen_curr <= ramp_limits[:, None, None], 
            name='Ramping_Down'
        )
        
    
    def _build_objective(self):

        # Vectorized operational cost for generation multiplied by generation
        op_cost = (self.var.EGen * self.D.OpCost[:, None, None]).sum() 

        # Set objective, minimizing total costs
        self.m.setObjective( op_cost, GRB.MINIMIZE)


    def _display_guropby_results(self):
        self.m.setParam('OutputFlag', self.Guroby_results)

    
    def _build_model(self):
        self.m = gp.Model('CapacityProblem')
        self._build_variables()
        self._build_constraints()
        self._build_objective()
        self._display_guropby_results()
        self.m.write("CapacityProblem.lp")
        self.m.optimize()
        self._results()
        if self.Model_results == 1:
            self._extract_results()
            
    def _extract_energy_balance_duals(self):
        # Ensure the model is optimized
        if self.m.status == GRB.OPTIMAL:
            print("Model solved successfully, extracting dual values...")

            # Initialize an empty list to collect duals
            dual_values = []

            # Loop through hours and scenarios to collect dual values
            for h in range(self.P.N_Hours):
                hour_values = []
                for s in range(self.P.N_Scen):
                    # Access the individual constraint
                    constraint = self.con.Balance[h, s]
                    # Extract the dual value (shadow price)
                    hour_values.append(constraint.Pi/1000)
                dual_values.append(hour_values)

            # Convert to a NumPy array and reshape
            dual_array = np.array(dual_values)

            # Store as DataFrame with labels for clarity
            self.res.DA_Prices = pd.DataFrame(
                dual_array,
                columns=[f"Scenario_{s}" for s in range(self.P.N_Scen)],
                index=[f"Hour_{h}" for h in range(self.P.N_Hours)]
            )

            print("Energy Balance Dual values (Day Ahead Prices) extracted successfully.")
        else:
            print("Model did not solve optimally; cannot extract dual values.")

    
    def _results(self):
        self.res.obj = self.m.objVal
        self._extract_energy_balance_duals()

        # Extract results
        self.res.EGen = self.var.EGen.X
        self.res.SOC = self.var.SOC.X
        self.res.EChar = self.var.EChar.X
        self.res.EDis = self.var.EDis.X


        # Create DataFrame for each scenario
        intra_day_arrays = []
        for s in range(self.P.N_Scen):
            print(f"Running Intraday Price Regression for Scenario {s}")
            
            # Fetch the features for the regression model
            DA_prices = self.res.DA_Prices.iloc[:, s].values
            Offshore_CF = self.D.Offwind_scenarios[:, s]
            Onshore_CF = self.D.Onwind_scenarios[:, s]
            Solar_CF = self.D.Solar_scenarios[:, s]
            Demand = self.D.Dem

            # Construct the DataFrame for the regression
            features_df = pd.DataFrame({
                'DA_price': DA_prices,
                'Demand': Demand,
                'Offshore_CF': Offshore_CF,
                'Onshore_CF': Onshore_CF,
                'PV_CF': Solar_CF
            })

            # Predict intraday prices using the pre-trained model
            intraday_prices = self.IDForecaster.predict(features_df)

            intra_day_arrays.append(intraday_prices)

        # Convert to a NumPy array and store it
        self.res.IntraDay_PricesArray = np.column_stack(intra_day_arrays)
        print("Intraday prices stored successfully for all scenarios.")

        # Save as CSV
        np.savetxt(f'Intraday_Prices_All_Scenarios.csv', self.res.IntraDay_PricesArray, delimiter=',')
        
        print("Intraday Price Regression completed for all scenarios.")



        # Calculate RES Share for each scenario
        self.res.RESShare = np.zeros((self.P.N_Scen, 1))
        for s in range(self.P.N_Scen):
            self.res.RESShare[s] =   np.sum(self.res.EGen[:, :, s][self.D.TechInfo['Type'] == 'RES'])/np.sum(self.D.Dem)
         
    
        # Store hourly results in one dataframe
        self.res.DA_EGen_Scenarios = {}

        for s in range(self.P.N_Scen):
            # Extract data for scenario `s`
            e_gen_scenario = self.res.EGen[:, :, s].T  
            soc_scenario = self.res.SOC[:, :, s].T  
            echar_scenario = self.res.EChar[:, :, s].T  
            edis_scenario = self.res.EDis[:, :, s].T  
            DA_prices = self.res.DA_Prices.iloc[:, s].values  # Extract Day Ahead Prices for scenario `s`

            # Extract Wind and Solar Production Factors
            Offwind_scenarios = self.D.Offwind_scenarios[:, s]  
            Onwind_scenarios = self.D.Onwind_scenarios[:, s]  
            Solar_scenarios = self.D.Solar_scenarios[:, s]  

            # Convert to DataFrames
            df_scenario = pd.DataFrame(e_gen_scenario, columns=['Offshore', 'Onshore', 'Biomass', 'Lignite', 'Hard Coal', 'Gas', 'Hydrogren', 'Nuclear', 'Solar PV', 'Hydro', 'Other RES', 'Other Conventional'])
            df_soc = pd.DataFrame(soc_scenario, columns=['Battery_SOC', 'PumpedHydro_SOC'])
            df_echar = pd.DataFrame(echar_scenario, columns=['Battery_Charge', 'PumpedHydro_Charge'])
            df_edis = pd.DataFrame(edis_scenario, columns=['Battery_Discharge', 'PumpedHydro_Discharge'])
            df_demand = pd.DataFrame({'Demand': self.D.Dem})
            df_DA = pd.DataFrame({'DA_Prices': DA_prices})

            # Add Wind & Solar Production Factors
            df_prod_factors = pd.DataFrame({'OffWind': Offwind_scenarios, 'OnWind': Onwind_scenarios, 'Solar': Solar_scenarios})

            # Add Hour column
            df_scenario['Hour'] = list(range(self.P.N_Hours))

            # Merge all DataFrames
            df_complete = pd.concat([df_scenario, df_soc, df_echar, df_edis, df_demand,df_DA, df_prod_factors], axis=1)

            #Reorder columns
            df_complete = df_complete[['Hour', 'Offshore', 'Onshore', 'Biomass', 'Lignite', 'Hard Coal', 'Gas', 'Hydrogren', 'Nuclear', 'Solar PV', 'Hydro', 'Other RES', 'Other Conventional',
                                    'Battery_SOC', 'PumpedHydro_SOC',
                                    'Battery_Charge', 'PumpedHydro_Charge',
                                    'Battery_Discharge', 'PumpedHydro_Discharge',
                                    'Demand', 'OffWind','OnWind', 'Solar', 'DA_Prices']]

            # Store in dictionary
            self.res.DA_EGen_Scenarios[f"DA_Scenario_{s}"] = df_complete
        
        # Initialize the DataFrame to store results
        technology_names = self.D.TechInfo['Technology'].values
        scenario_names = [f"Scenario_{s}" for s in range(self.P.N_Scen)]

        # Create an empty DataFrame with technologies as index and scenarios as columns
        self.res.CapturePrices = pd.DataFrame(index=technology_names, columns=scenario_names)

        # Loop through each scenario and technology to calculate capture prices
        for s in range(self.P.N_Scen):
            for g, tech_name in enumerate(technology_names):
                # Avoid division by zero
                total_generation = np.sum(self.res.EGen[g, :, s])
                if total_generation > 0:
                    capture_price = np.sum(self.res.EGen[g, :, s] * self.res.DA_Prices.iloc[:, s].values) / total_generation
                else:
                    capture_price = 0
                
                # Store in the DataFrame
                self.res.CapturePrices.loc[tech_name, f"Scenario_{s}"] = capture_price

        # Append average DA price to the DataFrame
        self.res.CapturePrices.loc['Average_DA_Price'] = self.res.DA_Prices.mean(axis=0).values

    def _extract_results(self):
        # Display the objective value
        print('Objective value: ', self.m.objVal)
        
        # # Display the new capacity for each type of generator technology
        # print('New capacity for each type of generator technology: ', self.res.CapNew)
        
        # #Display new storage capacity for each type of storage technology
        # print('New storage capacity for each type of storage technology: ', self.res.CapStor)

        # Display RES Share for each scenario
        print('RES Share for each scenario: ', self.res.RESShare)

         # Print constraint info
        # print(f"Total number of constraints: {self.res.TotalConstraints}")
        # print(f"Number of active (binding) constraints: {self.res.ActiveConstraints}")


        

        # Defining the capacity problem class

class RelaxedCapacityProblem():
    def __init__(self, ParametersObj, DataObj, Model_results = 1, Guroby_results = 1):
        self.P = ParametersObj # Parameters
        self.D = DataObj # Data
        self.Model_results = Model_results
        self.Guroby_results = Guroby_results
        self.var = Expando()  # Variables
        self.con = Expando()  # Constraints
        self.res = Expando()  # Results
        self._build_model() 


    def _build_variables(self):
        # Create the variables
        self.var.CapNew = self.m.addMVar((self.P.N_Cap), lb=0)  # New Capacity for each type of generator technology
        self.var.EGen = self.m.addMVar((self.P.N_Cap, self.P.N_Hours, self.P.N_Scen), lb=0)  # Energy Production for each type of generator technology for each hour and scenario
        self.var.CapStor = self.m.addMVar((self.P.N_Stor), lb=0)  # New storage capacity for each type of storage technology
        self.var.SOC = self.m.addMVar((self.P.N_Stor, self.P.N_Hours,self.P.N_Scen), lb=0)  # State of charge for each type of storage technology for each hour and scenario  
        self.var.EChar = self.m.addMVar((self.P.N_Stor, self.P.N_Hours,self.P.N_Scen), lb=0)  # Energy charged for each type of storage technology for each hour and scenario
        self.var.EDis = self.m.addMVar((self.P.N_Stor, self.P.N_Hours,self.P.N_Scen), lb=0)  # Energy discharged for each type of storage technology for each hour and scenario
        self.var.Lambda = self.m.addMVar(self.P.N_Scen, lb=0, name='Lambda') # Lagrange multipliers for each scenario


    def _build_constraints(self):
    
        G, H, S, U = self.P.N_Cap, self.P.N_Hours, self.P.N_Scen, self.P.N_Stor
        

        # Capacity is limited my maximum investable capacity for each technology
        self.con.CapLim = self.m.addConstr(self.var.CapNew <= self.D.CapLim, name='Capacitylimit')
        self.con.CapStorLim = self.m.addConstr(self.var.CapStor <= self.D.StorLim, name='Storage capacity limit')



          # Production limited by sum of new capacity, existing capacity and phased out capacity times the production factor
        
        # Generation for each hour is limited by capacity and production factors
        CapTotal = self.var.CapNew.reshape((G, 1, 1)) + self.D.CapExi.reshape((G, 1, 1)) + self.D.CapOut.reshape((G, 1, 1))

        Techs = self.D.TechInfo['Technology'].values
        OffshoreMask = (Techs == 'Wind Offshore')
        OnshoreMask = (Techs == 'Wind Onshore')
        SolarMask = (Techs == 'PV')
        ConvMask = ~(OffshoreMask | OnshoreMask | SolarMask)

        if OffshoreMask.any():
            self.con.ProdLim_Offshore = self.m.addConstr(
                self.var.EGen[OffshoreMask, :, :] <= CapTotal[OffshoreMask, :, :] * self.D.Offwind_scenarios[None, :, :],
                name='ProdLimit_Offshore'
            )
        if OnshoreMask.any():
            self.con.ProdLim_Onshore = self.m.addConstr(
                self.var.EGen[OnshoreMask, :, :] <= CapTotal[OnshoreMask, :, :] * self.D.Onwind_scenarios[None, :, :],
                name='ProdLimit_Onshore'
            )
        if SolarMask.any():
            self.con.ProdLim_Solar = self.m.addConstr(
                self.var.EGen[SolarMask, :, :] <= CapTotal[SolarMask, :, :] * self.D.Solar_scenarios[None, :, :],
                name='ProdLimit_Solar'
            )
        if ConvMask.any():
            self.con.ProdLim_Conv = self.m.addConstr(
                self.var.EGen[ConvMask, :, :] <= CapTotal[ConvMask, :, :],
                name='ProdLimit_Conventional'
            )

        # Demand and storage charge must be met for each hour and scenario by generation and storage discharge
        EGen_sum = self.var.EGen.sum(axis=0)    # shape (H, S)
        EDis_sum = self.var.EDis.sum(axis=0)    # shape (H, S)
        EChar_sum = self.var.EChar.sum(axis=0)  # shape (H, S)
        Demand_scaled = self.D.Dem[:H, None]  # shape (H, 1), broadcast to (H, S)

        self.con.Balance = self.m.addConstr(EGen_sum + EDis_sum == Demand_scaled + EChar_sum, name='EnergyBalance')

        
        # for h in range(self.P.N_Hours):
        #     for s in range(self.P.N_Scen):
        #         self.con.Balance = self.m.addConstr(gp.quicksum(self.var.EGen[:, h, s]) 
        #                                             #+ gp.quicksum(self.var.EDis[:, h, s]) - gp.quicksum(self.var.EChar[:, h, s])  
        #                                             == self.D.Dem[h], name=f'EnergyBalance_{h}_{s}') 


        # # Defining RES share as a percentage of total energy demand
        # for s in range(self.P.N_Scen):
        #     self.con.RESShare = self.m.addConstr(self.P.delta * gp.quicksum(self.D.Dem[h] for h in range(self.P.N_Hours)) - gp.quicksum(self.var.EGen[g, h,s] for h in range(self.P.N_Hours) for g in range(self.P.N_Cap) if self.D.TechInfo['Type'][g] == 'RES' )  <= self.P.BigM * (1 - self.var.u[s]), name=f'RES share S_{s}')

        #  # Defining how many scenarios are allowed to be violated
        # self.con.ScenViol = self.m.addConstr(gp.quicksum(self.var.u[s] for s in range(self.P.N_Scen))/self.P.N_Scen >= (1- self.P.epsilon), name=f'Scenario violation')

        
        #Defining SOC, Charge and Discharge for each scenario and each hour >0
        SOC = self.var.SOC
        EChar = self.var.EChar
        EDis = self.var.EDis
        SOC_prev = SOC[:, :-1, :]  # (U, H-1, S)
        EChar_prev = EChar[:, :-1, :]
        EDis_prev = EDis[:, :-1, :]
        SOC_curr = SOC[:, 1:, :]

        # Defining SOC as results of previous hours
        self.con.SOC_Update = self.m.addConstr(SOC_curr == SOC_prev + EChar_prev - EDis_prev, name='SOC_Update')

        # Defining Initial SOC
        self.con.SOC0 = self.m.addConstr(SOC[:, 0, :] == self.D.StorExi[:, None], name='SOC_Initial')

        # Limit SOC by existing storage capacity and new storage capacity
        self.con.SOCLim = self.m.addConstr(
            SOC <= self.var.CapStor[:, None, None] + self.D.StorExi[:, None, None],
            name='StorageCapLimit'
        )

        # Limit charge by existing storage capacity and new storage capacity - SOC
        self.con.ECharLim = self.m.addConstr(
            EChar <= self.var.CapStor[:, None, None] + self.D.StorExi[:, None, None] - SOC,
            name='ECharLimit'
        )
        # Limit Discharge by SOC
        self.con.EDisLim = self.m.addConstr(
            EDis <= SOC,
            name='EDisLimit'
        )

            # Indices of Lignite and Hard Coal
        lignite_index = list(self.D.TechInfo['Technology']).index('Lignite')
        hard_coal_index = list(self.D.TechInfo['Technology']).index('Hard Coal')

        # Ramp rates for Hard Coal and Lignite
        ramp_rates = np.zeros(G)  # Initialize a vector of zeros
        ramp_rates[lignite_index] = 0.1  # 10% ramp rate for Lignite
        ramp_rates[hard_coal_index] = 0.1  # 10% ramp rate for Hard Coal

        # Capacity including new investments, existing, and out capacity
        CapTotal = self.var.CapNew + self.D.CapExi + self.D.CapOut  # Shape: (G,)

        # Mask for only Lignite and Hard Coal
        ramp_mask = np.array([False] * G)
        ramp_mask[lignite_index] = True
        ramp_mask[hard_coal_index] = True

        # Extract only the technologies with ramp constraints
        ramp_EGen = self.var.EGen[ramp_mask, :, :]  # Shape: (2, H, S)
        ramp_limits = ramp_rates[ramp_mask] * CapTotal[ramp_mask]  # Shape: (2,)

        # Shifted versions of generation for ramping constraints
        ramp_EGen_prev = ramp_EGen[:, :-1, :]  # Shape: (2, H-1, S)
        ramp_EGen_curr = ramp_EGen[:, 1:, :]   # Shape: (2, H-1, S)

        # --- Vectorized Ramping Constraints ---
        # Up-ramp constraints
        self.con.RampUp = self.m.addConstr(
            ramp_EGen_curr - ramp_EGen_prev <= ramp_limits[:, None, None], 
            name='Ramping_Up'
        )

        # Down-ramp constraints
        self.con.RampDown = self.m.addConstr(
            ramp_EGen_prev - ramp_EGen_curr <= ramp_limits[:, None, None], 
            name='Ramping_Down'
        )
        
    
    def _build_objective(self):
        # Vectorized investment cost for generators multiplied by new capacity
        gen_capex_cost = self.var.CapNew @ self.D.CapCost 

        # Vectorized operational cost for generation multiplied by generation
        op_cost = (self.var.EGen * self.D.OpCost[:, None, None]).sum()

        # Vectorized storage investment cost multiplied by new storage capacity
        stor_cost = self.var.CapStor @ self.D.StorCost  

       # RES indices (where renewable energy technologies are located)
        res_indices = self.D.TechInfo.index[self.D.TechInfo['Type'] == 'RES'].tolist()

        # RES requirement, summed demand times the delta percentage
        RES_requirement = self.P.delta * self.D.Dem.sum()

        # List to collect all scenario penalties
        penalties = []

        # Loop through each scenario to build the expression
        for s in range(self.P.N_Scen):
            # Total RES generation in this scenario
            RES_generated = gp.quicksum(self.var.EGen[g, :, s].sum() for g in res_indices)

            # Lagrangian term for this scenario
            lagrangian_term = self.var.Lambda[s] * (RES_requirement - RES_generated)
            
            # Add to the list
            penalties.append(lagrangian_term)

        # Set objective, minimizing total costs
        self.m.setObjective(gen_capex_cost + op_cost/self.P.N_Scen + stor_cost, GRB.MINIMIZE)
        self.m.setObjective(self.m.getObjective() + gp.quicksum(penalties), GRB.MINIMIZE)

    def _display_guropby_results(self):
        self.m.setParam('OutputFlag', self.Guroby_results)

    
    def _build_model(self):
        self.m = gp.Model('CapacityProblem')
        self._build_variables()
        self._build_constraints()
        self._build_objective()
        self._display_guropby_results()
        self.m.write("CapacityProblem.lp")
        self.m.optimize()
        self._results()
        if self.Model_results == 1:
            self._extract_results()
            

    
    def _results(self):
        self.res.obj = self.m.objVal
        self.res.CapNew = self.var.CapNew.X
        self.res.EGen = self.var.EGen.X
        self.res.CapStor = self.var.CapStor.X
        self.res.SOC = self.var.SOC.X
        self.res.EChar = self.var.EChar.X
        self.res.EDis = self.var.EDis.X
        self.res.u = self.var.u.X
        # Calculate RES Share for each scenario
        self.res.RESShare = np.zeros((self.P.N_Scen, 1))
        for s in range(self.P.N_Scen):
            self.res.RESShare[s] =   np.sum(self.res.EGen[:, :, s][self.D.TechInfo['Type'] == 'RES'])/np.sum(self.D.Dem)
         
        # Total number of constraints
        self.res.TotalConstraints = self.m.NumConstrs

        # Count number of active constraints
        def count_active_constraints(model, tol=1e-6):
            model.update()  # Make sure model is up to date
            active = 0
            for c in model.getConstrs():
                slack = c.getAttr("Slack")
                sense = c.Sense

                if sense == '=' and abs(slack) <= tol:
                    active += 1
                elif sense == '<' and abs(slack) <= tol:
                    active += 1
                elif sense == '>' and abs(slack) <= tol:
                    active += 1
            return active


        self.res.ActiveConstraints = count_active_constraints(self.m)

        #Create Dataframe for new capacity
        self.res.CapNew = pd.DataFrame({
            'Plant': ['Offshore', 'Onshore', 'Biomass', 'Lignite', 'Hard Coal', 'Gas', 'Hydrogren', 'Nuclear', 'Solar PV', 'Hydro', 'Other RES', 'Other Conventional'],
            'CapNew[GW]': self.res.CapNew,
            'CapExi[GW]': self.D.CapExi,
            'CapOut[GW]': self.D.CapOut
        })

        #Create Dataframe for new storage capacity
        self.res.CapStor = pd.DataFrame({
            'Plant': ['Battery', 'Pumped Hydro'],
            'CapStor[GWh]': self.res.CapStor
        })

        # Create DataFrame for scenario violations
        self.res.Violation_Scenarios = pd.DataFrame({
            'Scenario': list(range(self.P.N_Scen)),
            'Violated (1=No, 0=Yes)': self.res.u
        })
        
        
    

        self.res.EGen_Scenarios = {}

        for s in range(self.P.N_Scen):
            # Extract data for scenario `s`
            e_gen_scenario = self.res.EGen[:, :, s].T  
            soc_scenario = self.res.SOC[:, :, s].T  
            echar_scenario = self.res.EChar[:, :, s].T  
            edis_scenario = self.res.EDis[:, :, s].T  

            # Extract Wind and Solar Production Factors
            Offwind_scenarios = self.D.Offwind_scenarios[:, s]  
            Onwind_scenarios = self.D.Onwind_scenarios[:, s]  
            Solar_scenarios = self.D.Solar_scenarios[:, s]  

            # Convert to DataFrames
            df_scenario = pd.DataFrame(e_gen_scenario, columns=['Offshore', 'Onshore', 'Biomass', 'Lignite', 'Hard Coal', 'Gas', 'Hydrogren', 'Nuclear', 'Solar PV', 'Hydro', 'Other RES', 'Other Conventional'])
            df_soc = pd.DataFrame(soc_scenario, columns=['Battery_SOC', 'PumpedHydro_SOC'])
            df_echar = pd.DataFrame(echar_scenario, columns=['Battery_Charge', 'PumpedHydro_Charge'])
            df_edis = pd.DataFrame(edis_scenario, columns=['Battery_Discharge', 'PumpedHydro_Discharge'])
            df_demand = pd.DataFrame({'Demand': self.D.Dem})

            # Add Wind & Solar Production Factors
            df_prod_factors = pd.DataFrame({'OffWind': Offwind_scenarios, 'OnWind': Onwind_scenarios, 'Solar': Solar_scenarios})

            # Add Hour column
            df_scenario['Hour'] = list(range(self.P.N_Hours))

            # Merge all DataFrames
            df_complete = pd.concat([df_scenario, df_soc, df_echar, df_edis, df_demand, df_prod_factors], axis=1)

            # Reorder columns
            df_complete = df_complete[['Hour', 'Offshore', 'Onshore', 'Biomass', 'Lignite', 'Hard Coal', 'Gas', 'Hydrogren', 'Nuclear', 'Solar PV', 'Hydro', 'Other RES', 'Other Conventional',
                                    'Battery_SOC', 'PumpedHydro_SOC',
                                    'Battery_Charge', 'PumpedHydro_Charge',
                                    'Battery_Discharge', 'PumpedHydro_Discharge',
                                    'Demand', 'OffWind','OnWind', 'Solar']]

            # Store in dictionary
            self.res.EGen_Scenarios[f"Scenario_{s}"] = df_complete
    

    def _extract_results(self):
        # Display the objective value
        print('Objective value: ', self.m.objVal)
        # Display the new capacity for each type of generator technology
        print('New capacity for each type of generator technology: ', self.res.CapNew)
        
        #Display new storage capacity for each type of storage technology
        print('New storage capacity for each type of storage technology: ', self.res.CapStor)

        # Display RES Share for each scenario
        print('RES Share for each scenario: ', self.res.RESShare)

         # Print constraint info
        print(f"Total number of constraints: {self.res.TotalConstraints}")
        print(f"Number of active (binding) constraints: {self.res.ActiveConstraints}")


# Defining the Day Ahead Problem class

class BatteryOptimization():
    def __init__(self, ParametersObj, DataObj, DayAheadPrices,IDPrices, Model_results = 1, Guroby_results = 1):
        self.P = ParametersObj # Parameters
        self.D = DataObj # Data
        self.Model_results = int(np.array(Model_results).flatten()[0])
        self.Guroby_results = Guroby_results
        self.DAPrices = DayAheadPrices
        self.IDPrices = IDPrices
        self.var = Expando()  # Variables
        self.con = Expando()  # Constraints
        self.res = Expando()  # Results
        self._build_model() 


    def _build_variables(self):
        # Create the variables
        self.var.DADC = self.m.addMVar((self.P.N_Hours, self.P.N_Scen), lb=0)
        self.var.DAC = self.m.addMVar((self.P.N_Hours, self.P.N_Scen), lb=0)
        self.var.IDDC = self.m.addMVar((self.P.N_Hours, self.P.N_Scen), lb=0)
        self.var.IDC = self.m.addMVar((self.P.N_Hours, self.P.N_Scen), lb=0)
        self.var.SOC = self.m.addMVar((self.P.N_Hours, self.P.N_Scen), lb=0)

    def _build_constraints(self):

        # Define SOC

        for h in range(self.P.N_Hours):
            for s in range(self.P.N_Scen):
                if h == 0:
                    self.con.SOC = self.m.addConstr(self.var.SOC[h, s] == 0, name=f"Initial_SOC_{h}_{s}")
                else:
                    self.con.SOC = self.m.addConstr(self.var.SOC[h, s] == self.var.SOC[h-1, s] + (self.var.DAC[h-1, s] + self.var.IDC[h-1,s])*0.9 - self.var.DADC[h-1, s] - self.var.IDDC[h-1,s], name=f"SOC_{h}_{s}")

        
        for h in range(self.P.N_Hours):
            for s in range(self.P.N_Scen):

                # Limit SOC by capacity
                self.con.SOC = self.m.addConstr(self.var.SOC[h, s] <= self.P.Capacity, name=f"SOCLimit_{h}_{s}")

                # Limit charging and discharging by power
                self.con.ChargePower = self.m.addConstr(self.var.DAC[h, s] + self.var.IDC[h, s] <= self.P.Power, name=f"PowerChargeLimite_{h}_{s}")
                self.con.DischargePower = self.m.addConstr(self.var.DADC[h, s] + self.var.IDDC[h, s] <= self.P.Power, name=f"PowerDischargeLimit_{h}_{s}")

                # Limit charging by available capacity
                self.con.ChargeCapacity = self.m.addConstr(self.var.DAC[h, s]*0.9 + self.var.IDC[h,s]*0.9 <= self.P.Capacity - self.var.SOC[h, s], name=f"ChargeLimit_{h}_{s}")

                #Limit discharging by SOC
                self.con.DischargeSOC = self.m.addConstr(self.var.DADC[h, s] + self.var.DADC[h,s] <= self.var.SOC[h, s], name=f"DischargeLimit_{h}_{s}")



        
    
    def _build_objective(self):

        # Battery Revenue
        DARevenue = gp.quicksum(self.var.DADC[h, s] * 0.9 * self.DAPrices[h, s] for h in range(self.P.N_Hours) for s in range(self.P.N_Scen))

        IDRevenue = gp.quicksum(self.var.IDDC[h, s] * 0.9 * self.IDPrices[h, s] for h in range(self.P.N_Hours) for s in range(self.P.N_Scen))

        Revenue = DARevenue + IDRevenue

        # Battery Costs
        Op_cost_DA = gp.quicksum(self.var.DAC[h, s] * self.DAPrices[h, s] for h in range(self.P.N_Hours) for s in range(self.P.N_Scen))
        Op_cost_ID = gp.quicksum(self.var.IDC[h, s] * self.IDPrices[h, s] for h in range(self.P.N_Hours) for s in range(self.P.N_Scen))
        Op_cost = Op_cost_DA + Op_cost_ID

        # Set objective, minimizing total costs
        self.m.setObjective((Revenue - Op_cost)/self.P.N_Scen, GRB.MAXIMIZE)


    def _display_guropby_results(self):
        self.m.setParam('OutputFlag', self.Guroby_results)

    
    def _build_model(self):
        self.m = gp.Model('CapacityProblem')
        self._build_variables()
        self._build_constraints()
        self._build_objective()
        self._display_guropby_results()
        self.m.write("CapacityProblem.lp")
        self.m.optimize()
        self._results()
        if self.Model_results == 1:
            self._extract_results()
            
   

    
    def _results(self):
        self.res.obj = self.m.objVal
        self.res.DADC = self.var.DADC.X
        self.res.DAC = self.var.DAC.X
        self.res.IDDC = self.var.IDDC.X
        self.res.IDC = self.var.IDC.X

        self.res.SOC = self.var.SOC.X

        #Save results in a DataFrame
        # Initialize a dictionary to store each scenario's DataFrame
        self.res.OptimizationResults = {}

        # Loop through each scenario and create a DataFrame
        for s in range(self.P.N_Scen):
            # Create DataFrames for discharge, charge, and SOC
            df_discharge = pd.DataFrame(self.res.DADC[:, s], columns=['DADischarge'])
            df_charge = pd.DataFrame(self.res.DAC[:, s], columns=['DACharge'])
            df_IDdischarge = pd.DataFrame(self.res.IDDC[:, s], columns=['IDDischarge'])
            df_IDcharge = pd.DataFrame(self.res.IDC[:, s], columns=['IDCharge'])
            df_soc = pd.DataFrame(self.res.SOC[:, s], columns=['SOC'])
            df_prices = pd.DataFrame(self.DAPrices[:, s], columns=['DA_Prices'])
            df_IDPrices = pd.DataFrame(self.IDPrices[:, s], columns=['ID_Prices'])

            # Merge them into one DataFrame for the scenario
            df_scenario = pd.concat([df_discharge, df_charge, df_IDdischarge,df_IDcharge, df_soc, df_prices,df_IDPrices], axis=1)

            # Store it in the dictionary with a descriptive name
            self.res.OptimizationResults[f"Scenario_{s}"] = df_scenario
        
        # --- Calculation of Revenue, Cost, and Profit ---
        DARevenue = (self.res.DADC * 0.9 * self.DAPrices).sum(axis=0)
        IDRevenue = (self.res.IDDC * 0.9 * self.IDPrices).sum(axis=0)
        DACost = (self.res.DAC * self.DAPrices).sum(axis=0)
        IDCost = (self.res.IDC * self.IDPrices).sum(axis=0)
        TotalProfit = (DARevenue + IDRevenue) - (DACost + IDCost)
        RelativeProfit = TotalProfit / self.P.Power/1000

        # Store in a DataFrame
        self.res.FinancialSummary = pd.DataFrame({
            'DA_Revenue': DARevenue,
            'ID_Revenue': IDRevenue,
            'DA_Cost': DACost,
            'ID_Cost': IDCost,
            'Total_Profit': TotalProfit,
            'Profit [EUR/kW]': RelativeProfit
        }, index=[f"Scenario_{s}" for s in range(self.P.N_Scen)])

        # Save to CSV
        self.res.FinancialSummary.to_csv('BatteryOptimization_FinancialSummary.csv')
        print("Financial Summary saved successfully.")

        
        
        

        
        
    
    
        
        

        # # Extract results
        # self.res.EGen = self.var.EGen.X
        # self.res.SOC = self.var.SOC.X
        # self.res.EChar = self.var.EChar.X
        # self.res.EDis = self.var.EDis.X



       

    def _extract_results(self):
        # Display the objective value
        print('Objective value: ', self.m.objVal)
        print('SOC', self.var.SOC.X)
        
       
