import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
from scipy.stats import norm
from DataHandling import *
import matplotlib.pyplot as plt

# Class for external input data
class InputData():
    def __init__(self, CapCost, OpCost, TechInfo, StorCost, CapLim,CapExi,CapOut,Dem,EtaCha,EtaDis,StorExi, Offwind_scenarios, Onwind_scenarios, Solar_scenarios, StorLim):
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
        

# Class for model parameters
class Parameters():
    def __init__(self, epsilon, delta, N_Hours, N_Cap, N_Stor, N_Scen, BigM):
        self.epsilon = epsilon
        self.delta = delta
        self.N_Hours = N_Hours
        self.N_Cap = N_Cap
        self.N_Stor = N_Stor
        self.N_Scen = N_Scen
        self.BigM = BigM
        

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


# Defining the optimization model class

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
        ramp_rates = {
            lignite_index: 0.1,  # 7% of capacity
            hard_coal_index: 0.1  # 8% of capacity
        }

        # Capacity including new investments, existing, and out capacity
        #CapTotal = self.var.CapNew.reshape((G, 1, 1)) + self.D.CapExi.reshape((G, 1, 1)) + self.D.CapOut.reshape((G, 1, 1))

        # --- Ramping Constraints ---
        for g, ramp_rate in ramp_rates.items():
            for s in range(S):
                for t in range(1, H):  # Start from 1 to prevent indexing error
                    # Up-ramping constraint
                    self.m.addConstr(
                        self.var.EGen[g, t, s] - self.var.EGen[g, t - 1, s] <= ramp_rate * CapTotal[g, 0, 0],
                        name=f"Ramping_Up_{self.D.TechInfo['Technology'][g]}_S{s}_H{t}"
                    )

                    # Down-ramping constraint
                    self.m.addConstr(
                        self.var.EGen[g, t - 1, s] - self.var.EGen[g, t, s] <= ramp_rate * CapTotal[g, 0, 0],
                        name=f"Ramping_Down_{self.D.TechInfo['Technology'][g]}_S{s}_H{t}"
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
            'CapNew[MW]': self.res.CapNew,
            'CapExi[MW]': self.D.CapExi,
            'CapOut[MW]': self.D.CapOut
        })

        #Create Dataframe for new storage capacity
        self.res.CapStor = pd.DataFrame({
            'Plant': ['Battery', 'Pumped Hydro'],
            'CapStor[MWh]': self.res.CapStor
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

        