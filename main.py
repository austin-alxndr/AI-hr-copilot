import pandas as pd

ter_thresholds = {
    'A': [
{'lower_threshold': 0.0, 'upper_threshold': 5400000.0, 'tax_rate': 0.0},
  {'lower_threshold': 5400001.0,
   'upper_threshold': 5650000.0,
   'tax_rate': 0.0025},
  {'lower_threshold': 5650001.0,
   'upper_threshold': 5950000.0,
   'tax_rate': 0.005},
  {'lower_threshold': 5950001.0,
   'upper_threshold': 6300000.0,
   'tax_rate': 0.0075},
  {'lower_threshold': 6300001.0,
   'upper_threshold': 6750000.0,
   'tax_rate': 0.01},
  {'lower_threshold': 6750001.0,
   'upper_threshold': 7500000.0,
   'tax_rate': 0.0125},
  {'lower_threshold': 7500001.0,
   'upper_threshold': 8550000.0,
   'tax_rate': 0.015},
  {'lower_threshold': 8550001.0,
   'upper_threshold': 9650000.0,
   'tax_rate': 0.0175},
  {'lower_threshold': 9650001.0,
   'upper_threshold': 10050000.0,
   'tax_rate': 0.02},
  {'lower_threshold': 10050001.0,
   'upper_threshold': 10350000.0,
   'tax_rate': 0.0225},
  {'lower_threshold': 10350001.0,
   'upper_threshold': 10700000.0,
   'tax_rate': 0.025},
  {'lower_threshold': 10700001.0,
   'upper_threshold': 11050000.0,
   'tax_rate': 0.03},
  {'lower_threshold': 11050001.0,
   'upper_threshold': 11600000.0,
   'tax_rate': 0.035},
  {'lower_threshold': 11600001.0,
   'upper_threshold': 12500000.0,
   'tax_rate': 0.04},
  {'lower_threshold': 12500001.0,
   'upper_threshold': 13750000.0,
   'tax_rate': 0.05},
  {'lower_threshold': 13750001.0,
   'upper_threshold': 15100000.0,
   'tax_rate': 0.06},
  {'lower_threshold': 15100001.0,
   'upper_threshold': 16950000.0,
   'tax_rate': 0.07},
  {'lower_threshold': 16950001.0,
   'upper_threshold': 19750000.0,
   'tax_rate': 0.08},
  {'lower_threshold': 19750001.0,
   'upper_threshold': 24150000.0,
   'tax_rate': 0.09},
  {'lower_threshold': 24150001.0,
   'upper_threshold': 26450000.0,
   'tax_rate': 0.1},
  {'lower_threshold': 26450001.0,
   'upper_threshold': 28000000.0,
   'tax_rate': 0.11},
  {'lower_threshold': 28000001.0,
   'upper_threshold': 30050000.0,
   'tax_rate': 0.12},
  {'lower_threshold': 30050001.0,
   'upper_threshold': 32400000.0,
   'tax_rate': 0.13},
  {'lower_threshold': 32400001.0,
   'upper_threshold': 35400000.0,
   'tax_rate': 0.14},
  {'lower_threshold': 35400001.0,
   'upper_threshold': 39100000.0,
   'tax_rate': 0.15},
  {'lower_threshold': 39100001.0,
   'upper_threshold': 43850000.0,
   'tax_rate': 0.16},
  {'lower_threshold': 43850001.0,
   'upper_threshold': 47800000.0,
   'tax_rate': 0.17},
  {'lower_threshold': 47800001.0,
   'upper_threshold': 51400000.0,
   'tax_rate': 0.18},
  {'lower_threshold': 51400001.0,
   'upper_threshold': 56300000.0,
   'tax_rate': 0.19},
  {'lower_threshold': 56300001.0,
   'upper_threshold': 62200000.0,
   'tax_rate': 0.2},
  {'lower_threshold': 62200001.0,
   'upper_threshold': 68600000.0,
   'tax_rate': 0.21},
  {'lower_threshold': 68600001.0,
   'upper_threshold': 77500000.0,
   'tax_rate': 0.22},
  {'lower_threshold': 77500001.0,
   'upper_threshold': 89000000.0,
   'tax_rate': 0.23},
  {'lower_threshold': 89000001.0,
   'upper_threshold': 103000000.0,
   'tax_rate': 0.24},
  {'lower_threshold': 103000001.0,
   'upper_threshold': 125000000.0,
   'tax_rate': 0.25},
  {'lower_threshold': 125000001.0,
   'upper_threshold': 157000000.0,
   'tax_rate': 0.26},
  {'lower_threshold': 157000001.0,
   'upper_threshold': 206000000.0,
   'tax_rate': 0.27},
  {'lower_threshold': 206000001.0,
   'upper_threshold': 337000000.0,
   'tax_rate': 0.28},
  {'lower_threshold': 337000001.0,
   'upper_threshold': 454000000.0,
   'tax_rate': 0.29},
  {'lower_threshold': 454000001.0,
   'upper_threshold': 550000000.0,
   'tax_rate': 0.3},
  {'lower_threshold': 550000001.0,
   'upper_threshold': 695000000.0,
   'tax_rate': 0.31},
  {'lower_threshold': 695000001.0,
   'upper_threshold': 910000000.0,
   'tax_rate': 0.32},
  {'lower_threshold': 910000001.0,
   'upper_threshold': 1400000000.0,
   'tax_rate': 0.33},
  {'lower_threshold': 1400000001.0,
   'upper_threshold': 9999999999.0,
   'tax_rate': 0.34}],
    'B': [{'lower_threshold': 0.0, 'upper_threshold': 6200000.0, 'tax_rate': 0.0},
  {'lower_threshold': 6200001.0,
   'upper_threshold': 6500000.0,
   'tax_rate': 0.0025},
  {'lower_threshold': 6500001.0,
   'upper_threshold': 6850000.0,
   'tax_rate': 0.005},
  {'lower_threshold': 6850001.0,
   'upper_threshold': 7300000.0,
   'tax_rate': 0.0075},
  {'lower_threshold': 7300001.0,
   'upper_threshold': 9200000.0,
   'tax_rate': 0.01},
  {'lower_threshold': 9200001.0,
   'upper_threshold': 10750000.0,
   'tax_rate': 0.015},
  {'lower_threshold': 10750001.0,
   'upper_threshold': 11250000.0,
   'tax_rate': 0.02},
  {'lower_threshold': 11250001.0,
   'upper_threshold': 11600000.0,
   'tax_rate': 0.025},
  {'lower_threshold': 11600001.0,
   'upper_threshold': 12600000.0,
   'tax_rate': 0.03},
  {'lower_threshold': 12600001.0,
   'upper_threshold': 13600000.0,
   'tax_rate': 0.04},
  {'lower_threshold': 13600001.0,
   'upper_threshold': 14950000.0,
   'tax_rate': 0.05},
  {'lower_threshold': 14950001.0,
   'upper_threshold': 16400000.0,
   'tax_rate': 0.06},
  {'lower_threshold': 16400001.0,
   'upper_threshold': 18450000.0,
   'tax_rate': 0.07},
  {'lower_threshold': 18450001.0,
   'upper_threshold': 21850000.0,
   'tax_rate': 0.08},
  {'lower_threshold': 21850001.0,
   'upper_threshold': 26000000.0,
   'tax_rate': 0.09},
  {'lower_threshold': 26000001.0,
   'upper_threshold': 27700000.0,
   'tax_rate': 0.1},
  {'lower_threshold': 27700001.0,
   'upper_threshold': 29350000.0,
   'tax_rate': 0.11},
  {'lower_threshold': 29350001.0,
   'upper_threshold': 31450000.0,
   'tax_rate': 0.12},
  {'lower_threshold': 31450001.0,
   'upper_threshold': 33950000.0,
   'tax_rate': 0.13},
  {'lower_threshold': 33950001.0,
   'upper_threshold': 37100000.0,
   'tax_rate': 0.14},
  {'lower_threshold': 37100001.0,
   'upper_threshold': 41100000.0,
   'tax_rate': 0.15},
  {'lower_threshold': 41100001.0,
   'upper_threshold': 45800000.0,
   'tax_rate': 0.16},
  {'lower_threshold': 45800001.0,
   'upper_threshold': 49500000.0,
   'tax_rate': 0.17},
  {'lower_threshold': 49500001.0,
   'upper_threshold': 53800000.0,
   'tax_rate': 0.18},
  {'lower_threshold': 53800001.0,
   'upper_threshold': 58500000.0,
   'tax_rate': 0.19},
  {'lower_threshold': 58500001.0,
   'upper_threshold': 64000000.0,
   'tax_rate': 0.2},
  {'lower_threshold': 64000001.0,
   'upper_threshold': 71000000.0,
   'tax_rate': 0.21},
  {'lower_threshold': 71000001.0,
   'upper_threshold': 80000000.0,
   'tax_rate': 0.22},
  {'lower_threshold': 80000001.0,
   'upper_threshold': 93000000.0,
   'tax_rate': 0.23},
  {'lower_threshold': 93000001.0,
   'upper_threshold': 109000000.0,
   'tax_rate': 0.24},
  {'lower_threshold': 109000001.0,
   'upper_threshold': 129000000.0,
   'tax_rate': 0.25},
  {'lower_threshold': 129000001.0,
   'upper_threshold': 163000000.0,
   'tax_rate': 0.26},
  {'lower_threshold': 163000001.0,
   'upper_threshold': 211000000.0,
   'tax_rate': 0.27},
  {'lower_threshold': 211000001.0,
   'upper_threshold': 374000000.0,
   'tax_rate': 0.28},
  {'lower_threshold': 374000001.0,
   'upper_threshold': 459000000.0,
   'tax_rate': 0.29},
  {'lower_threshold': 459000001.0,
   'upper_threshold': 555000000.0,
   'tax_rate': 0.3},
  {'lower_threshold': 555000001.0,
   'upper_threshold': 704000000.0,
   'tax_rate': 0.31},
  {'lower_threshold': 704000001.0,
   'upper_threshold': 957000000.0,
   'tax_rate': 0.32},
  {'lower_threshold': 957000001.0,
   'upper_threshold': 1405000000.0,
   'tax_rate': 0.33},
  {'lower_threshold': 1405000001.0,
   'upper_threshold': 9999999999.0,
   'tax_rate': 0.34}],
    'C': [{'lower_threshold': 0.0, 'upper_threshold': 6600000.0, 'tax_rate': 0.0},
  {'lower_threshold': 6600001.0,
   'upper_threshold': 6950000.0,
   'tax_rate': 0.0025},
  {'lower_threshold': 6950001.0,
   'upper_threshold': 7350000.0,
   'tax_rate': 0.005},
  {'lower_threshold': 7350001.0,
   'upper_threshold': 7800000.0,
   'tax_rate': 0.0075},
  {'lower_threshold': 7800001.0,
   'upper_threshold': 8850000.0,
   'tax_rate': 0.01},
  {'lower_threshold': 8850001.0,
   'upper_threshold': 9800000.0,
   'tax_rate': 0.0125},
  {'lower_threshold': 9800001.0,
   'upper_threshold': 10950000.0,
   'tax_rate': 0.015},
  {'lower_threshold': 10950001.0,
   'upper_threshold': 11200000.0,
   'tax_rate': 0.0175},
  {'lower_threshold': 11200001.0,
   'upper_threshold': 12050000.0,
   'tax_rate': 0.02},
  {'lower_threshold': 12050001.0,
   'upper_threshold': 12950000.0,
   'tax_rate': 0.03},
  {'lower_threshold': 12950001.0,
   'upper_threshold': 14150000.0,
   'tax_rate': 0.04},
  {'lower_threshold': 14150001.0,
   'upper_threshold': 15550000.0,
   'tax_rate': 0.05},
  {'lower_threshold': 15550001.0,
   'upper_threshold': 17050000.0,
   'tax_rate': 0.06},
  {'lower_threshold': 17050001.0,
   'upper_threshold': 19500000.0,
   'tax_rate': 0.07},
  {'lower_threshold': 19500001.0,
   'upper_threshold': 22700000.0,
   'tax_rate': 0.08},
  {'lower_threshold': 22700001.0,
   'upper_threshold': 26600000.0,
   'tax_rate': 0.09},
  {'lower_threshold': 26600001.0,
   'upper_threshold': 28100000.0,
   'tax_rate': 0.1},
  {'lower_threshold': 26600001.0,
   'upper_threshold': 30100000.0,
   'tax_rate': 0.11},
  {'lower_threshold': 30100001.0,
   'upper_threshold': 32600000.0,
   'tax_rate': 0.12},
  {'lower_threshold': 32600001.0,
   'upper_threshold': 35400000.0,
   'tax_rate': 0.13},
  {'lower_threshold': 35400001.0,
   'upper_threshold': 38900000.0,
   'tax_rate': 0.14},
  {'lower_threshold': 38900001.0,
   'upper_threshold': 43000000.0,
   'tax_rate': 0.15},
  {'lower_threshold': 43000001.0,
   'upper_threshold': 47400000.0,
   'tax_rate': 0.16},
  {'lower_threshold': 47400001.0,
   'upper_threshold': 51200000.0,
   'tax_rate': 0.17},
  {'lower_threshold': 51200001.0,
   'upper_threshold': 55800000.0,
   'tax_rate': 0.18},
  {'lower_threshold': 55800001.0,
   'upper_threshold': 60400000.0,
   'tax_rate': 0.19},
  {'lower_threshold': 60400001.0,
   'upper_threshold': 66700000.0,
   'tax_rate': 0.2},
  {'lower_threshold': 66700001.0,
   'upper_threshold': 74500000.0,
   'tax_rate': 0.21},
  {'lower_threshold': 74500001.0,
   'upper_threshold': 83200000.0,
   'tax_rate': 0.22},
  {'lower_threshold': 83200001.0,
   'upper_threshold': 95600000.0,
   'tax_rate': 0.23},
  {'lower_threshold': 95600001.0,
   'upper_threshold': 110000000.0,
   'tax_rate': 0.24},
  {'lower_threshold': 110000001.0,
   'upper_threshold': 134000000.0,
   'tax_rate': 0.25},
  {'lower_threshold': 134000001.0,
   'upper_threshold': 169000000.0,
   'tax_rate': 0.26},
  {'lower_threshold': 169000001.0,
   'upper_threshold': 221000000.0,
   'tax_rate': 0.27},
  {'lower_threshold': 221000001.0,
   'upper_threshold': 390000000.0,
   'tax_rate': 0.28},
  {'lower_threshold': 390000001.0,
   'upper_threshold': 463000000.0,
   'tax_rate': 0.29},
  {'lower_threshold': 463000001.0,
   'upper_threshold': 561000000.0,
   'tax_rate': 0.3},
  {'lower_threshold': 561000001.0,
   'upper_threshold': 709000000.0,
   'tax_rate': 0.31},
  {'lower_threshold': 709000001.0,
   'upper_threshold': 965000000.0,
   'tax_rate': 0.32},
  {'lower_threshold': 965000001.0,
   'upper_threshold': 1419000000.0,
   'tax_rate': 0.33},
  {'lower_threshold': 1419000001.0,
   'upper_threshold': 9999999999.0,
   'tax_rate': 0.34}]
    }

####################  Define custom functions w/in Python #########################

def calculate_ter(marriage_status, dependencies):
    if not marriage_status:
        if dependencies <= 1:
            return "A"
        elif dependencies <= 3:
            return "B"
    else:
        if dependencies == 0:
            return "A"  # Correctly handles married with no dependencies
        elif dependencies <= 2:
            return "B"
        elif dependencies == 3:
            return "C" 
    return "C"

# Calculate taxable gross salary
def calculate_gross_salary(monthly_salary):
    bonus = (monthly_salary * 0.24 / 100) + (monthly_salary * 0.3 / 100)
    additional = monthly_salary * 4 / 100 if monthly_salary <= 10000000 else 480000
    return monthly_salary + bonus + additional

# Find Tax Rate using dictionary ter_thresholds
def find_tax_rate(gross_salary, classification):
    for range in ter_thresholds[classification]:
        if range["lower_threshold"] <= gross_salary <= range["upper_threshold"]:
            return range["tax_rate"]
    return 0

# Calculate Salary Tax 
def calculate_salary_tax(gross_salary, tax_rate):
    return gross_salary * tax_rate

# Calculate Net Salary
def calculate_nett_salary1(monthly_salary, salary_tax):
    bpjs_kes_deduction = monthly_salary * 0.01 if monthly_salary < 12000000 else 120000
    bpjs_tk_deduction = monthly_salary * 2 / 100
    conditional_deduction = monthly_salary * 1 / 100 if monthly_salary < 10042300 else 100423
    nett_salary = monthly_salary - bpjs_kes_deduction - bpjs_tk_deduction - conditional_deduction - salary_tax
    return nett_salary

# Main python function to calculate MONTHLY to NETT

def tax_calculator(monthly_salary, marriage_status, dependencies):
    ter = calculate_ter(marriage_status, dependencies)
    gross_salary = calculate_gross_salary(monthly_salary)
    tax_rate = find_tax_rate(gross_salary, ter)
    salary_tax = calculate_salary_tax(gross_salary, tax_rate)
    nett_salary = calculate_nett_salary1(monthly_salary, salary_tax)
    return {
        "TER": ter,
        "Gross Salary": f"IDR {gross_salary:,.0f}",
        "Tax Rate (%)": f"{tax_rate:.2%}",
        "Salary Tax": f"IDR {salary_tax:,.0f}",
        "Nett Salary": f"IDR {nett_salary:,.0f}"
    }

# Main python function to calculate NETT to MONTHLY

def nett_to_monthly_calculator(nett_salary, marriage_status, dependencies):
    # Implement the logic to calculate the monthly salary based on the nett salary
    # You can use a loop or a mathematical formula to find the monthly salary that results in the given nett salary
    # Here's a simple example using a loop:
    monthly_salary = nett_salary
    while True:
        result = tax_calculator(monthly_salary, marriage_status, dependencies)
        calculated_nett_salary = float(result["Nett Salary"].replace("IDR ", "").replace(",", ""))
        if abs(calculated_nett_salary - nett_salary) < 1:
            break
        monthly_salary += 1

    return {
        "Nett Salary": f"IDR {nett_salary:,.0f}",
        "Monthly Salary": f"IDR {monthly_salary:,.0f}",
        "TER": result["TER"],
        "Gross Salary": result["Gross Salary"],
        "Tax Rate (%)": result["Tax Rate (%)"],
        "Salary Tax": result["Salary Tax"]
    }

############################## Make Custom Tool ###############################

# Langchain-specific code
from langchain.agents import AgentExecutor
from langchain.agents import OpenAIFunctionsAgent
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Type
import os

class TaxCalculatorInput(BaseModel):
    monthly_salary: float = Field(description="Monthly salary as a number")
    marriage_status: bool = Field(description="Marriage status as a boolean (True for married, False for unmarried)")
    dependencies: int = Field(description="Number of dependencies as an integer")

class TaxCalculatorTool(BaseTool):
    name = "tax_calculator"
    description = """
        Used to calculate taxes based on monthly salary, marriage status, and number of dependencies.
        Enter the monthly salary as a number, marriage status as a boolean (True for married, False for unmarried),
        and the number of dependencies as an integer. The output will be the calculated tax information.
        """
    args_schema: Type[BaseModel] = TaxCalculatorInput

    def _run(self, monthly_salary: float, marriage_status: bool, dependencies: int):
        result = tax_calculator(monthly_salary, marriage_status, dependencies)
        return str(result)

    def _arun(self, monthly_salary: float, marriage_status: bool, dependencies: int):
        raise NotImplementedError("tax_calculator does not support asynchronous")
    
class NettToMonthlySalaryInput(BaseModel):
    nett_salary: float = Field(description="Desired nett salary as a number")
    marriage_status: bool = Field(description="Marriage status as a boolean (True for married, False for unmarried)")
    dependencies: int = Field(description="Number of dependencies as an integer")

class NettToMonthlySalaryTool(BaseTool):
    name = "nett_to_monthly_calculator"
    description = """
        Used to calculate the monthly salary required to achieve a desired nett salary, based on marriage status and number of dependencies.
        Enter the desired nett salary as a number, marriage status as a boolean (True for married, False for unmarried),
        and the number of dependencies as an integer. The output will be the calculated monthly salary and tax information.
        """
    args_schema: Type[BaseModel] = NettToMonthlySalaryInput

    def _run(self, nett_salary: float, marriage_status: bool, dependencies: int):
        result = nett_to_monthly_calculator(nett_salary, marriage_status, dependencies)
        return str(result)

    def _arun(self, nett_salary: float, marriage_status: bool, dependencies: int):
        raise NotImplementedError("nett_to_monthly_calculator does not support asynchronous")

########################### Create Agent #####################################

os.environ["OPENAI_API_KEY"] = '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'


llm = ChatOpenAI(
    model='gpt-3.5-turbo-0613',
    temperature=0
)

tools = [
    TaxCalculatorTool(),
    NettToMonthlySalaryTool(),
    # Add more tools as needed
]

agent = OpenAIFunctionsAgent.from_llm_and_tools(llm, tools)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

agent = OpenAIFunctionsAgent.from_llm_and_tools(llm, tools)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

# Example usage
# query = "Calculate the taxes for a monthly salary of 30000000, married, and two dependencies"
# result = agent_executor.run(query)
# print(result)

query2 = "Calculate the monthly salary required to achieve a nett salary of IDR 15 million, unmarried, and 0 dependencies"
result2 = agent_executor.run(query2)
print(result2)