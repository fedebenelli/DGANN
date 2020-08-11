import pandas as pd
import tabula

filename = 'DGA.pdf'
tables = tabula.read_pdf(filename,pages='all',stream=False,lattice=True)

df = pd.DaraFrame()

for table in tables:
    match = re.findall(r'Identified as (.*) by',table.columns[0])
    if match != []:
        table.columns = table.iloc[0]
        table['fault'] = match*table.shape[0]
        table = table.drop(0).drop(1)
        df = df.append(table, ignore_index = True)

df = df.replace(np.nan, 0)
df = df.replace('-',0)
df[['H2','CH4','C2H2','C2H4','C2H6','CO','CO2']] = df[['H2','CH4','C2H2','C2H4','C2H6','CO','CO2']].astype('float')

"""
Replacing fault types with coding
Partial Discharges (PD)            = PD  > 0
Discharges of Low Energy ( D1 )    = D1  > 1
Discharges of High Energy ( D2 )   = D2  > 2 
Thermal Faults < 700 C (T1 and T2) = T12 > 3
Thermal Faults > 700 °C (T3)       = T13 > 4
"""

df = df.replace('Partial Discharges (PD)',0)
df = df.replace('Discharges of Low Energy ( D1 )',1)
df = df.replace('Discharges of High Energy ( D2 )',2)
df = df.replace('Thermal Faults < 700 C (T1 and T2)',3)
df = df.replace('Thermal Faults > 700 °C (T3)',4)

df.to_excel('data.xlsx')