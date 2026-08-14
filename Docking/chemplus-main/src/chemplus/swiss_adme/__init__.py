import pandas as pd

def get_swiss_adme_table(swiss_adme_csv, xlsx = True):
    """ Takes: either a link to SwissAdme csv file or a file.
        Returns: either an xlsx (default) or a csv file with the data for reports. 
        
        Data for reports: 'Ligand', 'Chemical formula', 'Molecular weight (Da)', 'LogPo/w', 
                          'Number of hydrogen bond donors', 'Number of hydrogen bond acceptors', 'Synthetic accessibility' """
    try:
        content = pd.read_csv(swiss_adme_csv)
        data = content[['Molecule', 'Canonical SMILES', 'Formula', 'MW', 'Consensus Log P', '#H-bond donors', '#H-bond acceptors', 'Synthetic Accessibility']]
        data_swiss_adme = data.rename(columns = {'Molecule': 'Ligand',
                       'Formula' : 'Chemical formula',
                       'MW' : 'Molecular weight (Da)',
                       'Consensus Log P' : 'LogPo/w',
                       '#H-bond donors' : 'Number of hydrogen bond donors',
                       '#H-bond acceptors' :'Number of hydrogen bond acceptors',
                       'Synthetic Accessibility' : 'Synthetic accessibility'})
        if xlsx:
            data_swiss_adme.to_excel("swiss_adme_table.xlsx", index = False) # !pip install openpyxl
        else:
            data_swiss_adme.to_csv("swiss_adme_tale.csv", index = False)
            
    except Exception as e:
        print(e)
        
#csv_URL = "http://www.swissadme.ch/results/662337373/swissadme.csv"
#get_swiss_adme_table(csv_URL)
