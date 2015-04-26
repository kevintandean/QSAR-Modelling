__author__ = 'kevintandean'
import requests, xmltodict

def get_smiles(id):
    r = requests.get('http://www.chemspider.com/Search.asmx/GetCompoundInfo?CSID='+str(id)+'&token='+TOKEN)
    s = r.text
    doc = xmltodict.parse(s)
    smiles = doc['CompoundInfo']['SMILES']
    return smiles

def get_csid(name):
    url = 'http://www.chemspider.com/Search.asmx/SimpleSearch?query='+name+'&token='+TOKEN
    r = requests.get(url)
    doc = xmltodict.parse(r.text)
    print r.text
    try:
        csid = doc['ArrayOfInt']['int'][0]
        return csid
    except KeyError:
        pass

    return None



TOKEN = '9c4931c7-cea7-4e2e-af04-b5fd4f960e12'
id = get_csid('john')
smiles = get_smiles(id)
