__author__ = 'kevintandean'
import requests, xmltodict
r = requests.get('http://www.chemspider.com/Search.asmx/GetCompoundInfo?CSID=15266&token=9c4931c7-cea7-4e2e-af04-b5fd4f960e12')
s = r.text
print s