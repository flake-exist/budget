import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm

class ModularityClass:
    def __init__(self, sep=' / ',channel_list=[], index_list=[0,1,2]):
        self.sep=sep
        self.channel_list=channel_list
        self.index_list=index_list
        self.social_attribute=['vkontakte', 'vk_','_vk','_vk_', 'facebook','fb_','_fb','_fb_', 'instagram',
                               '_ig','ig_','_ig_' 'social']
        self.olv_attribute=['olv', 'cpv', 'youtube', 'roll','video']
        self.display_attribute=['dco','banner','display','cpm','banner','carousel','multiformat']
        self.promo_attrubute=['email', 'e-mail', 'sms']  
        
    def calc_modularity(self):
        modularity_l=[]
        for sm in tqdm(self.channel_list):
            source=sm.split(self.sep)[self.index_list[0]]
            medium=sm.split(self.sep)[self.index_list[1]]
            campaign=sm.split(self.sep)[self.index_list[2]]
            if sm.startswith('(direct) / (none)'):
                modularity_l.append('direct')
                #cpc:
            elif (source.find('yandex')!=-1 or source.find('google')!=-1) and medium.find('cpc')!=-1:
                if campaign.find('brand')==-1:
                    modularity_l.append('Paid Search (general)')
                elif  campaign.find('brand')!=-1:
                    modularity_l.append('Paid Search (brand)')
            elif sum(list(map(lambda x: sm.find(x)!=-1, self.social_attribute)))!=0:
                if medium=='referral':
                    modularity_l.append('Organic Social')
                else:
                    modularity_l.append('Paid Social')
            elif sm.find('view')==1 or sm.find('click')==1 \
            or sum(list(map(lambda x: sm.find(x)!=-1, self.display_attribute)))!=0:
                modularity_l.append('Display')
            elif sum(list(map(lambda x: sm.find(x)!=-1, self.olv_attribute)))!=0:
                modularity_l.append('OLV')
            elif medium=='organic':
                modularity_l.append('Organic Search')
            elif medium =='referral':
                modularity_l.append('Referral')
            elif medium.startswith('ag'):
                modularity_l.append('Display')
            elif source=='adsniper':
                modularity_l.append('Display')
            elif sum(list(map(lambda x: sm.find(x)!=-1, self.olv_attribute)))!=0:
                modularity_l.append('OLV')
            elif sum(list(map(lambda x: medium.find(x)!=-1, self.promo_attrubute)))!=0:
                modularity_l.append('Promo')
            else:
                if medium.find('cpc')!=-1:
                    modularity_l.append('Display')
                else:
                    modularity_l.append('other')
        return modularity_l
    

def run( sep=' / ',
        channel_list=[], 
        index_list=[0,1,2]): 
    """
    input: 
    sep - separator inside the channel
    channel_list - list of channels for which you need to define ModularityClass
    index_list - list of indices, for subsequent division into source-index_list[0]/medium-index_list[1]/campaign-index_list[2]
    
    return:
    list of modularityclass
    
    """
    mc=ModularityClass(sep=sep,channel_list=channel_list, index_list=index_list)
    return mc.calc_modularity()   


"""
run:
from ModularityClass import run
res=run( sep=' / ',channel_list=list(df['column_channel_name']), index_list=[0,1,2])
df['Modularity_Class']=res
"""