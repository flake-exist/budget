"""
нужно чтобы столбцы в файле с бюджетами шли также, как формируется канал: source,medium,campaign,content,term
из них формируется ga_channel и является дополнительным ключом в склейке.
так же, производится группировка га-каналов и кликовых каналов. 
чтобы убрать это действие необходимо закомментировать исполнение метода dop_transformation_shapley_data в calc_metrix
при проставлении channel_number за уникальный канал принимается строка: profile/source/medium/campaign/content/term/creative
аргумент mode земенен на type_report
удален аргумент project_id
"""

import os
import pandas as pd
import numpy as np
from functools import reduce
import glob
import argparse
from pulp import *


class CalcBudget:
    def __init__(self,template,budget_path, atrubution_res_path,outputpath,type_report):
        self.budget_path=budget_path
        self.atrubution_res_path=atrubution_res_path
        self.outputpath=outputpath
        self.template = template
        self.type_report=type_report # "creative"/"profile". default-'profile'        
        
    def get_goal(self,path): # 7_2_2020-10-09_2021-01-25
        return int(path.split('/')[-1].split('_')[1])
    
    def get_goal2(self,path):# 16-2020-10-09-2021-01-25-creative-null-true
        return int(path.split('/')[-1].split('-')[0])
    
    def read_budget_data(self):
        #budget_data = pd.read_excel(self.budget_path)
        budget_data = pd.read_csv(self.budget_path,sep=',', encoding='utf-8')
        budget_data.profile=budget_data.profile.astype(str)
        if self.type_report=='creative':
            budget_data.term=budget_data.term.astype(str)
        budget_data.loc[:,'ga_channel']=budget_data.loc[:, 'source':'term'].astype(str).apply('_>>_'.join,1)
        return budget_data  
    
    def get_full_shapley_result(self):

        path_list = glob.glob(self.atrubution_res_path + self.template)
        df_list=[]
        for path in path_list:
            goal=self.get_goal2(path)
            data=pd.read_csv(path, sep=',')
            data.loc[:,'NN']=goal
            df_list.append(data)

        shapley_result = reduce(lambda up,down: pd.concat([up, down],axis=0, sort=False,ignore_index=False), df_list) 

        shapley_result.reset_index(inplace=True)
        shapley_result.drop(['index','Unnamed: 0'], axis=1, inplace=True)
        return shapley_result
    
    def transformation_shapley_data(self,shapley_result=None):
        ##                     1
        intt,profileID_l,ad_id_,creative_id,ga_chanl=[],[],[],[],[]
        for chan in shapley_result.channel_name:
            if len(chan.split('_>>_'))==9:
                intt.append(chan.split('_>>_')[0]),profileID_l.append(chan.split('_>>_')[-3])
                ad_id_.append(chan.split('_>>_')[-2]),creative_id.append(chan.split('_>>_')[-1])
                ga_chanl.append(('_>>_').join(chan.split('_>>_')[1:6]))
            else:
                intt.append(0),profileID_l.append(0),ad_id_.append(0),creative_id.append(0),ga_chanl.append(chan)

        shapley_result.loc[:,'interaction_type']=intt        
        shapley_result.loc[:,'profile']=profileID_l
        shapley_result.loc[:,'ad_id']=ad_id_
        shapley_result.loc[:,'creative']=creative_id
        shapley_result.loc[:,'ga_channel']=ga_chanl          
        
        return shapley_result    
    
    def dop_transformation_shapley_data(self,shapley_result=None):
        # dop
        shapley_result_test_p1=shapley_result[shapley_result.interaction_type=='view']
        shapley_result_test_p2=shapley_result[shapley_result.interaction_type!='view']
        shapley_result_test_p3=shapley_result[shapley_result.interaction_type=='click']

        df=shapley_result_test_p2[['ga_channel','shapley_value','NN']].groupby(['ga_channel','NN']).sum().reset_index()
        df1=shapley_result_test_p3.merge(df, on=['ga_channel','NN'], how='left')
        df1.rename(columns={'shapley_value_y':'shapley_value'}, inplace=True)
        df1.drop('shapley_value_x', axis=1, inplace=True)

        shapley_result_test2=pd.concat([shapley_result_test_p1,df1])
        
        return shapley_result_test2
    
    def _get_max_use_conv(self, data=None, number_of_goal=[]):
        conv_dict={}
        for goal in number_of_goal:
            df = data[data.NN==goal]
            conv_dict[goal] = df.shapley_value.sum()
        return conv_dict
    
    def calc_cpm(self,data=None):
        ll=[]
        for b,i in zip(data.new_goal_budget,data.impressions):
            if i!=0:
                ll.append((b/i)*1000)
            else:
                ll.append(0)
        return ll
    
    def make_new_format(self, data=None):
        data.rename(columns={'click':'click_api'}, inplace=True)

        val='shapley_value'
        col='interaction_type'    
        drop_list=['channel_name','date_start','date_finish']

        collist=list(data.columns)
        for i in drop_list: collist.remove(i)

        index_list=collist.copy()
        for i in [val,col]: index_list.remove(i)

        res_3=pd.pivot_table(data[collist], values=val, index=index_list,columns=[col], aggfunc=np.sum)
        res_3.reset_index(inplace=True)
        res_3.fillna(0, inplace=True)

        res_3.loc[:,'total_shapley_value']=res_3.loc[:,'click']+res_3.loc[:,'view']


        num_chan_dd={i[1]:i[0] for i in enumerate(set(zip(res_3.profile, res_3.source, res_3.medium,
                                                      res_3.campaign,res_3.content,res_3.term,res_3.creative)))}
        res_3.loc[:,'channel_number']=[num_chan_dd[i] for i in list(zip(res_3.profile, res_3.source, res_3.medium,
                                      res_3.campaign,res_3.content,res_3.term,res_3.creative))]
        return res_3    
    
    def Debil(self,input_dict,total):

        if sum(input_dict.values())==0:
            return input_dict

        else:

            goal_dict_sorted = {k: v for k, v in sorted(input_dict.items(), key=lambda item: item[1])}

            prob=LpProblem("CPA_problem",LpMinimize)

            decision_vars_dict = {"x_{0}".format(i) : LpVariable("{}".format(i),0,None,LpContinuous ) for i in goal_dict_sorted.keys()}

            x_list = [decision_vars_dict[i] for i in decision_vars_dict.keys()]
            c_list = [i for i in goal_dict_sorted.values()]

            prob += sum([l * (1/r) for l,r in zip(x_list,c_list)]) - sum(x_list) * 1/(sum(c_list)) , "Budget_Distribution"

            #constraints
            prob += sum(x_list) == total,"const_budget"
            for i in range(len(x_list) - 1):
                prob += x_list[i]*(1/c_list[i]) >= x_list[i+1]*(1/c_list[i+1]),"constraint{0}".format(i)

            # prob.writeLP("BudgetDistrib.lp")
            if prob.solve() == 1:     
                # print("Status:", LpStatus[prob.status])
                r = {v.name : v.varValue for v in prob.variables()}
                return r

            else:

                r = {v : 0 for v in goal_dict_sorted.keys()}
                return r   

    def run_DEBIL(self,input_dict=None): return {k:self.Debil(v[0],v[1]) for k,v in input_dict.items()}


    def make_dict(self, data=None):
        big_dict={}
        for num in data.channel_number:
            big_dict[num]=({i:j for i,j in zip(data[data.channel_number==num].NN,
                                              data[data.channel_number==num].total_shapley_value) if j!=0},
                           data[data.channel_number==num].budget_fact.values[0] )
        return big_dict
    
    def calc_metrix(self, shapley_result=None,budget_data=None ):
        conv_dict=self._get_max_use_conv(data=shapley_result,number_of_goal=shapley_result.NN.unique())
        shapley_result=self.transformation_shapley_data(shapley_result=shapley_result)
        shapley_result=self.dop_transformation_shapley_data(shapley_result=shapley_result)

        if self.type_report=='creative':
            result_full_money=budget_data.merge(shapley_result, on=['ga_channel','profile','creative'], how='left')
        else:
            result_full_money=budget_data.merge(shapley_result, on=['ga_channel','profile'], how='left') 
        
        if result_full_money[result_full_money.NN.isna()].shape[0]==result_full_money.shape[0]:
            raise Exception("no data matches")

        result_full_money=self.make_new_format(data=result_full_money)
        big_dict=self.make_dict(data=result_full_money)
        dict_from_debil=self.run_DEBIL(input_dict=big_dict) 
        
        for k,v in dict_from_debil.items():
            for kk,vv in v.items():
                indx=result_full_money[(result_full_money.channel_number==k) & (result_full_money.NN==int(float(kk)))].index[0]
                result_full_money.loc[indx,'new_goal_budget']=vv        
        result_full_money.fillna(0, inplace=True)
        
        result_full_money.loc[:,'CPM'] = self.calc_cpm(data=result_full_money)
        result_full_money['total_sum']=result_full_money.NN.apply(lambda row: conv_dict[row])   
        result_full_money['new_CPA']=result_full_money.new_goal_budget/result_full_money.total_shapley_value 
        
        result_full_money['old_CPA']=result_full_money.budget_fact/result_full_money.total_shapley_value
        result_full_money.new_CPA=np.round(result_full_money.new_CPA.values,1)
        result_full_money.old_CPA=np.round(result_full_money.old_CPA.values,1)
        
        return result_full_money
        
    def check_result(self, data):
        for num in data.channel_number.unique():
            budget_fact=data[data.channel_number==num].budget_fact.unique()[0]
            goal_budget=list(data[data.channel_number==num].new_goal_budget)
            if round(budget_fact,1)==round(sum(goal_budget),1) or data[data.channel_number==num].NN.unique()[0]==0:
                continue
            else:
                print('channel num:',num,'|','FALSE','|','budget_fact:',budget_fact,'|','sum_goal_budget:',sum(goal_budget))
        return 
      

    def safe_file(self, data_to_safe): data_to_safe.to_csv(self.outputpath, index=False)
        
        
def run_budget(template= '', budget_path='', atrubution_res_path='',outputpath='',
               type_report='profile'
              ):
    
    cb=CalcBudget(template,budget_path, atrubution_res_path,outputpath,type_report)
    shapley_result =cb.get_full_shapley_result()
    
    budget_data=cb.read_budget_data()
    result_full_money= cb.calc_metrix(shapley_result=shapley_result,budget_data=budget_data )
    cb.safe_file(result_full_money)
    cb.check_result(result_full_money)
     
    return result_full_money        

if __name__ == '__main__':
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--template', action='store', type=str, required=True)
    my_parser.add_argument('--budget_path', action='store', type=str, required=True)
    my_parser.add_argument('--atrubution_res_path', action='store', type=str, required=True)
    my_parser.add_argument('--outputpath', action='store', type=str, required=True)
#     my_parser.add_argument('--project_id', action='store', type=str, required=True)
    my_parser.add_argument('--creative', action='store', type=str, required=True)

    args = my_parser.parse_args()
    
    # **args
    res=run_budget(template=args.template, budget_path=args.budget_path, 
               atrubution_res_path=args.atrubution_res_path,outputpath=args.outputpath,
               mode=args.mode )
