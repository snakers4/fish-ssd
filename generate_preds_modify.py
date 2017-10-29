import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import gc

# Do a linear regression of fish sizes

LABELS =  ['species_fourspot',
           'species_grey sole',
           'species_other', 
           'species_plaice',
           'species_summer',
           'species_windowpane',
           'species_winter']

def change_coords(x1,y1,x2,y2,f_len):
    
    max_x = 1280
    max_y = 720
    
    x_av = (x2+x1)/2
    y_av = (y2+y1)/2
    

    if(x_av-f_len/2)>max_x:
        x1_new = max_x
    elif (x_av-f_len/2)<0:
        x1_new = 0
    else:
        x1_new=x_av-f_len/2 
        
    if(x_av+f_len/2)>max_x:
        x2_new = max_x
    else:
        x2_new=x_av+f_len/2 
        
    if(y_av-f_len/2)>max_y:
        y1_new = max_y
    elif (y_av-f_len/2)<0:
        y1_new = 0
    else:
        y1_new=y_av-f_len/2 

    if(y_av+f_len/2)>max_y:
        y2_new = max_y
    else:
        y2_new=y_av+f_len/2

    return x1_new,y1_new,x2_new,y2_new

df = pd.read_csv('../fish-video/train.csv')

df['_'] = df.apply(lambda row: change_coords(row['x1'],row['y1'],row['x2'],row['y2'],row['length']), axis=1)
df[['x1_new','y1_new','x2_new','y2_new']] = df['_'].apply(pd.Series)
del df['_']

df['x_d'] = df['x2_new'] - df['x1_new']
df['y_d'] = df['y2_new'] - df['y1_new']

X = df[df['x_d']>0][['x_d','y_d']].values
y = df[df['x_d']>0]['length'].values

clr = LinearRegression()
clr.fit(X,y)

del df,X,y

THRESHOLDs = [0.95]
box_list = ['xmin','xmax', 'ymin', 'ymax']

for THRESHOLD in THRESHOLDs:
    

    df_sub = pd.read_csv('fish_ssd_v2.csv')
    # leave only one video
    # df_sub = df_sub[df_sub.video_id.isin(['09WWcMSr5nbKk0lb', '01rFQwp0fqXLHg33'])]
    del df_sub['Unnamed: 0']

    df_sub.loc[df_sub[LABELS].max(axis=1)<THRESHOLD, LABELS+['xmin','ymin','xmax','ymax']] = 0

    # df_sub = df_sub[df_sub.video_id.isin(['01rFQwp0fqXLHg33','09WWcMSr5nbKk0lb'])]

    current_idx = 0
    block_start_idx = -1
    block_end_idx = 0
    is_block = 0
    fish_number = 0
    previous_vid = 'dfsdfsdfsdfsdfsddfs'
    current_vid = 'asfsadfsdafdsafasdfsdfa'

    def update_block(df,
                     idx_start,idx_end,
                     best_class,
                     best_class_prob,
                     xmin,xmax,ymix,ymax,
                     fish_number,
                     ):

        # do some tweaking here
        # df.loc[idx_start:idx_end-1,LABELS+box_list] = 0
        # df.loc[idx_start:idx_end-1,best_class] = best_class_prob * 0.95
        df.loc[idx_start:idx_end-1,box_list] = xmin,xmax,ymix,ymax

        # df.loc[idx_start:idx_end-1,second_best_class] = second_best_class_prob * 0.2
        
        df.loc[idx_start:idx_end-1,'fish_number'] = fish_number
        

    def get_block(df,idx_start,idx_end):
        return df[LABELS+box_list][idx_start:idx_end]
    def process_block(block):
        best_class = block[LABELS].sum(axis=0).sort_values(ascending=False).index[0]
        
        best_class_block = block[block[best_class]>0][box_list].mean(axis=0)
        best_class_prob = block[block[best_class]>0][best_class].max(axis=0)

        
        return best_class,best_class_prob,best_class_block['xmin'],best_class_block['xmax'],best_class_block['ymin'],best_class_block['ymax']

    with tqdm(total = df_sub.shape[0]) as pbar:
        for index, row in df_sub.iterrows():

            # print(index)
            current_vid = row['video_id']

            if(row['xmin']>0): 
                is_block = 1
                # print('is block triggered')
            else:
                is_block = 0

            if ((is_block == 1) & (block_start_idx == -1)):
                block_start_idx = current_idx
                # print('block_start_idx triggered')

            if((block_start_idx > -1) & (is_block==0) & (df_sub.loc[current_idx:current_idx+5,'xmin'].sum()==0) & (current_vid == previous_vid)):
                block_end_idx = current_idx
                # print('block_end_idx triggered')

            if((block_start_idx > -1) & (is_block==1) & (current_vid != previous_vid)):
                block_end_idx = current_idx 

            if (block_end_idx>0):
                block = get_block(df_sub,block_start_idx,block_end_idx)
                best_class,best_class_prob,xmin,xmax,ymin,ymax =  process_block(block)
                fish_number += 1
                del block
                update_block(df_sub,
                             block_start_idx,
                             block_end_idx,
                             best_class,
                             best_class_prob,
                             xmin,xmax,ymin,ymax,
                             int(fish_number)
                             )
                gc.collect()
                if current_vid != previous_vid:
                    fish_number = 0

                if current_vid != previous_vid:
                    block_start_idx = current_idx
                else:
                    block_start_idx = -1

                block_end_idx = 0
                # print('block update triggered')        
            current_idx += 1
            previous_vid = row['video_id']
            # print(current_vid,previous_vid,fish_number)

            pbar.update(1) 

    df_sub['x_d'] = df_sub['xmax'] - df_sub['xmin']
    df_sub['y_d'] = df_sub['ymax'] - df_sub['ymin']
    preds = clr.predict(df_sub[df_sub['x_d']>0][['x_d','y_d']])
    df_sub.loc[df_sub['x_d']>0,'length'] = preds

    df_sub.to_csv('ssd_fish_thres_max_all_class_{}.csv'.format(THRESHOLD))

    del df_sub['xmin'],df_sub['xmax'],df_sub['ymin'],df_sub['ymax'],df_sub['x_d'],df_sub['y_d']
    df_sub = df_sub.set_index('row_id')
    df_sub = df_sub.fillna(value=0)
    df_sub.to_csv('ssd_fish_thres_max_all_class_{}.csv'.format(THRESHOLD))
