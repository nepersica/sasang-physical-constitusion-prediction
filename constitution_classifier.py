import os
from tqdm import tqdm
import cv2 as cv
import numpy as np
import pandas as pd

"""
늑골둘레, 장골둘레, 곡골둘레 데이터가 없는 것과 체질분석을 수행하는데 영상만으로 판단
=> 현재 있는 데이터인 높이 데이터와, segment로부터 추출한 너비 데이터를 최대한 활용

"""
def get_segment_height_ratio(df):    
    """
    전체 segment로부터 체질 분석에 필요한 segment 영역을 추출하기 위한 height 백분율 계산
    
    > cervical: 목 뒤 뼈 높이
    > armpit : 겨드랑 높이
    > armpit : 엉덩이 높이 => 본래 사상 체질 분석 때 장골을 사용하는데 우선 대체하여 사용
    """
    cervical = df['목뒤높이']- 4.0         # 대한민국 평균 여성 목 뒤~겨드랑 사이 길이
    armpit = df['겨드랑높이']
    hip = df['엉덩이높이']
    
    cervical_to_hip = cervical - hip
    cervical_to_armpit = cervical - armpit   
    
    armpit_to_hip = cervical_to_hip - cervical_to_armpit
    
    height_ratio = (armpit_to_hip / cervical_to_hip) 
    
    return height_ratio

def _calc_histogram(segment, get_threshold=False):
    height, _ = segment.shape
    
    if get_threshold:
        img_row_sum = np.sum(segment[int(height*0.8):], axis=1).tolist()
    else:
        img_row_sum = np.sum(segment, axis=1).tolist()
        
    img_row_sum = [x / height for x in img_row_sum]
    
    if get_threshold:
        max_index = img_row_sum.index(max(img_row_sum))  
        lower_threshold_height = int(height*0.8) + max_index        
        return lower_threshold_height
    
    return img_row_sum
        
def crop_segment(segment, height_ratio):
    """
        관심 영역 segment crop 수행
    """
    coords = np.argwhere(segment == 255)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    cropped_segment = segment[x_min:x_max+1, y_min:y_max+1]
    height, _ = cropped_segment.shape
    
    # 목~겨드랑이 segment 제거
    upper_threshold_height = height - int(height * height_ratio)
    cropped_segment = cropped_segment[upper_threshold_height:, :]
       
    # segment 밑 부분 제거
    lower_threshold_height = _calc_histogram(cropped_segment, get_threshold=True)
    cropped_segment = cropped_segment[:lower_threshold_height, :]
    
    return cropped_segment

def predict_constitution(segment):    
    """
    추출된 segment로부터 mask width 획득
    """
    
    df_body_inform = pd.DataFrame(columns=['윗가슴 너비', '젖가슴 너비', '허리 너비', '위배 너비', '엉덩이뼈 너비', '태양인 측정치', '태음인 측정치', '소양인 측정치', '소음인 측정치'])
    height, width = segment.shape
    
    intensity_histogram = _calc_histogram(segment)
    # segment = cv.cvtColor(segment, cv.COLOR_GRAY2BGR)
    
    # crop best        
    A12 = 0.17      # 겨드랑이-가슴
    A23 = 0.33      # 가슴 - 허리
    A34 = 0.44      # 허리 - 위배
    
    A1 = 0
    A2 = int(height * A12)
    A3 = int(height * A23) + A2
    A4 = int(height * A34) + A3
    A5 = height-1
    
    # 체간 영역 기준선 빨간색 라인으로 그리기
    # segment = cv.line(segment, (0, A1), (width-1, A1), (0, 0, 255), 3)
    # segment = cv.line(segment, (0, A2), (width-1, A2), (0, 0, 255), 3)
    # segment = cv.line(segment, (0, A3), (width-1, A3), (0, 0, 255), 3)
    # segment = cv.line(segment, (0, A4), (width-1, A4), (0, 0, 255), 3)
    # segment = cv.line(segment, (0, A5), (width-1, A5), (0, 0, 255), 3)   
    
    # cv.imshow("a", segment)
    # cv.waitKey(0)
        
    A1_width = np.mean(intensity_histogram[A1:1])
    A2_width = np.mean(intensity_histogram[(A2-2):(A2+2)])
    A3_width = np.mean(intensity_histogram[(A3-2):(A3+2)])
    A4_width = np.mean(intensity_histogram[(A4-2):(A4+2)])
    A5_width = np.mean(intensity_histogram[A5-2:A5])
    
    A2_14 = (A1_width/A4_width)*100
    A2_41 = (A4_width/A1_width)*100
    A2_25 = (A2_width/A5_width)*100
    A2_52 = (A5_width/A2_width)*100
    
    constitution_list = ['Taeyangin', 'Taeumin', 'Soyangin', 'Soeumin']
    width_list = [A2_14, A2_41, A2_25, A2_52]
    max_idx = np.argmax(width_list)
    
    body_inform = {'윗가슴 너비': A1_width, '젖가슴 너비': A2_width, '허리 너비': A3_width, '위배 너비': A4_width, '엉덩이뼈 너비': A5_width,
                   '태양인 측정치': A2_14, '태음인 측정치': A2_41, '소양인 측정치': A2_25, '소음인 측정치': A2_52, '예측 결과': constitution_list[max_idx]}
    df_body_inform = df_body_inform.append(body_inform, ignore_index=True)
    
    # 2. 체질에 속하지 않는 중간선인 허리(A23)에 대한 비율
    # A1_ratio = A1_width/A3_width
    # A2_ratio = A2_width/A3_width
    # A3_ratio = A3_width/A3_width
    # A4_ratio = A4_width/A3_width
    # A5_ratio = A5_width/A3_width
    
    return df_body_inform, constitution_list[max_idx]
    

def classify_constitution():
    csv_file = pd.read_csv('./dataset/v2/female_csv_all_test.csv', encoding='CP949')
    
    segment_path = './result/predict/female'
    files = os.listdir(segment_path)
    
    df_result = pd.DataFrame(columns=['pt_id', '윗가슴 너비', '젖가슴 너비', '허리 너비', '위배 너비', '엉덩이뼈 너비', '태양인 측정치', '태음인 측정치', '소양인 측정치', '소음인 측정치', '예측 결과'])
    result_constitution = {'Taeyangin': 0, 'Taeumin': 0, 'Soyangin': 0, 'Soeumin':0}
        
    for file in files:
        pt_id = file[6:-7]
        df_predict = pd.DataFrame(columns=['pt_id'])
        df_person_inform = csv_file[csv_file['pt_id'] == pt_id]
        
        # 관심 영역 segment height ratio 계산
        height_ratio = get_segment_height_ratio(df_person_inform)
            
        # load segemnt
        segment = cv.imread(os.path.join(segment_path, file), cv.IMREAD_GRAYSCALE)
        
        # height ratio에 따른 segment 관심 영역 crop
        segment = crop_segment(segment, height_ratio)
                
        # 추출된 segment로부터 mask width 획득
        df_extracted_inform, predicted_constitution = predict_constitution(segment)
        print(f'{pt_id} constitution: {predicted_constitution}')
        
        result_constitution[predicted_constitution] += 1
        
        df_predict = df_predict.append({'pt_id': pt_id}, ignore_index=True)
        df_predict = pd.concat([df_predict, df_extracted_inform], axis=1)
        
        df_result = df_result.append(df_predict)       
    
    df_result.to_csv('./result/체질 분석 결과.csv', index=False, encoding='euc-kr')
    print(result_constitution)
    
        

if __name__ == '__main__':
    classify_constitution()

