import os
from tqdm import tqdm
from ftplib import FTP_TLS

def ftp_pull():
    
    host = '58.229.37.30'
    port = 5503
    usr = 'hackathon'
    pwd = '2021wswg#$'
    ftps = FTP_TLS()
    
    ftps.connect(host, port)
    
    # ftps.decode = 'euc-kr'
    # Output: '220 Server ready for new user.'
    ftps.login(usr, pwd)
    # Output: '230 User usr logged in.'
    ftps.prot_p()
    # Output: '200 PROT command successful.'
    dir_list = open('./dataset/file name.txt', 'r').read().split('\n')\
        
    sort_list = [f'4.Y자/1.앞']
        
    # outlier_file = open("./dataset/outlier.txt", "w+")
    for sort in sort_list:
        for dir in tqdm(dir_list) : 
            ftps.encoding = 'euc-kr'
            init_path = f'/21nia01/정제데이터/{dir}/Image/{sort}'
            
            try:    
                ftps.cwd(init_path)
            except:
                print(f"{dir} does not have '1.앞.' or 'A.자'")
            ftps.encoding = 'utf-8'
            image_list = ftps.nlst()     # 현재 경로 파일 이름 리스트 반환
            
            ftps.encoding = 'euc-kr'
            destination_path = f'dataset/v2/origin_image/{dir}'
            if not os.path.isdir(destination_path):
                os.mkdir(destination_path)

            for image in image_list: 
                path = os.path.join(init_path, image)
                path = path.replace("\\","/")
                
                dest_image_path = os.path.join(destination_path, image)
                
                file = open(dest_image_path, 'wb')
                ftps.retrbinary('RETR '+ image, file.write)
                file.close()

        ftps.close()



if __name__ == '__main__':
    ftp_pull()

