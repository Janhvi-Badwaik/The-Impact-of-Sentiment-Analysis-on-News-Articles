import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import mysql.connector
import streamlit as st
from PIL import Image
from pathlib import Path
import cv2
from PIL import Image
from pytesseract import pytesseract
import os
import glob
import shutil
from imutils import paths
import csv
from ultralytics import YOLO
import os
from io import StringIO
from tempfile import NamedTemporaryFile
from transformers import pipeline



#connection
mydb=mysql.connector.connect(host="localhost",user="root",password="anku",database="demo")
mycursor=mydb.cursor()  


# Prediction model
def modelWine():
    #Input of Components
    
    #Frontend
    st.title("Analyzing the sentiments of News Articles :bar_chart: :chart_with_upwards_trend: ")
    st.write(" ")
    st.write("**Note : Upload image of any News Article**")
    st.write(" ")
    st.write(" ")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        #print(uploaded_file.name)
        save_path = os.path.join(os.path.expanduser("~"), "Desktop\\Images", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())
        
    
        
        #objectdetection model

        # Load a pretrained YOLOv8n model
        model = YOLO("C:\\Users\\manis\\Downloads\\ocr\\OCR Pipeline\\best_yolov8s_newsarticle_5k_80mAP_Mar4.pt/")
        fname = uploaded_file.name[:-4]
        # Define path to directory containing images and videos for inference
        source = f"C:\\Users\\manis\\Desktop\\Images\\{fname}.jpg" #"C:\Users\manis\Desktop\Images\0004.jpg"
        # Run inference on the source
        results = model(source,save_txt=True, imgsz=320, conf=0.5)




        #sort files

        path_to_text="C:\\Users\\manis\\Desktop\\research_ppr\\clg_demo\\runs\\detect/"
        path = f"C:\\Users\\manis\\Desktop\\research_ppr\\clg_demo\\runs\\detect\\predict\\labels\\{fname}.txt"
        #print(path)
        df = pd.read_csv(str(path), header=None)
        df = df.sort_values(by=[0], ascending=True)
        name=str(path)
        #print(name[72:-4])

        df.to_csv(f"C:\\Users\\manis\\Desktop\\research_ppr\\clg_demo\\sorted_coordinates\\sorted_{fname}.txt", header=None,index=False)
            
            
            
            
            
            
            
            

        #ocr
            
        imagePaths = source
        files = f"C:\\Users\\manis\\Desktop\\research_ppr\\clg_demo\\sorted_coordinates\\sorted_{fname}.txt"

        path_to_tesseract = r'C://Users//manis//AppData//Local//Programs//Tesseract-OCR//tesseract.exe'
        pytesseract.tesseract_cmd = path_to_tesseract

        fields = ['Name', 'heading', 'sub_heading','crux','content','caption']

        filename = "university_records_demo.csv"
        with open(filename, 'w',newline='',encoding="utf-8") as csvfile:
            writer1 = csv.DictWriter(csvfile, fieldnames=fields)
            writer1.writeheader()
            img1 = cv2.imread(imagePaths)
            pathf1=str(imagePaths)
            name=fname

            img1.shape
            oh = img1.shape[0]
            ow = img1.shape[1]
                
            with open(files) as file:
                text1=""
                text2=""
                text3=""
                text4=""
                text5=""
                l=[]
                thisdict = {"Name": "NA","heading": "NA", "sub_heading": "NA", "crux": "NA", "content": "NA", "caption": "NA" }
                thisdict["Name"]=name
                #print(file)
                    
                for line in file:
                    list =[]
                    for index in line.split():
                        list.append(index)

                    oix = float(list[1])*ow
                    oiy = float(list[2])*oh
                    oiw = (float(list[3])*ow)//2
                    oih = (float(list[4])*oh)//2

                    xmin = int((oix-oiw))
                    ymin = int((oiy-oih))
                    xmax = int((oix+oiw))
                    ymax = int((oiy+oih))

                    region_of_interest = img1[ymin:ymax,xmin:xmax]
                    if list[0]=='0':
                        region_of_interest = cv2.bitwise_not(region_of_interest)
                        kernel = np.ones((2,2),np.uint8)
                        region_of_interest = cv2.erode(region_of_interest,kernel,iterations=1)
                        region_of_interest = cv2.bitwise_not(region_of_interest)
                        text = pytesseract.image_to_string(region_of_interest,lang='eng+hin+mar').replace('\n',' ')#.replace('\n',' ')
                        text1+=text
                        thisdict["heading"] = text1
                        head=text1
                    if list[0]=='1':
                        text = pytesseract.image_to_string(region_of_interest,lang='eng+hin+mar').replace('\n',' ')
                        text2+=text
                        thisdict["sub_heading"] = text2
                    if list[0]=='2':
                        text = pytesseract.image_to_string(region_of_interest,lang='eng+hin+mar').replace('\n',' ')
                        text3+=text
                        thisdict["crux"] = text3
                    if list[0]=='3':
                        text = pytesseract.image_to_string(region_of_interest,lang='eng+hin+mar').replace('\n',' ')
                        text4+=text 
                        thisdict["content"] = text4
                    if list[0]=='5':
                        text = pytesseract.image_to_string(region_of_interest,lang='eng+hin+mar').replace('\n',' ')
                        text5+=text
                        thisdict["caption"] = text5
                l.append(thisdict)
                writer1.writerows(l)
    
    button_c = st.button("Classify")
    if button_c:            
        pipe = pipeline('sentiment-analysis')        
        if head:
            out = pipe(head)
            neut = {
                    0:{
                        "label":"NEUTRAL",
                        "score": out[0]['score']
                        }
                    }
            if out[0]['score']>0.85:
                a=out[0]['score']
                b=out[0]['label']
                #st.json(out[0]['score'])
                st.markdown(f'**{b}**')
                st.markdown(f'**{a}**')
            else:
                a=neut[0]['score']
                b=neut[0]['label']
                #b='Class:'+str(neut[0]['label'])
                #st.json(out[0]['score'])
                st.markdown(f'**{b}**')
                st.markdown(f'**{a}**')        
    retry = st.button("Try Another Image")
    if retry:
        directory_path = 'C:\\Users\\manis\\Desktop\\research_ppr\\clg_demo\\runs'
        # Check if the directory exists before attempting to delete it
        if os.path.exists(directory_path):
            shutil.rmtree(directory_path)
 

    
    
    
    
    

    



def main():
    
    
    menu = ["Home","Login","SignUp","LogOut"]
    choice = st.sidebar.selectbox("Menu",menu)
    
    #Home Page
    if choice == "Home":
        st.title("Welcome to Sentiment Analyzer.")
        st.write(" ")
        st.markdown("Decoding News Emotions: Unveiling Truths, Not Just Words.")
        st.write("  ")
        image1=Image.open("C:\\Users\\manis\\Desktop\\research_ppr\\clg_demo\\news-1172463_640.jpg")
        st.image(image1)
        
    #Login Page
    elif choice == "Login": 
        lusername = st.sidebar.text_input("**Username**")
        lpassword = st.sidebar.text_input("**Password**",type='password')
        if st.sidebar.checkbox("Login"):
            sql="select opassword from account where username = %s and opassword = %s"
            val=(lusername,lpassword)
            mycursor.execute(sql,val)
            data=mycursor.fetchone()
            mydb.commit()
            if  data:
                st.sidebar.success("Logged in {}".format(lusername))
                modelWine()                
            else:
                st.sidebar.warning("Incorrect Username/Password")
     #LogOut Page           
    elif choice == "LogOut":
        st.header("You have Logged Out Successfully!")
        st.header("THANK YOU!")
       
    #SignUp Page    
    elif choice == "SignUp":
        st.subheader("**Create new Account**")
        email = st.text_input("**Email**")
        username = st.text_input("**Username**")
        opassword = st.text_input("**Password**",type='password')
        cpassword = st.text_input("**Confirm Password**",type='password')
        try:
            if st.button("SignUp"):
                if opassword==cpassword:
                    sql = "insert into account(email,username,opassword,cpassword) values(%s,%s,%s,%s)"
                    val=(email,username,opassword,cpassword)
                    mycursor.execute(sql,val)
                    mydb.commit()
                    st.success("You have successfully created a valid Account")
                    st.info("Go to Login Menu to Login")
                else:
                    st.warning("Confirm password is not same")
        except:
            st.warning("Account is already created")
    
    
    
    
    
if __name__ == '__main__':
    main()
    
    
    
    
