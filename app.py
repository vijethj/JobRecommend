import smtplib
import docx2txt
import nltk
import requests
import pdfminer
from collections import defaultdict
from pdfminer.high_level import extract_text
import numpy as np
import pandas as pd
#from resume import Resume

nltk.download('stopwords')
nltk.download('punkt')

from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, send_file
from flask_mail import Mail, Message
from pymongo import MongoClient
from werkzeug.utils import secure_filename
import os
import glob
import base64
import nbformat
import nbconvert
import io
import importlib.util
from smtplib import *

global l

app = Flask(__name__)
app.secret_key = "super_secret_key"
app.config['UPLOAD_FOLDER'] = 'uploads'

client = MongoClient('mongodb+srv://ka1:vijeth1728@cluster.pwupy.mongodb.net/test')
db = client['pdf_files']
collection = db['pdf_collection']

app.config['MAIL_SERVER']='smtp.rediffmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = 'email'
app.config['MAIL_PASSWORD'] = 'password'
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
mail = Mail(app)

def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)

def extract_text_from_docx(docx_path):
    txt = docx2txt.process(docx_path)
    if txt:
        return txt.replace('\t', ' ')
    return None
 
SKILLS_DB = [
    'machine learning',
    'data science',
    'python',
    'c/c++',
    'c',
    'C++',
    'javascript',
    'docker',
    'kubernetes',
    'kafka',
    'cloud',
    'MERN',
    'OOAD',
    'tensorflow',
    'jenkins',
    'java',
    'react',
    'angular',
    'aws',
    'css',
    'html',
    'flask',
    'nodejs',
    'fastapi'
]
 
def extract_skills(input_text):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    word_tokens = nltk.tokenize.word_tokenize(input_text)
 
    # remove the stop words
    filtered_tokens = [w for w in word_tokens if w not in stop_words]
 
    # remove the punctuation
    filtered_tokens = [w for w in word_tokens if w.isalpha()]
 
    # generate bigrams and trigrams (such as artificial intelligence)
    bigrams_trigrams = list(map(' '.join, nltk.everygrams(filtered_tokens, 2, 3)))
 
    # we create a set to keep the results in.
    found_skills = set()
 
    # we search for each token in our skills database
    for token in filtered_tokens:
        if token.lower() in SKILLS_DB:
            found_skills.add(token)
 
    # we search for each bigram and trigram in our skills database
    for ngram in bigrams_trigrams:
        if ngram.lower() in SKILLS_DB:
            found_skills.add(ngram)
 
    return found_skills

def exec(res_path):        
    text = extract_text_from_pdf(res_path)
    skills = extract_skills(text)
    st = set()
    for i in skills:
        st.add(i.lower())
    print(list(st)) 
    flash(list(st), 'success') 
    return list(st)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'pdf'}

@app.route('/')
def index():
    files = list(collection.find())
    return render_template('index.html', files=files)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        pdf_file = {'filename': filename}
        db.pdf.insert_one(pdf_file)
        with open(file_path, 'rb') as f:
            file_contents = f.read()
        collection.insert_one({'filename': file.filename, 'contents': file_contents})
        flash('File uploaded successfully', 'error')
        return redirect(url_for('index'))
    else:
        flash('Invalid file format', 'error')
        return redirect(url_for('index'))
@app.route('/view/<filename>')
def view(filename):
    file = collection.find_one({'filename': filename})
    if file:
        return render_template('view.html', file=file)
    else:
        flash('File not found', 'error')
        return redirect(url_for('index'))

@app.route('/uploads/<filename>', endpoint ='get_file')
def get_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
           
@app.route('/delete/<string:filename>', methods=['POST'])
def delete(filename):
    if request.method == 'POST':
        # Delete the file from the MongoDB server using the filename
        collection.delete_one({'filename': filename})
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        files = glob.glob('uploads/*')
        for file in files:
            if os.path.basename(file) == filename:
                files.remove(file)
                break
        flash('File deleted successfully', 'error')
    return redirect(url_for('index'))

@app.route('/submit', methods=['POST'])
def submit():
    latest_record = collection.find_one(sort=[('_id', -1)])
    latest_filename = latest_record['filename']
    # print(latest_record)
    # Code to handle form submission goes here
    features = exec("./uploads/"+latest_filename)
    temp = node(features)
    print(l)
    ans = predict(temp , l)
    print(ans)
    flash(ans, 'final')
    smtp_conn = smtplib.SMTP(app.config['MAIL_SERVER'], app.config['MAIL_PORT'])
    smtp_conn.set_debuglevel(True)
    smtp_conn.ehlo()
    smtp_conn.starttls()
    smtp_conn.ehlo()
    smtp_conn.login(app.config['MAIL_USERNAME'], app.config['MAIL_PASSWORD'])
    name = request.form['name']
    email = request.form['email']
    msg = Message('Hello', sender='jyothsnajain@rediffmail.com', recipients=[email])
    msg.body = f"Hello {name}, aap muh me loge kya?"
    smtp_conn.sendmail('jyothsnajain@rediffmail.com', email, msg)
    mail.send(msg)
    return redirect(url_for('index'))

class node:
    attributes = []
    connections = {}
    jobs = []
    def __init__(self , j):
        self.attributes = j
    def setJob(self , s):
        self.jobs = s
    def connect(self , node , value):
        self.connections[node] = value

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def Union(lst1, lst2):
    final_list = lst1 + lst2
    return final_list


def predict(n , l):
    for i in l:
        sim_score = 0
        int_l = len(intersection(n.attributes , i.attributes))
        un_l = len(Union(i.attributes , n.attributes))
        sim_score = int_l/un_l
        n.connect(i , sim_score)
    m = 0
    n_maxval = {}
    for i in n.connections:
        if(n.connections[i] == max(n.connections.values())):
            return i.jobs
    return n.connections

def prepare():
    df = pd.read_csv(r'data_set_fin.csv')
    graph = defaultdict(list)
    l = []
    for i in range(len(df['skills'])):
        graph[df['skills'][i]].append(df['job'][i])
    for i in graph:
        attr = i.split('|')
        n = node(attr)
        n.setJob(graph[i])
        l.append(n)
    s = set(df['job'])
    max_attr = len(s)
    for i in range(len(l)):
        for j in range (i+1 , len(l)):
            sim_score = 0
            int_l = len(intersection(l[i].attributes , l[j].attributes))
            un_l = len(Union(l[i].attributes , l[j].attributes))
            
            #print(int_l , un_l)
            sim_score = int_l/un_l
            #print(sim_score)
            l[i].connect(l[j] , sim_score)
    return l

if __name__ == '__main__':
    l = prepare()
    app.run(debug=True)
