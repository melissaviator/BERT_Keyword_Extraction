'''
Melissa Viator
CS688 Term Project
Keyword Extraction of TED Talk Videos with BERT
'''


#Libraries
#Note - Some installation might be necessary
import numpy as np
import wave, math, contextlib
import speech_recognition as sr
from moviepy.editor import AudioFileClip
from keybert import KeyBERT
import en_core_web_sm
from collections import Counter
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt 
import plotly.express as px
from plotly.offline import plot
import plotly.graph_objects as go
from wordcloud import WordCloud



#FUNCTIONS

#Convert audio wav to transcription txt
def transcript(audio_file, video_file, file_name):
    audioclip = AudioFileClip(video_file)
    audioclip.write_audiofile(audio_file)
    with contextlib.closing(wave.open(audio_file,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    total_duration = math.ceil(duration / 60)
    r = sr.Recognizer()
    for i in range(0, total_duration):
        with sr.AudioFile(audio_file) as source:
            audio = r.record(source, offset=i*60, duration=60)
        f = open(file_name, "a")
        f.write(r.recognize_google(audio))
        f.write(" ")
    f.close()


#Read in txt files
def read_txt(file):
    lines = []
    with open(file,  encoding='utf8') as file:
        lines = file.readlines()
    transcription = " ".join(lines)
    return transcription


#Keywords extraction - transcript to key words
def keywords(txt_file):
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(txt_file, keyphrase_ngram_range=(1,2), 
                                         stop_words="english",
                                         use_mmr = True, diversity= 0.6)
    return keywords


#Keyword lists similarity 
def lst_similarity(kw1, kw2):
    kw1_str = " ". join(kw1)
    kw2_str = " ". join(kw2)
    nlp = en_core_web_sm.load()
    return nlp(kw1_str).similarity(nlp(kw2_str))


#Similarity matrix for human vs BERT
def similarity_matrix(human_kw, bert_kw):
    arr1 = np.array(human_kw)
    arr2 = np.array(bert_kw)
    nlp = en_core_web_sm.load()
    matrix = cdist(arr2.reshape(-1, 1), arr1.reshape(-1, 1), lambda x, y: nlp(str(x[0])).similarity(nlp(str(y[0]))))
    matrix = matrix.round(2)
    
    fig = px.imshow(matrix, color_continuous_scale="Viridis",
                    labels=dict(x="Human Keywords", y="BERT Keywords", color="Similarity"),
                    x=arr1.tolist(), y=arr2.tolist(),
                    text_auto=True)
    
    return matrix, fig



#TED TALK 1

#Human keywords
human1_kw = read_txt("data/tedtalk1/my_keywords1.txt").split(", ")
#word cloud
wordcloud = WordCloud(background_color="white").generate_from_frequencies(Counter(human1_kw))
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

#Transcript keywords
#transcript("data/tedtalk1/tedtalk1.wav", "data/tedtalk1/tedtalk1.mp4", "data/tedtalk1/transcription1.txt")
transcription1 = read_txt("data/tedtalk1/transcription1.txt")
transcript1_kw = keywords(transcription1)
transcript1_kw_lst = list(list(zip(*transcript1_kw))[0])

#Summary keywords
summary1 = read_txt("data/tedtalk1/summary1.txt")
summary1_kw = keywords(summary1)
summary1_kw_lst = list(list(zip(*summary1_kw))[0])

#Results
transcript_summary_sim1 = lst_similarity(transcript1_kw_lst, summary1_kw_lst)

human_transcript_sim1 = lst_similarity(human1_kw, transcript1_kw_lst)
human_transcript_matrix1 = similarity_matrix(human1_kw, transcript1_kw_lst)[0]
fig = similarity_matrix(human1_kw, transcript1_kw_lst)[1]
plot(fig, auto_open=True)

human_summary_sim1 = lst_similarity(human1_kw, summary1_kw_lst)
human_transcript_matrix1 = similarity_matrix(human1_kw, summary1_kw_lst)[0]
fig = similarity_matrix(human1_kw, summary1_kw_lst)[1]
plot(fig, auto_open=True)



#TED TALK 2

#Human keywords
human2_kw = read_txt("data/tedtalk2/my_keywords2.txt").split(", ")
#word cloud
wordcloud = WordCloud(background_color="white").generate_from_frequencies(Counter(human2_kw))
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

#Transcript keywords
#transcript("data/tedtalk2/tedtalk2.wav", "data/tedtalk2/tedtalk2.mp4", "data/tedtalk2/transcription2.txt")
transcription2 = read_txt("data/tedtalk2/transcription2.txt")
transcript2_kw = keywords(transcription2)
transcript2_kw_lst = list(list(zip(*transcript2_kw))[0])

#Summary keywords
summary2 = read_txt("data/tedtalk2/summary2.txt")
summary2_kw = keywords(summary2)
summary2_kw_lst = list(list(zip(*summary2_kw))[0])

#Results
transcript_summary_sim2 = lst_similarity(transcript2_kw_lst, summary2_kw_lst)

human_transcript_sim2 = lst_similarity(human2_kw, transcript2_kw_lst)
human_transcript_matrix2 = similarity_matrix(human2_kw, transcript2_kw_lst)[0]
fig = similarity_matrix(human2_kw, transcript2_kw_lst)[1]
plot(fig, auto_open=True)

human_summary_sim2 = lst_similarity(human2_kw, summary2_kw_lst)
human_transcript_matrix2 = similarity_matrix(human2_kw, summary2_kw_lst)[0]
fig = similarity_matrix(human2_kw, summary2_kw_lst)[1]
plot(fig, auto_open=True)



#TED TALK 3

#Human keywords
human3_kw = read_txt("data/tedtalk3/my_keywords3.txt").split(", ")
#word cloud
wordcloud = WordCloud(background_color="white").generate_from_frequencies(Counter(human3_kw))
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

#Transcript keywords
#transcript("data/tedtalk3/tedtalk3.wav", "data/tedtalk3/tedtalk3.mp4", "data/tedtalk3/transcription3.txt")
transcription3 = read_txt("data/tedtalk3/transcription3.txt")
transcript3_kw = keywords(transcription3)
transcript3_kw_lst = list(list(zip(*transcript3_kw))[0])

#Summary keywords
summary3 = read_txt("data/tedtalk3/summary3.txt")
summary3_kw = keywords(summary3)
summary3_kw_lst = list(list(zip(*summary3_kw))[0])

#Results
transcript_summary_sim3 = lst_similarity(transcript3_kw_lst, summary3_kw_lst)

human_transcript_sim3 = lst_similarity(human3_kw, transcript3_kw_lst)
human_transcript_matrix3 = similarity_matrix(human3_kw, transcript3_kw_lst)[0]
fig = similarity_matrix(human3_kw, transcript3_kw_lst)[1]
plot(fig, auto_open=True)

human_summary_sim3 = lst_similarity(human3_kw, summary3_kw_lst)
human_transcript_matrix3 = similarity_matrix(human3_kw, summary3_kw_lst)[0]
fig = similarity_matrix(human3_kw, summary3_kw_lst)[1]
plot(fig, auto_open=True)



#COMPARE RESULTS
tt = ["Ted Talk 1", "Ted Talk 2", "Ted Talk 3"]
transcript_sim = [human_transcript_sim1*100, human_transcript_sim2*100, human_transcript_sim3*100]
summary_sim = [human_summary_sim1*100, human_summary_sim2*100, human_summary_sim3*100]

transcript_txt = ["Transcript %.2f p" % (transcript_sim[0]),
        "Transcript %.2f p" % (transcript_sim)[1],
        "Transcript %.2f p" % (transcript_sim)[2]]
summary_txt = ["Summary %.2f p" % (summary_sim[0]),
        "Summary %.2f p" % (summary_sim)[1],
        "Summary %.2f p" % (summary_sim)[2]]

fig = go.Figure()
fig.add_trace(go.Bar(
    x=tt,
    y=transcript_sim,
    name="Transcript",
    text= transcript_txt,
    marker_color=["#440154", "#31688e", "#35b779"]))
fig.add_trace(go.Bar(
    x=tt,
    y=summary_sim,
    name="Summary",
    text= summary_txt,
    marker_color=["#482878", "#26828e", "#6ece58"]))
fig.update_traces(marker_opacity=0.8,textfont_size=14, textangle=0, textposition="outside", cliponaxis=False)
fig.update_layout(yaxis_title="Cosine Similarity (%)", showlegend=False)
plot(fig, auto_open=True)
