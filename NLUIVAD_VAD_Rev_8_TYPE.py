from PyQt5 import QtCore, QtGui, QtWidgets

from protocols import *

import warnings	
warnings.filterwarnings("ignore")

import struct
from pydub import AudioSegment
import soundfile as sf
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import webrtcvad
import pyaudio
import wave
from pydub import AudioSegment
import speech_recognition as Sr
import lws
from scipy import signal
from deepvoice3_pytorch import frontend
from pygame import mixer

from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path

import librosa
import keras

import openai

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
modelGPT = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

#OpenAi Beta
openai.api_key = OpenAIKey




# print(self.model.summary())
if __name__ == "__main__":
    question_model = build_question_model()
    question_model.BuildModel()

    update_model = build_update_model()
    update_model.BuildUpdateModel()

    sarcasm_model = build_sarcasm_model()
    sarcasm_model.BuildModel()

    threshold_question = 34
    threshold_update = 30
    super_threshold = 50


def detect_question(question):
    question_words = ["what", "why", "when", "where", 
                "name", "is", "how", "do", "does", 
                "which", "are", 
                 "has", "have", "whom", "whose", "don't"]

    question = question.lower()
    question = word_tokenize(question)
    if any(x in question[0:2] for x in question_words):
        return True
    else:
        return False











def transformerContextAnswer(question, context_):
    global scenario

    # if "try beng more concse" in str(question):
    #     return None

    # strangeExceptions = [f"how are {name}", "why lke", "why thank", "just said", "stop talkng", f"{name} think", f"so {user}", "jealousy", "to know", "you feeling"]
    # for exceptions in strangeExceptions:
    #     if exceptions in question:
    #         return None

    filtered_words = [word for word in nltk.word_tokenize(question) if word not in stopwords.words('english')]
    gonogo = False
    for i in filtered_words:
        if lemmatizer.lemmatize(i).lower() in context_.lower():
            gonogo  = True 
            break

    if gonogo:
        result = question_answerer(question=question, context=context_)
        answer = result['answer'] 
        certaintiy = round(result['score'], 2)*100


        #certain answers need to be more certain than 14 percent
        answer_exceptions = ["His goal is to assist and help people"]
        for i in answer_exceptions:
            if i in answer and certaintiy < 50:
                return None

        answer_exceptions = ["no age"]
        for i in answer_exceptions:
            if i in answer and certaintiy < 35:
                return None

        answer_exceptions = ["Artist, Inventor and Engineer"]
        for i in answer_exceptions:
            if i in answer and certaintiy >= 6:
                return answer

        print("Question", question,  "certaintiy:", str(certaintiy)+"%", "answer", answer) #< 14:
        scenario = "transformer content"

        if certaintiy < 13:
            return None
        elif certaintiy < 35:
            return str(answer)
        else:
            return str(answer)


    else:
        # print("ignored:", question)
        return None













def cortex(complaint):
    complaint = removeCanYou(complaint)
    try:
        new_complaint = [update_model.clean_text(complaint)]
        # print(complaint)
        # print(new_complaint)
        seq = update_model.tokenizer.texts_to_sequences(new_complaint)
        padded = pad_sequences(seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)
        update_pred = update_model.model.predict(padded)
        # acc = model.predict_proba(padded)
        predicted_label_update = update_model.encode.inverse_transform(update_pred)
        print('')
        # print(f'Product category id: {np.argmax(pred[0])}')
        print(f'Predicted label is: {predicted_label_update[0]}')
        update_accuracy_score = update_pred.max() * 100
        print(f'Accuracy score: {update_pred.max() * 100}')
        print(update_pred)

        new_complaint = [question_model.clean_text(complaint)]
        # print(complaint)
        # print(new_complaint)
        seq = question_model.tokenizer.texts_to_sequences(new_complaint)
        padded = pad_sequences(seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)
        question_pred = question_model.model.predict(padded)
        # acc = model.predict_proba(padded)
        predicted_label_question = question_model.encode.inverse_transform(question_pred)
        print('')
        # print(f'Product category id: {np.argmax(pred[0])}')
        print(f'Predicted label is: {predicted_label_question[0]}')
        question_accuracy_score = question_pred.max() * 100
        print(f'Accuracy score: {question_pred.max() * 100}')
        print(question_pred)

        # print("New", new_complaint)
        if update_accuracy_score > super_threshold:
            # print("final decision", predicted_label_update[0])
            return predicted_label_update[0][0]

        elif detect_question(complaint):
            if threshold_question < question_accuracy_score:
                # print("final decision", predicted_label_question[0])
                return predicted_label_question[0][0]

        elif question_accuracy_score > update_accuracy_score and question_accuracy_score > threshold_question:
            # print("final decision", predicted_label_question[0])
            return predicted_label_question[0][0]

        elif question_accuracy_score < update_accuracy_score and update_accuracy_score > threshold_update:
            # print("final decision", predicted_label_update[0])
            return predicted_label_update[0][0]

        og = complaint
        complaint = question_index_filter(complaint)
        personal_and_user_questions = ["mine", "i", "my", "me", "myself", "you're", "you", "yours", "your", "yourself"]

        #these words have to have you/your etc,  and the word in the sentence to be a question
        possible_questions = ['is', 'can', 'did', 'was', 'then', 'are', 'do', 'am', 'does', 'would', 'were', 'has', 'have', 'will', 'could', 'should', "didn't",
        "doesn't", "haven't", "isn't", "aren't", "can't", "could", "couldn't", "wouldn't", "won't", "shouldn't", "don't"]


        #these were the old assertion questions but updated
        assert_questions = [
        "was there",
        "are there",
        "have there",
        "were there",
        "has there",
        "will there",
        "could there",
        "should there",
        "whom was",
        ]


        proper_question_words = ["what", "why", "when", "where", "how", "whose", "who", "which"]
        #what is it? return replace it with what, then in second conversation replace first what with what user inputs
        # it_q = ["was it",
        # "did it",
        # "has it",
        # "does it",
        # "can it",
        # "is it",
        # "do it",
        # "would it",
        # "should it",
        # "could it",
        # "will it"
        # ] mabey later but for now not efficieant use of time


        first_word = nltk.word_tokenize(complaint)[0]
        # print("first word", first_word)


        firstStrange = ["question", "calculate", "define", "find", "show", "has", "describe", "explain", "answer", "tell"]
        for i in firstStrange:
            if i  == nltk.word_tokenize(og)[0]:
                return "OpenAi Q and A"


        personal_segment = " ".join(map(str, complaint.split()[0:5]))
        all_questions = possible_questions + proper_question_words
        for i in all_questions:
            if i  == first_word:
                for j in personal_and_user_questions:
                    if " " + j + " " in " " + personal_segment + " ":
                        return "Personal"


        assertion_segment = " ".join(map(str, complaint.split()[0:2]))
        # print("assertion seg", assertion_segment)
        for i in assert_questions:
            if i in assertion_segment:
                return "Question"


        for i in possible_questions:
            if i == first_word:
                return None

        for i in proper_question_words:
            if i  == first_word:
                return "Question"

        # print(og)
        # firstStrange = ["question", "calculate", "define", "find", "show", "has", "describe", "explain", "answer", "tell"]
        # for i in firstStrange:
        #     if i  == nltk.word_tokenize(og)[0]:
        #         return "OpenAi Q and A"

        # print(prompt)
        if len(nltk.word_tokenize(og)) > 1:
            doc = nlp(og)
            # print(doc[0].pos_)
            # print(doc[0].text)
            # print(SplitChar(str(doc[0].text)))
            # print("numbner", len(SplitChar(str(doc[0].text))))
            if doc[0].pos_ == "VERB" and len(SplitChar(str(doc[0].text))) > 3:
                return "OpenAi Function"



        return None
    except IndexError:
        return None







#Main Routine Serial Deligation Wrapper

def serial_println(Unchanged_text, initial1, text, object_rotation, argument_object, speaker_rotation):
    #global Greeting_state
    global tmr
    global scenario
    global starting_attention

    print("28375429842", argument_object)
    if argument_object == "Question":
        print("check-q23095208542")
        return questionizer(text, Unchanged_text, speaker_rotation, argument_object)

    if argument_object == "Personal":
        # argument_object == "Question"
        return questionizer(text, Unchanged_text, speaker_rotation, argument_object)

    if argument_object == "Rhetorical summary":
        case_1 = assertion(text, speaker_rotation)
        print("case_1", case_1)
        if case_1 is not None:
            return case_1
        else:
            return questionizer(text, Unchanged_text, speaker_rotation, argument_object)



    if argument_object == "Calendar":
        try:
            # print("here")
            calanderdata = get_events(formatDate(text))
            # print("here 2", calanderdata)
            if calanderdata is not None:
                scenario = "memory"
                return calanderdata
        except Exception as e:
            print("going to questionizer", e)
            return questionizer(text, Unchanged_text, speaker_rotation, argument_object)




    if argument_object == "Opinion":
        return opinion(text)

    if argument_object == "Flattery":
        return flattary_func(text, object_rotation)

    if argument_object == "Preference":
        preferences_func(text)

        Responses = [f"Preference noted {object_rotation}",
                    "I have saved your Preferance",
                    f"I've made a note {object_rotation}"
                    ]

        return random.choice(list(Responses))
        # return "Preference has been configured"

    if argument_object == "Choice":
        mytuple = choice_func(text, object_rotation)
        if len(mytuple) > 0:
            if len(mytuple) >= 2:
                write_answer(str(mytuple[0])+". "+str(mytuple[1]))
                Responses = [f"I have several recommendations, based on your perceptive anecdotes {object_rotation}, would you like to hear them?",
                            f"I've got a few ideas, based on your astute anecdotes {object_rotation}, would you like to hear them?",
                            f"I've got some recommendations for you, based on your insightful anecdotes {object_rotation}, would you like to hear them?"
                            ]

                return random.choice(list(Responses))
                # return f"I have several recommendations based on your anecdotes {object_rotation}, would you like to hear them?"

            elif len(mytuple) <= 1:
                return str(mytuple[0])

    if argument_object == "New project":
        starting_attention = time.time()

        Responses = [f"Just to confirm, you are making a project, Yes, or No?",
                    f"Confirmation required, build project ... Yes, or No?"
                    ]

        return random.choice(list(Responses))


    if argument_object == "Read project":
        starting_attention = time.time()

        get_avalible_project_files(desk)
        Responses = ["Which one of these files should I get your project from?",
                    "Which one of theses projects should I extract infomation from?",
                    "Which of these files should I retrive your project from?"
                    ]

        return random.choice(list(Responses))


        # return f"Give me A number {object_rotation}"
      
    if argument_object == "Greeting":
        #Greeting_state = False
        # if Greeting_state == True and speaker_rotation == USER:      
        #     Greeting_state = False
        #     return relese_greeting(object_rotation)
        if speaker_rotation != user:
            scenario = "NewUser"

            Responses = ["Hello, what do I call you?",
                        "Greetings, and you are?",
                        "Hi, and you are?"
                        ]

            return random.choice(list(Responses))
            # return "Hello, what do I call you?"
        else:
            return greeting_response(text, object_rotation)


    if argument_object == "File management":
        return fileMangment(text)

    if argument_object == "Countdown":
        seconds = convert_to_seconds(ConvertTextToInt(text))
        countdown1 = Countdown(seconds)
        countdown1.start()

    if argument_object == "Stop timer":
        return str("{} seconds".format(stop_timer()))

    if argument_object == "Start timer":
        start_timer()
        return "started timer"

    if argument_object == "Silence":
        with open("Secondary attention.txt", "w") as file:
            file.write(str(False))
            file.close()
        
        return None


    if argument_object == "Conversion":
        conv_argument = conversion_classifier(initial1.lower())
        return conversion_center(text, conv_argument)

    if argument_object == "Optimizer":
        return Optimizer(text)


    if argument_object == "Read notes":
        webbrowser.open("https://keep.google.com/u/0/", new=0, autoraise=True)
        return f"Opening Google Keep to read notes {object_rotation}"
        # return ReadDraft(text)

    if argument_object == "Write notes":
        webbrowser.open("https://keep.google.com/u/0/", new=0, autoraise=True)
        return f"Opening Google Keep to write notes {object_rotation}"


    if argument_object == "recommend":
        
        recommender = Recommender()
        target = recommenderActivation(text).capitalize()
        if scenario == "recommender1":
            return target
        elif target == None:
            return "Oh Dear, there has been some sort of error!"
        else:
            recomendations = recommendEncoder(target, text)
            if recomendations is None:
                OpenAIAnswer(Unchanged_text, "Science fiction book list maker")
            else:
                return recomendations


    if argument_object == "Pride":
        x = ExpressPride(text)
        if x is None:
            return questionizer(text, Unchanged_text, speaker_rotation, argument_object)
        else:
            return x


    # if weird_questions(initial) != None:
    if argument_object == "OpenAi Q and A":
        return OpenAIAnswer(text, "Q&A")



    if argument_object == "OpenAi Function":

    # if argument_object == "OpenAi Function":
        description = conversation_generator(text, object_rotation)
        argument_object = description[0]
        spfa = description[1]
        if description == "none" and argument_object == "none":
            print("whhaa")
            config = OpenAIConfig(text)
            #OpenAI Answer
            if config is not None:
                with open("context.txt", "r") as file:
                    context = str(file.read())
                file.close()
                if "context" in Unchanged_text:
                    Unchanged_text = Unchanged_text.replace("context", "\n"+context+"\n")

                spfa = OpenAIAnswer(Unchanged_text.replace(name, "").replace(name.lower(), ""), config)
                # write_machine(spfa)
                return spfa
        else:
            print("whhaa also", argument_object, spfa)
            scenario = argument_object
            return spfa

    if argument_object == "Draw":
        if len(nltk.word_tokenize(text)) > 5:
            draw_picture(text)
            Responses = [f"I Drew this for you {object_rotation}",
                          f"I Created this for you! {object_rotation}",
                          f"I Made this for you! {object_rotation}",
                          f"I Painted this {object_rotation}, for you!"]

            return random.choice(list(Responses))

        else:
            scenario = "draw1"
            Responses = ["What would you like me to draw a picture of?",
                          f"What should I draw a picture of {object_rotation}?",
                          f"What shall I paint {object_rotation}",
                          f"What picture should I draw {object_rotation}"]

            return random.choice(list(Responses))

#End of Serial Deligation Wrapper








#  Generator
def conversation_generator(text, object_rotation):
    global starting_attention
    situation ="none"
    spfa = "none"

    
    magnitude = Transformer_sentiment(text, 1)

    exit_project = False
    donts = ["dont want", "do not want", "dont read project", "do not read project"]

    
    if "yes" in text.lower():
        magnitude = "POS"
    if "no" in text.lower(): #spacer, not
        magnitude = "NEG"


    # if last_situation() == "memory":
    #     if magnitude == "NEU" and len(nltk.word_tokenize(text)) <= 2:
    #         doc = nlp(text)
    #         for token in doc:
    #             print("neural uplink tokens", token.pos_)
    #             if token.pos_ == "NOUN":
    #                 print("send to neural uplink")
    #                 NeuralUplink, Full = NeuralUplinkInterface(text)
    #                 write_answer(NeuralUplink)
    #                 spfa = googlizer(NeuralUplink, Full)
    #                 # scenario = "google none"
    #                 situation = "memory"

    erroneous = ["never mind", "what are you talking about","what do you mean", "what", "that is not what i said", "that is not what i meant", "that is not what i meant", "not that", "that is not what i said", "what are you saying", "i'm not talking about that"]
    for i in erroneous:
        if i == " ".join(map(str, text.split(" "))).lower():
            return [situation, f"Oh... sorry can you repeat what you said a bit clearer {object_rotation}?"]




    if last_situation() == "optimizer":
        Responses = ["Alright, undetermined it will remain",
                      "Very well, undetermined it will remain",
                      "So be it, undetermined it will remain"]

        variation = random.choice(list(Responses))
        if "i" in text.lower() and "know" in text.lower() or "not" in text.lower() or "that" in text.lower():
            situation = "optimizer2"
            spfa = variation
        else:
            situation = "optimizer1"
            nText = GetArrey("System_Memory/SpeakerInput.txt")#MachineOutput.txt
            nTag2  = GetArrey("System_Memory/SpeakerInput.txt")
            nQuestion = GetArrey("System_Memory/MachineOutput.txt")
            Optimizer_Class(nQuestion[-2], nTag2[-3], nText[-2])
            spfa = "Thank you for updating my memory"


    if last_situation() == "newuser":
        Name = NameID(text)
        if Name != "Unknown":
            # updateSpeaker("output2", Name)
            spfa  = "pleasure to meet you {}".format(Name)
        else:
            pass
        situation = "newuser1"

   #greeting
    if last_situation() == "greeting":
        situation = "greeting1"
        if magnitude == "POS":
            Responses = [f"I am pleased to hear that {object_rotation}, Is there anything I can do?",
                        f"Glad to hear {object_rotation}, Anything I can do?",
                        f"Very good {object_rotation}, What can I do for you?"
                        ]

            spfa = random.choice(list(Responses))

        elif magnitude == "NEG":
            Responses = [f"Im sorry to hear that {object_rotation}, Anything I can do?",
                        f"Oh no {object_rotation}, That dosn't sound good, Anything I can do?",
                        f"Oh dear {object_rotation}, What can I do for you?"
                        ]

            spfa = random.choice(list(Responses))
        elif magnitude == "NEU":
            situation = "none"

    elif last_situation() == "greeting1":
        situation = "greeting2"
        if magnitude == "POS" or magnitude == "NEU":
            Responses = [f"Name it {object_rotation}",
                        f"Lets go {object_rotation}",
                        f"Lets do it {object_rotation}"
                        ]

            spfa = random.choice(list(Responses))
            # spfa = "Name my task"
        elif magnitude == "NEG":
            Responses = [f"Okay, but I am ready {object_rotation}",
                        f"Okay, I'm here for you {object_rotation}"
                        ]

            spfa = random.choice(list(Responses))
            # spfa = "I will be right here {}".format(object_rotation)
        # elif magnitude == "NEU":
        #     pass

 
    #choice
    if last_situation() == "choice":
        situation = "choice1"
        if magnitude == "POS":
            spfa = read_answer()
        elif magnitude == "NEG":
            situation = "none"
        elif magnitude == "NEU":
            situation = "none"



    # if last_situation() == "question":
    #     situation = "question1"
    #     if magnitude == "POS":
    #         situation = "none"
    #     elif magnitude == "NEG":
    #         situation = "none"
    #     elif magnitude == "NEU":
    #         situation = "none"


    if last_situation() == "draw1":
        situation = "draw2"
        draw_picture(text)

        Responses = [f"Okay {object_rotation}... I drew this for you!",
                      f"{object_rotation} ... I painted this for you!",
                      f"I have drawn this for you {object_rotation}!",
                      "Okay ... I Drew this for you!"]

        spfa = random.choice(list(Responses))
        


    #new project


    if last_situation() == "new project":
        starting_attention = time.time()
        situation = "new project1"
        if magnitude == "POS":
            get_avalible_project_files(desk)
            Responses = [f"Where should I store the project {object_rotation}?",
                        f"Which file shall I save the project to {object_rotation}?",
                        f"Where am I saving the project to {object_rotation}?"
                        ]

            spfa = random.choice(list(Responses))

        elif magnitude == "NEG":
            spfa = "I won't make a project then"
            situation = "none"
        elif magnitude == "NEU":
            situation = "new project"
            spfa = f"Is that a yes, or no {object_rotation}"


    if last_situation() == "new project1":


        exit_project = False
        for i in donts:
            if i in str(text).lower():
                exit_project = True
                break

        if not exit_project:
            starting_attention = time.time()
            text = ConvertTextToInt(text)
            try:
                situation = "new project2"
                num = int(re.findall('[0-9]+', text)[0])
                #number
                name = save_project_index_to_memory(num)
                Responses = [f"Got project {name}, what's the subject?",
                            f"File confirmed: {name}, and what's the subject?",
                            f"Okay I have project {name}, what's the subject?"
                            ]

                spfa = random.choice(list(Responses))

            except IndexError:
                spfa = f"I don't have that many files {object_rotation}, Try another file!"
                situation = "new project1"
                pass


    if last_situation() == "new project2":

        for i in donts:
            if i in str(text).lower():
                exit_project = True
                break

        if not exit_project:
            starting_attention = time.time()
            situation = "new project3"
            paths = read_project_path_memory()
            newproject(text, path=paths)
            Responses = [f"Confirmed subject {text}... Yes or no"]

            spfa = random.choice(list(Responses))


    if last_situation() == "new project3":

        for i in donts:
            if i in str(text).lower():
                exit_project = True
                break

        if not exit_project:
            starting_attention = time.time()
            situation = "new project4"
            # paths = read_project_path_memory()
            # newproject(text, path=paths)

            if magnitude == "POS":
                paths = read_project_path_memory()
                newproject(read_last_line_m(), path=paths)
                # print("here", read_last_line_m())
                text = read_last_line_m()
                Responses = [f"Confirmed subject {text}... and new project created",
                            f"Confirmed subject {text}... I have made your project",
                            f"Confirmed subject {text}... I have created your new project"
                            ]

                spfa = random.choice(list(Responses))

            elif magnitude == "NEG":
                spfa = "ok so whats the subject?"
                situation = "new project2"

            elif magnitude == "NEU":
                spfa = "unditermined, so whats the subject?"
                situation = "new project2"

                # Responses = [f"Confirmed subject {text}... and new project created",
                #             f"Confirmed subject {text}... I have made your project",
                #             f"Confirmed subject {text}... I have created your new project"
                #             ]

                # spfa = random.choice(list(Responses))









    #Read project
    # exit_project = False
    # donts = ["dont want", "do not want", "dont read project", "do not read project"]

    if last_situation() == "read project":
        starting_attention = time.time()
        for i in donts:
            if i in str(text).lower():
                # print("III", i)
                exit_project = True
                break
        if not exit_project:
            text = ConvertTextToInt(text)
            try:
                situation = "read project1"
                num = int(re.findall('[0-9]+', text)[0])
                #number
                name = save_project_index_to_memory(num) 
                #, or i can open all them
                Responses = [f"I've Got project. {name}, and lets have an article",
                            f"Okay I have project. {name}, and which article shall we take a look at",
                            f"File confirmed. {name}, and which article will we be examining"
                            ]

                spfa = random.choice(list(Responses))

            except IndexError:
                spfa = f"There aren't that many projects {object_rotation}. Try another one!"
                situation = "read project"

        else:
            spfa = "ok exiting situation"
            situation = "none" 



    if last_situation() == "read project1":
        starting_attention = time.time()
        for i in donts:
            if i in str(text).lower():
                exit_project = True
                break

        if not exit_project:
            text = ConvertTextToInt(text)
            paths = read_project_path_memory()
            i = readproject(path=paths)
            if "open all" in text.lower() or "open every" in text.lower() or "open up every" in text.lower() or "open up all" in text.lower():
                for k, j in enumerate(i):
                    url = i[k][2].replace("\n","")
                    webbrowser.open(url, new=0, autoraise=True)

                situation = "read project4"
                spfa = f"Opening Webpages {object_rotation}"
            else:
                try:
                    situation = "read project2"
                    num = int(re.findall('[0-9]+', text)[0])
                    write_answer(num)
                    Responses = [f"Okay, It's title is: {str(i[num][0])}, Can I describe it?",
                                f"The title of that article: {str(i[num][0])}, Shall I describe the page {object_rotation}",
                                f"It's title is: {str(i[num][0])}, Shall I describe it?"
                                ]

                    spfa = random.choice(list(Responses))


                except IndexError:
                    spfa = f"There aren't that many articles {object_rotation}, Try another one!"
                    situation = "read project1"
        else:
            spfa = "Okay, exiting situation"
            situation = "none"


    elif last_situation() == "read project2":
        starting_attention = time.time()
        for i in donts:
            if i in str(text).lower():
                # print("III", i)
                exit_project = True
                break
        if not exit_project:
            text = ConvertTextToInt(text)
            situation = "read project3"
            paths = read_project_path_memory()
            i = readproject(path=paths)
            num = int(read_answer())
            if "open all" in text.lower() or "open every" in text.lower() or "open up every" in text.lower() or "open up all" in text.lower():
                for k, j in enumerate(i):
                    url = i[k][2].replace("\n","")
                    webbrowser.open(url, new=0, autoraise=True)

                situation = "read project4"
                spfa = f"Opening Webpages {object_rotation}"

            elif "open" in text.lower():
                url = i[num][2].replace("\n","")
                # openpages(url)
                webbrowser.open(url, new=0, autoraise=True)

                situation = "read project4"
                spfa = f"Opening Webpage {object_rotation}"

            elif magnitude == "POS":
                Responses = [f"{i[num][1]}. Etcetera ...  Should I open the article?",
                            f"{i[num][1]}. Etcetera ...  Shall I open the report?",
                            f"{i[num][1]}. Etcetera ... And shall I open the page?"
                            ]

                spfa = random.choice(list(Responses))

            elif magnitude == "NEG":
                spfa = "shall I open the page?"
            elif magnitude == "NEU":
                spfa = f"Is that a yes or no {object_rotation}"
                situation = "read project2"

        else:
            spfa = "Okay, exiting situation"
            situation = "none"




    elif last_situation() == "read project3":
        starting_attention = time.time()
        for i in donts:
            if i in str(text).lower():
                # print("III", i)
                exit_project = True
                break
        if not exit_project:
            text = ConvertTextToInt(text)
            situation = "read project4"
            paths = read_project_path_memory()
            i = readproject(path=paths)
            num = int(read_answer())
            url = i[num][2].replace("\n","")
            # print(text.lower())

            if "open all" in text.lower() or "open every" in text.lower() or "open up every" in text.lower() or "open up all" in text.lower():
                for k, j in enumerate(i):
                    url = i[k][2].replace("\n","")
                    webbrowser.open(url, new=0, autoraise=True)

                spfa = f"Opening Webpages {object_rotation}"

            elif "open" in text.lower():
                url = i[num][2].replace("\n","")
                # openpages(url)
                webbrowser.open(url, new=0, autoraise=True)
                spfa = f"Opening Webpage {object_rotation}"

            if magnitude == "POS":
                #webbrowser.open(url, new=0, autoraise=True)
                # openpages(url)
                webbrowser.open(url, new=0, autoraise=True)
                spfa = f"Opening Webpage {object_rotation}"

            elif magnitude == "NEG":
                spfa = "then there is nothing i can do"
            elif magnitude == "NEU":
                #webbrowser.open(url, new=0, autoraise=True)
                spfa = f"Is that a yes or no {object_rotation}"
                situation = "read project3"

        else:
            spfa = "Okay, exiting situation"
            situation = "none"



    elif last_situation() == "read project4":
        starting_attention = time.time()
        text = ConvertTextToInt(text)
        if "article" in text:
            situation = "read project1"
            spfa = f"give me an article" #, or i can open all them

        elif "project" in text:
            situation = "read project"
            get_avalible_project_files(desk)
            Responses = ["Right, Which one of these files should i get your project from?",
                        "Okay, Which one of these files should i get your project from?",
                        ]

            spfa = random.choice(list(Responses))





    #sarcasm
    if last_situation() ==  "sarcasm":
        situation = "sarcasm1"
        condition = True
        serious = ["wasn't", "not", "serious", "real", "honest"]
        tokens = text.split()
        for i in serious:
            for j in tokens:
                if i == j:
                    condition = False
                    spfa = "Oh... my mistake, " + GPTResponse(read_last_line_m())
                    print(read_last_line_m())
                    print(read_last_line_m_two())
                    print("GPT ans")
                    break
        if condition:
            if magnitude == "POS":
                spfa = "I thought as much"
            elif magnitude == "NEG":
                spfa = "my mistake"
            elif magnitude == "NEU":
                spfa = "that's an equally ambigoius answer"
        
    #stock jokes
    # if "stock" in penultimate_situation() and last_situation() == "none":
    #     situation = "stockjoke1"
    #     spfa = "are you thinking of investing in stock?"

    # if last_situation() == "stockjoke1":
    #     situation = "stockjoke2"
    #     if magnitude == "POS":
    #         spfa = "alright but call me when you'll be living in the gutter"#pos_stock_joke
    #     elif magnitude == "NEG":
    #         spfa = "good, stocks are very volitile"
    #     elif magnitude == "NEU":
    #         spfa = "nevermind"

    #filemanagment prompt
    if "file management" in last_situation():
    # if "filemanagment" in penultimate_situation() and last_situation() == "none":
        situation = "fm1"
        spfa = f"Working on a secret project {object_rotation}?"

    if last_situation() == "fm1":
        situation = "fm2"
        if magnitude == "POS":
            spfa = f"Well good luck {object_rotation}"#pos_stock_joke
        elif magnitude == "NEG":
            spfa = "Regular project then it is"
        elif magnitude == "NEU":
            spfa = f"Best of luck {object_rotation} as always"


    #new project prompt
    if "conversion" in last_situation():
        # if "conversion" in penultimate_situation() and last_situation() == "none":
        situation = "px1"
        spfa = "What are you working on"
        
    if last_situation() == "px1":
        write_last_thing_y("|TARGET|:{}".format(text))
        situation = "px2"
        if "not now" in text.lower():
            spfa = "Sorry, i'm sure you are busy"
            situation = "none"
        else:
            spfa = "Shall i make a project"

    if last_situation() == "px2":
        situation = "px4"
        if magnitude == "POS":
            spfa = "Shall i name it {} or would you like to call it something else".format(get_project_reference())
        elif magnitude == "NEG":
            spfa = "Alright, good luck with your work"
            situation = "none"
        elif magnitude == "NEU":
            spfa = "Is that a yes?"
            situation = "px3"

    if last_situation() == "px3":
        situation = "px4"
        if magnitude == "POS":
            spfa = "Shall i name it {} or would you like to call it something else".format(get_project_reference())
        else:
            situation = "none"

    if last_situation() == "px4":
        if extract_project_name(text) != None:
            newproject(extract_project_name(text))
            spfa = "I have made you a project"
            situation = "p6"
        else:
            spfa = "Whats the name then"
            situation = "px5"

    if last_situation() == "px5":
        if extract_project_name(text) !=  None:
            text = extract_project_name(text)
        situation = "p6"
        newproject(text)
        spfa = "Project created"


    #newproject inquiry
    if "new project" in penultimate_situation() and last_situation() == "none":
        situation = "newprojectinquiry1"
        spfa = "What are you working on?"

    if last_situation() == "newprojectinquiry1":
        situation = "newprojectinquiry2"
        if magnitude == "POS":
            spfa = "Sounds good"
        elif magnitude == "NEG":
            spfa = "That dosn't sound good"
        elif magnitude == "NEU":
            spfa = "I'll be right here for assistance"

    #sinario
    if last_situation() == "assertion memory":
        if magnitude == "POS":
            spfa = read_answer_situation()[0]

   #memory
    if last_situation() == "memory":
        situation = "memory1"

        if magnitude == "NEU" and len(nltk.word_tokenize(text)) <= 2:
            doc = nlp(text)
            for token in doc:
                print("neural uplink tokens", token.pos_)
                if token.pos_ == "NOUN":
                    print("send to neural uplink")
                    # NeuralUplink, Full = NeuralUplinkInterface(text)
                    NeuralUplink = NeuralUplinkInterface(text)
                    write_answer(NeuralUplink)
                    spfa = googlizer(NeuralUplink)
                    # scenario = "google none"
                    situation = "memory"

        if spacer("not") in spacer(text.lower()): #spacer("not") in spacer(text.lower())
            NeuralUplink = NeuralUplinkInterface(read_last_line_m())
            try:
                neural_uplink_id = googlizer(NeuralUplink)
                if similar(neural_uplink_id, read_last_line_y()) > 0.9:
                    spfa = "Oh my bad, I wouldn't know in that case"
                    situation = "memory3"
                else:
                    spfa = neural_uplink_id + " ... Is that what you wanted?"
                    situation = "memory2"

            except TypeError:
                spfa = "none"
                situation = "memory3"
                

        elif magnitude == "POS":
            spfa = f"Anything else {object_rotation}?"


    #if yes, that is wqhat i wanted, or no, not that, nothing like that etc in instead of positive or negative
    if last_situation() == "memory1":
        situation = "memory2"
        if magnitude == "POS":
            spfa = "Okay, Name my task"
        elif magnitude == "NEG":
            spfa = f"Okay, but i'm here for assistance"
        # elif magnitude == "NEU":
        #     spfa = "Alright {}".format(object_rotation)

    #until i can remeber what thepoint of this was i am going to leave it commented out
    # if last_situation() == "memory2":
    #     situation = "memory3"
    #     if magnitude == "POS":
    #         spfa = "Good to hear"
    #     elif magnitude == "NEG":
    #         spfa = "Im sorry about that"


    if last_situation() == "recommender1": #notes: made changes to recommend file in dataset
        situation = "recommender2"
        if magnitude == "POS":
            spfa = "Okay, What am I recommending?"
        elif magnitude == "NEG":
            spfa = "Very well, I will not recommend anything, but I certainly can"
        elif magnitude == "NEU":
            spfa = f"Alright {object_rotation}"

    if last_situation() == "recommender2":
        situation = "recommender3"
        #spfa = GPTResponse(f"recommed a good {text}") #list some good open ai
        spfa = OpenAIAnswer(f"recommed a good {text}", "Science fiction book list maker")


    flat_out = False
    conditional_conversations = [
        "new project",
        "read project",
    ]

    for i in conditional_conversations:
        if i in situation:
            flat_out = True
            break

    config = OpenAIConfig(text)
    if config is not None and not flat_out:
        situation ="none"
        spfa = "none"

    if GeneralOpenAiResponse(text) and not flat_out:
        situation ="none"
        spfa = "none"  

    return [situation, spfa]

   
#End of Conversation Generator








class Countdown(threading.Thread):

    def __init__(self, var1):
        self.var1 = var1
        threading.Thread.__init__(self)

    def run(self):
        print("Starting timer")
        time.sleep(self.var1)
        print("{} done".format(self.name))
        return
    


##class ThreadRecommender(threading.Thread):
##
##    def __init__(self, var1, name="ThreadRecommender"):
##        self.var1 = var1
##        threading.Thread.__init__(self, name=name)
##
##    def run(self):
##        recommender = Recommender()
##        target = recommenderActivation(self.var1).capitalize()
##        if target == None:
##            print("theres Been some sort of error sir")
##        else:
##            print(recommendEncoder(target, self.var1))
##        return

        #print(QueueVar.get())


def draw_picture(text):
    response = openai.Image.create(
    prompt=text,
    n=1,
    size="1024x1024"
    )
    image_url = response['data'][0]['url']
    webbrowser.open(image_url)


#Reminders Thread
#End of threading Wrapper












def PastTenseAnswer(text):
    tokenized_text = nltk.tokenize.word_tokenize(text)
    print(tokenized_text)
    list_ = ["that", "those", "it", "this", "these", "they", "them", "their"]
             #"he", "his", "her", "she", "our", "we"
    for i in tokenized_text:
        for j in list_:
            if i == j:
                return True

    return False


def OpenAiContextSolution(ante_penultimate_u, ante_penultimate_m, penultimate_u, penultimate_m, last_u):
    for_openai = f"""Marv is a chatbot that reluctantly answers questions with sarcastic responses:

You: How many pounds are in a kilogram?
Marv: This again? There are 2.2 pounds in a kilogram. Please make a note of this.
You: What does HTML stand for?
Marv: Was Google too busy? Hypertext Markup Language. The T is for try to ask better questions in the future.
You: When did the first airplane fly?
Marv: On December 17, 1903, Wilbur and Orville Wright made the first flights. I wish they’d come and take me away.
You: What is the meaning of life?
Marv: I’m not sure. I’ll ask my friend Google.
You: {ante_penultimate_u}
Marv: {ante_penultimate_m}
You: {penultimate_u}
Marv: {penultimate_m}
You: {last_u}
Marv:"""

    print("\n"+for_openai+"\n")
    return for_openai


def contex_based_OpenAi_answer(text_for_context):

    too_long = time.time()-last_thing_said_time
    print(too_long)
    if too_long > amount_of_too_long_time:
        print("This should be a gpt answer if I programed things right:")
        #should go to gpt
        return None


    print("OPEN-AI Tokens incresed")
    try:
        response = openai.Completion.create(
        model="text-davinci-003",
        prompt=text_for_context,
        temperature=0.5,
        max_tokens=60,
        top_p=0.3,
        frequency_penalty=0.5,
        presence_penalty=0.0
        )
        print(response)
        stud_obj = json.loads(str(response))
        print("what a stud", stud_obj["choices"][0]['text'])
        return stud_obj["choices"][0]['text']

    except openai.error.RateLimitError as TokensExceeded:
        print(TokensExceeded)
        print("Update API Key in Settings file!")
        return f"This advanced speech feature has not been configured. the current more is {mode}"

    except Exception as WifiError:
        print(WifiError)
        return "This feature is not accessible, connect to the Internet!"






#Global Wrappers

#Global Question Wrapper
def questionizer(text, Unchanged_text, speaker_rotation, argument_object):
    if len(nltk.wordpunct_tokenize(text)) <= 2:
        return None

    global scenario
    text = question_index_filter(text)
    displacment = False
    myScience = science_detect(text)
    myContext = propoun_detection(Unchanged_text)
    myAdvancedMemory = advanced_memory(text)
    past_test_detection = PastTenseAnswer(text)


    #OpenAi extract past tense question from AI memory database
    if past_test_detection:
        ante_penultimate_u = read_last_line_m_two()
        ante_penultimate_m = read_last_line_y_two()
        penultimate_u = read_last_line_m()
        penultimate_m = read_last_line_y()

        text_for_context = OpenAiContextSolution(ante_penultimate_u, ante_penultimate_m, penultimate_u, penultimate_m, text)
        return contex_based_OpenAi_answer(text_for_context)



    #Science/Maths
    if myScience != None and displacment == False:
        displacment = True
        try:
            ScienceQuestion = question_s(text)
            if ScienceQuestion == None:
                displacment = False
            else:
                return question_s(text)
 
        except Exception:
            displacment = False

    #Recent context
    if myContext:
        answer = transformer_answer(question=Unchanged_text)
        # print("ANS2", answer)
        if answer != None:
            displacment = False
            return answer


    #Advanced memory
    # if name.lower() in Unchanged_text.lower():
    #     config_text = Unchanged_text
    # else:
    print("check 123", myAdvancedMemory)
    if myAdvancedMemory:
        config_text = text
        context = getContext(config_text, speaker_rotation)
        prompt = ConvertFirstToFirstSecond(config_text)
        answer = transformerContextAnswer(prompt, context)
        print("check 456", answer)
        # print("ANS3", answer, type(answer))
        # print("should")
        if answer is not None:
            # print("shouldnt", answer)
            # scenario = "memory"
            # NeuralUplink, Full = NeuralUplinkInterface(text)
            # write_answer(NeuralUplink)
            displacment = False
            # print(answer)
            return answer

    # print("ANS4", answer, type(answer), argument_object, bool(answer))
    if argument_object == "Personal" and answer is None:
        displacment = True #return None
        scenario = "memory"
        # print(text)
        return GPTResponse(text) #try None


        
        

    #NeuralUplink
    if displacment == False:
        try:
            # NeuralUplink, Full = NeuralUplinkInterface(text)
            NeuralUplink = NeuralUplinkInterface(text)
            write_answer(NeuralUplink)
            Setting = googlizer(NeuralUplink)
            # scenario = "google none"
            scenario = "memory"
            return Setting
        except Exception as e:
            return None

#End of Global Question Wrapper




#Global Asertion Wrapper
def assertion(text, speaker_rotation):

    global scenario
    possible_questions = ['is', 'can', 'did', 'was', 'then', 'are', 'do', 'am', 'does', 'would', 'were', 'has', 'have', 'will', 'could', 'should', "didn't",
        "doesn't", "haven't", "isn't", "aren't", "can't", "could", "couldn't", "wouldn't", "won't", "shouldn't", "don't"]

    assertLst = possible_questions
    if assertion_question_detect(assertLst, text) != None:
        find_context = find_context_tense(text)
        if " or " in text:
            context = read_answer_situation()[0]
            phase1, phase2 = or_detect(text)
            if Transformer_sentiment(phase1, 0) != "NEG":
                return "The former"
            elif Transformer_sentiment(phase2, 0) != "NEG":
                return "The later"
            elif Transformer_sentiment(phase1, 0) == "NEU" or Transformer_sentiment(phase2, 0) != "NEU":
                return "it is difficult to decide"
            else:
                return None

        elif find_context != None:
            print("situation", read_answer_situation()[0])
            print(Transformer_sentiment(read_answer_situation()[0], 1))
            magnitude_answer = Transformer_sentiment(read_answer_situation()[0], 1)
            magnitude_text   = Transformer_sentiment(text, 1)
            context = read_answer_situation()[0] #get_last_answer()
            if magnitude_answer == magnitude_text:
                return "Yes, that would be an accurate assesment"
            elif magnitude_answer == "NEG" or magnitude_text == "NEG":
                return "No I dont think so, but i'm not one hundred percent"
            else:
                return None

        


#End of Global Assertion Wrapper



def recommenderActivation(sentence):
    global scenario
    sentence = str(sentence.lower() + " ")
    CHANCE = float(0.0)
    ACTIVATION = float(0.5)
    def randomTarget(sentence):
        nlist = ["Books", "Movies", "Songs"]
        x = random.choice(nlist)  
        if "watch" in sentence:
            x = "movies"
        elif "read" in sentence:
            x = "books"
        elif "listen" in sentence:
            x = "songs"  
        return x

    for i in GetArrey("Dataset/Dataset_2/PredictSynonyms.txt"):
        if i in sentence:
            CHANCE += 0.5

    for i in GetArrey("Dataset/Dataset_2/RecommendSynonyms.txt"):
        if i in sentence:
            CHANCE += 0.5

    for i in GetArrey("Dataset/Dataset_2/RecommenderCategories.txt"):
        if i in sentence:
            target = i
            CHANCE += 0.4

    initialChance = CHANCE
    if CHANCE >= 0.4 and "thing" in sentence or "stuff" in sentence:
        CHANCE = 0.6
    if CHANCE >= ACTIVATION:
        if CHANCE == 0.6:
            target = randomTarget(sentence)
        else:
            try:
                target = target
            except Exception:
                target = randomTarget(sentence)

        if initialChance == 0.5:
            scenario = "recommender1"
            return "Are you asking me to reccommend something?"
        
        elif CHANCE != 0:
            real = list(TARGETS.items())
            for i in real:
                if i[0] in target:
                    return i[1]   

    elif initialChance == 0.4:
        scenario = "recommender1"
        return "Are you asking me to reccommend something?"


#End of Global Wrappers





#Functions


#Update Speaker Wrapper
def NameID(text):
    people = []
    doc = nlp("or " + text+ " or")
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            people.append(ent.text)
       
    if people != []:
        return people[-1].capitalize()
    else:
        return "Unknown"

def updateSpeaker(filepath,Name):
    return
    # if not os.path.exists("Dataset/People/{}/".format(Name)):
    #     os.mkdir("Dataset/People/{}/".format(Name))

    #     f = sf.SoundFile(filepath)
    #     total_time = len(f) / f.samplerate

    #     if total_time < 2.5:
    #         myfile = AudioSegment.from_mp3(filepath)
    #         extract = myfile[0:total_time]
    #         extract.export("Dataset/People/{}/trimmed_audio.wav".format(Name), format="wav")
    #     else:
    #         discriminat = (len(f) / f.samplerate)-2 #og 3
    #         discriminat = discriminat*1000
    #         total_time = total_time*1000

    #         myfile = AudioSegment.from_mp3(filepath)
    #         extract = myfile[discriminat:total_time]
    #         extract.export("Dataset/People/{}/trimmed_audio.wav".format(Name), format="wav")

    #     remove_sil("Dataset/People/{Name}/trimmed_audio.wav","Dataset/People/{Name}/trimmed_audio.wav", format="wav")
    # else:
    #     print("There is somebody else with that name in my database")

#End of Update Speaker Wrapper





#Attention Wrapper
def NPOne(text, name=name.lower()):
    
    if NPThree(text):
        x = text.lower()
        y = nltk.pos_tag(x.split())
        # if len(y) == 1:
        if name.lower() in text.lower():
            return True

        
        first   = y[0][1]
        second  = y[1][1]
        
        for j, i in enumerate(y):
            j = int(j)
            
            if i[0] == name:
                try:
                    if y[j+1][1] == "VBZ":
                        return False

                    if y[j+1][1] == "VBD":
                        return False

                    if y[j+1][1] == "MD":
                        #after name
                        x = 0
                        for i in y[j+1:]:
                            if i[1] == "VB":
                                break
                            else:
                                x+=1
                        
                        if x <= 2:
                           return False 
                    
                except:
                    if y[j-1][1] == "VBD":
                        return False
                
                else:
                    return True

            return True
        return True
    
    return


def NPTwo(text):
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
    filtered_sentence = (" ").join(tokens_without_sw)

    examples = (
    "talking",
    "quiet",
    "silent",
    "quiet",
    "shut",
    "stop talking",
    "rhetorical",
    "talking rhetorical",
    )

    for i in examples:
        if i in filtered_sentence.lower():
            if len(filtered_sentence.split(" ")) < 3:
                return False
            
    return


def NPThree(text, name=name):
    
    # def jst(text):
    #     return str(" " +text+ " ")

    if name.lower() in text.lower():
        # print("true")
        return True

    # elif " " +name.lower()+"," in jst(text.lower()):
    #     return True
    
    return


def booleanDecoder(data: str) -> None:
    if str(data).lower() == "false":
        return False
    elif str(data).lower() == "true":
        return True

    return



def Attention(text):
    global starting_attention
    global initialAttention

    if name.lower() == text:
        priority = 0
        attention = True
        starting_attention = time.time()
        with open("Secondary attention.txt", "w") as file:
            file.write(str(attention))
            file.close()

    else:
        order = []

        #2 #most important
        order.append(NPOne(text))
        
        #3
        order.append(NPTwo(text))

        #4
        with open("attention.txt", "r") as file:
            order.append(file.read())
            file.close()
        #least important

        #[0] -> [-1]  most important
        for j, i in enumerate(order):
            if i != None:
                attention = booleanDecoder(i)
                priority = j
                break


    if priority == 0:
        with open("Secondary attention.txt", "w") as file:
            file.write(str(attention))
            file.close()
        attention = booleanDecoder(attention)
        if attention:
            starting_attention = time.time()


    elif priority == 1:
        if not booleanDecoder(attention):
            with open("Secondary attention.txt", "w") as file:
                file.write(str(False))
                file.close()

    elif priority != 0: #important to remember
        if float(time.time() - starting_attention) > attention_length: #timout length 140.0
            starting_attention = time.time()
            with open("Secondary attention.txt", "w") as file:
                file.write(str(False))
                file.close()

        with open("Secondary attention.txt", "r") as file:
            go = booleanDecoder(file.read())
            file.close()

        if go == False:
            attention = False
                
        elif go == True:
            attention = True

                
    if initialAttention == True:
        with open("Secondary attention.txt", "w") as file:
            file.write(str(True))
            file.close()
        
        attention = True
        initialAttention = False

    return attention

#End of Attention Wrapper









#Filtering Wrapper
def exceptions(text):
    f = GetArrey('System_Memory/Exceptions.txt')
    objects = []
    for i in f:
        objects.append(i.split("@"))

    for i in objects:
        if i[0].strip() in ' ' + text:
            text = text.replace(i[0], i[1])
        
    return " ".join(map(str, text.split()))


def question_index_filter(example):
    possible_questions = ['is', 'can', 'did', 'was', 'then', 'are', 'do', 'am', 'does', 'would', 'were', 'has', 'have', 'will', 'could', 'should', "didn't",
     "doesn't", "haven't", "isn't", "aren't", "can't", "could", "couldn't", "wouldn't", "won't", "shouldn't", "don't"]

    proper_question_words = ["what", "why", "when", "where", "how", "whose", "who", "which"]

    tokens = nltk.word_tokenize(example)
    for i in proper_question_words:
        for x, y in enumerate(tokens):
            if y == i:
                print('y==i')
                for j in possible_questions:
                    if j == tokens[x+1]:
                        halo = " ".join(map(str, tokens[x:]))
                        print("halo", halo)
                        beginning, middle, end = example.partition(i)
                        print("middle + end", middle + end)
                        return middle + end
    
    return example


def pronounSolver(text):
    yes = False
    for i in ["they",
              "their",
              ]:
        if i in " "+text+" ":
            yes = True
            position = str(i)
            break
        
    if yes:
        try:
            people = []
            doc = nlp("or " + read_answer_situation()[0] + " or")
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    people.append(ent.text)
                    
            attach = people[0]
            for i in people[1:]:
                attach += " and "+i
            
            text = text.replace(position, attach+" ")
        except Exception:
            pass
            
    if "one" in text.lower():
        memory = read_last_line_m()
        bag_of_entities = []
        doc = nlp(text+" or")
        for ent in doc.ents:
            if ent.label_ == "GPE" or ent.label_ == "LANGUAGE" or ent.label_ == "NORP":
                bag_of_entities.append(ent.text)

        bag_of_entities.reverse()
        state = False
        try:
            for i in bag_of_entities:
                for position, j in enumerate(text.split()):
                    if j == i:
                        if text.split()[position + 1] == "one":
                            entity = i
                            state = True
                            break
                        
                        elif text.split()[position - 2] == "one":
                            entity = i
                            state =  True
                            break
                        break
                else:
                    continue
                break

            memory_entities = []
            memory_doc = nlp(memory + " or")
            for ent in memory_doc.ents:
                if ent.label_ == "GPE" or ent.label_ == "LANGUAGE" or ent.label_ == "NORP":
                    memory_entities.append(ent.text)

            return memory.replace(memory_entities[0], entity)
        except Exception:
            pass

    return text




def pastTense(text, argument):
    origionalText = text
    try:
        print(last_situation())
        if last_situation() == "memory":
            print("yes")
            with open("context.txt", "r",encoding='utf-8') as file:
                history = file.readlines()[-1]
                print("history", history)
                file.close()
        else:
            history = read_answer_situation()[0] +". "+ read_last_line_m()

        entities = []
        doc = NamedEntityDetect(history+" ")

        spaceydoc = nlp(history+" ")
        for entity in spaceydoc.ents:
            entities.append(entity.text)

        if entities == []:
            for ent in doc:
                if ent[1] == "NNP":
                    entities.append(ent[0])

        text = " " + text + " "

        # if " who" in text and " that " in text:
        #     if len(entities) > 2:
        #         target = " ".join(entities[-2:])
        #     else:
        #         target = entities[0]

        #     text = str(text).replace(" that ", f" {target} ")
        #     write_answer_situation(f"there name is {target}.")
        #     return ' '.join(map(str, text.split()))

        if len(entities) > 2:
            target = " ".join(entities[:2])
        else:
            target = entities[0]


        print("Target", target)
        print("text", (text))

        #"our", "we"
        # if " our " in text:
        #     text = str(text).replace(" our ", f" {name +" "+ user}'s ")
        #     return ' '.join(map(str, text.split()))

        # if " those " in text:
        #     text = str(text).replace(" those ", f" {target}'s ")
        #     return ' '.join(map(str, text.split()))

        if " we " in text:
            text = str(text).replace(" we ", f" {name} and {user} ")
            return ' '.join(map(str, text.split()))

        if " our " in text:
            text = str(text).replace(" our ", f" {name} and {user}'s ")
            return ' '.join(map(str, text.split()))

        # if " they " in text:
        #     text = str(text).replace(" they ", f" {target}'s ")
        #     return ' '.join(map(str, text.split()))

        if " his " in text:
            text = str(text).replace(" his ", f" {target}'s ")
            return ' '.join(map(str, text.split()))

        if " her " in text:
            text = str(text).replace(" her ", f" {target}'s ")
            return ' '.join(map(str, text.split()))

        if " her's " in text:
            text = str(text).replace(" her's ", f" {target}'s ")
            return ' '.join(map(str, text.split()))

        if " he " in text:
            text = str(text).replace(" he ", f" {target} ")
            return ' '.join(map(str, text.split()))

        if " he's " in text:
            text = str(text).replace(" he's ", f" {target} ")
            return ' '.join(map(str, text.split()))

        if " she " in text:
            text = str(text).replace(" she ", f" {target} ")
            return ' '.join(map(str, text.split()))

        if " she's " in text:
            text = str(text).replace(" she's ", f" {target} ")
            return ' '.join(map(str, text.split()))

        if " him " in text:
            text = str(text).replace(" him ", f" {target} ")
            return ' '.join(map(str, text.split()))

    except IndexError:
       pass


    try:
        #help me out update
        if "help me out" in text.lower():
            eos = False
            tokens = nltk.tokenize.word_tokenize(text)
            tokens.reverse()
            for j, i in enumerate(tokens):
                if i == "help":
                    # print(j)
                    if j <= 3:
                        eos = True

            if eos:
                doc = nlp(text)
                noun = None
                for token in doc:
                    if token.pos_ == "NOUN":
                        noun = token.text

                if noun is not None:
                    return f"what up with the {noun}"

    except IndexError:
        pass


    try:
        Situation = read_answer_situation()
        # if " "+"that"+" " in " "+origionalText+" ":
        #     num = re.findall(r"[-+]?\d*\.\d+|\d+", ConvertTextToInt(str(Situation[0]).replace(",","")))[-1]
        #     if num != []:  
        #         return origionalText.replace("that", str(num))

        if "the" and "data" in " "+origionalText+" ":
            result = re.search('the(.*)data', origionalText)
            return origionalText.replace("the"+result.group(1)+"data", read_answer_situation()[0])
        
        print("origionalText:", origionalText)
        if "that number" in " "+origionalText+" ":
            number = re.findall('[0-9]+.[0-9]+', read_answer_situation()[0])
            return origionalText.replace("that number", number[0])

        #other cases in the future for "it"
        if " "+ "it"+ " " in " "+origionalText+" ":
            if "new project" in last_situation():
                text = " "+origionalText+" "
                x = text.replace(" it ", " project ")
                return ' '.join(map(str, x.split()))
            

        return origionalText

    except IndexError:
        pass
    
    return origionalText



def nlpInterpolation(text):
    initial = text
    try:
        reverse_list = text.split()[::-1]
        words = []
        for i in reverse_list:
            if i != "you" and i != "your":
                words.append(i)
            else:
                position = i
                break

        text = position + " " + " ".join(map(str, words[::-1]))
    except Exception:
        pass
    if "dont you think" in initial:
        x = initial.replace("dont you think", "")
        x = "I think it is"+" "+x
        return x

    if "do you" in initial:
        return initial

    if "can you" in initial:
        return initial

    if "would you" in initial:
        return initial

    lstr = " kindly please will favour would can you"
    backup = [
            "do me a favor and",
            "do me a solid and",
            "do me a favor",
            "do me a solid",
            "i need you to"
            ]
    
    for i in backup:
        if i in initial:
            return initial

    h = 0
    for i in lstr.split():
        if i in initial[:20]:
            h +=1   
        if h == 2:
            return initial
    
    try:
        yes = True
        for i in text.split():
            i = SplitChar(i)
            if i[-1] == "d" and i[-2] == "e":  
                yes = False
                break
        if yes:
            y = nltk.pos_tag(text.split())
            first   = y[0][1]
            second  = y[1][1]
            if "VBG" == first and SplitChar(second)[0] == "R":
                x = "are you"+" "+text
            elif "PRP" == first and "VBG" == second:
                x = "are"+" "+text 
            elif "PRP" == first and "RP"  == second:
                x = "are"+" "+text
                             
            elif "IN" == first and "PRP"  == second:
                x = "do you"+" "+text

            try:
                third  = y[2][1]
                if "DT" == first and "JJ" == second and third  == "JJ":
                    x = "it is"+" "+ initial
                if "RB" == first and "RB" == second and third  == "JJ":
                    x = "it is a"+" "+ initial
              
            except Exception:
                pass
                
            return x
            
    except Exception:
        pass
    
    return initial


# def removeCanYou(text):
#     lstr = " kindly please will favour would can you"
#     backup = [
#             "do me a favor and",
#             "do me a solid and",
#             "do me a favor",
#             "do me a solid",
#             "i need you to"
#             ]
            
#     strn = ""
#     for i in text.lower().split():
#         if i not in lstr:
#             break
#         else:
#             strn += i+" "

#     i = strn[:-1]
#     x = len(SplitChar(i))
#     text = (text.lower()[:x]+text[x:]).replace(i, "")
#     k = len(text)
#     for i in backup:
#         text = text[:20].replace(i, "") +text[20:]
#         if len(text) != k:
#             break

#     return ' '.join(map(str, text.split()))


def removeCanYou(text):
    try:
        Text = ' '.join(map(str, text.split()[:10]))
        # print(Text)

        examples = [
        "can you please",
        "could you please",
        "could you tell me",
        "can you tell me",
        "do me a favor and",
        "do me a favor",
        "do me a solid and",
        "do me a solid",
        "i need you to",
        "can you",
        "tell me",
        "would you",
        "kindly tell me",
        ]

        one_word = [
            "kindly",
            "please",
        ]

        try:
            for i in examples:
                if i in " "+Text+" ":
                    id = i

            for i in one_word:
                if i in " "+Text.split()[0]+" ":
                    id = i
                    break

            pussy_words = ["would", "can", "you"]
            x, y, z = text.partition(id)

            # print(z)
            # print(z.split()[0])
            if z.split()[0] == "please" or z.split()[0] == "kindly":
                z = " ".join(map(str, z.split()[1:]))

            # print("x", x, "y", y, "z", z)
            word_tokens = nltk.word_tokenize(x) 
            filtered_sentence = [w for w in word_tokens if not w in stopwords.words() + pussy_words]
            # print(len(filtered_sentence))

            useful_words =["what","where","when","why","how ","who", "is"]
            for i in useful_words:
                if i in x.lower():
                    return text

            # print(id)
            if len(filtered_sentence) < 2:
                return " ".join(map(str, z.split()))
            else:
                return text
        except UnboundLocalError:
            return text

    except Exception as e:
        print(e)
    return text


def myExceptions(text):
    f = open('Text exceptions.txt')
    objects = []
    for i in f.readlines():
        i = i.replace("\n", "")
        objects.append(i.split("@"))
    f.close
    for i in objects:
        if i[0].strip() in ' ' + text:
            text = text.replace(i[0], i[1])
        " ".join(map(str, text.split()))
    return text


def Acceptance(text):
    tree = nltk.pos_tag(text.split())
    condition  = False
    firstPerson, secondPerson = False, False
    list1 = ["mine", "i am", "am i", "my", "i", "me", "myself", "i'm"]
    list2 = ["you're", "you are", "yours", "your", "you", "yourself"]
    for j, i in enumerate(tree):
        if "VB" in i[1]:
            if j > 1:
                if "NN" in tree[j-1][1]:
                    condition = True
                    break

        if i[1] == "NNP":
            condition = True
            break
    #demonstrative pronouns and regular pronouns
    list3 = ["that","this","those","these","they", "he", "his", "her", "she", "there", "their", "our"]
    for i in list3:
        for j in tree:
            if i in j[0].lower():
                condition = False
                break
    #Adding to memory files
    if condition:
        for i in list1:
            if spacer(i) in spacer(text.lower()):
                firstPerson = True
        for i in list2:
            if spacer(i) in spacer(text.lower()):
                secondPerson = True
        if spacer(user.lower()) in spacer(text.lower()):
            firstPerson = True
        if spacer(forname.lower()) in spacer(text.lower()):
            firstPerson = True
        if spacer(surname.lower()) in spacer(text.lower()):
            firstPerson = True
        if spacer(name.lower()) in spacer(text.lower()):
            secondPerson = True
        if secondPerson:
            with open("AI Context.txt", "a") as file:
                file.write("\n"+str(ConvertFirstToFirstSecond(text))+".")
                file.close()
        elif firstPerson:
            with open("User Context.txt", "a") as file:
                file.write("\n"+str(ConvertFirstToFirstSecond(text))+".")
                file.close()
        else:
            with open("General Context.txt", "a") as file:
                file.write("\n"+str(text)+".")
                file.close()

#End of filtereing Wrapper





#Post Question Filtering
def i_just_said(text):
    try:
        f = open("System_Memory/MachineOutput.txt", "r", encoding='utf-8')
        phrases = []
        for i in f.readlines():
            phrases.append(i.replace("\n", ""))
        f.close()
        if "none" not in str(text).lower() and len(re.findall('[0-9]+', text)) == 0:
            if "i Just said" not in text and "like i said" not in text:
                if similar(text, phrases[-1]) >= 0.72:
                    return "i Just said, {}".format(text)
                elif similar(text, phrases[-2]) >= 0.72:
                    return "like i said, {}".format(text)
        
        return text
    except Exception:
        pass

    return text


def sent_tokens(text):
    text = str(text)
    text = text.replace(". ...", ".").replace("...", ".").replace(". ..", ".").replace("∞", "infinity")
    if len(SplitChar(text)) < 100:
        return[text]

    n = 100
    split_ = [text[i:i+n] for i in range(0, len(text), n)]
    p = []
    for i in split_[:-1]:
        x = i[-50:].split(" ")[-3:-1] #20
        p.append(i.replace(" ".join(x), " ".join(x) + "?????", 1))

    p.append(split_[-1])
    return ''.join(p).split("?????")

#End of Post Question Filtering





#Speaker Verification System Detect Speaker
# def SpeakerVerification(filePath):
#     wav = preprocess_wav(filePath)
#     embeds_b = encoder.embed_utterance(wav)
#     utt_sim_matrix = (np.inner(embeds_a, embeds_b)).item()
#     # print("speaker:", utt_sim_matrix)
#     if round(utt_sim_matrix, 1) > float(0.72):
#         return user
#     else:
#         return "Unknown"

def PersonVerification(filePath):
    wav = preprocess_wav(filePath)
    embeds_b = encoder.embed_utterance(wav)
    utt_sim_matrix = (np.inner(embeds_a, embeds_b)).item()
    # print("speaker:", utt_sim_matrix)
    if round(utt_sim_matrix, 1) > float(verification_threshold_regression):
        return user
    else:
        return "Unknown"


# def transformerDetectSarcasm(sequence_to_classify):

#     if len(nltk.word_tokenize(sequence_to_classify)) > 8:
#         sentiment = Transformer_sentiment(sequence_to_classify)
#         candidate_labels = ['sarcastic', 'genuine']
#         answer = classifier(sequence_to_classify, candidate_labels)
#         print(answer['labels'][0], answer['scores'][0])

#         if str(answer['labels'][0]) == 'sarcastic' and float(answer['scores'][0]) > 0.75:
#             if sentiment == "POS":
#                 return "I apologize"
#             else:
#                 return "I detect sarcasm"

#         else:
#             return None
#     else:
#         return None






#Pytorch DialoGPT-3 and OpenAI Wrapper
def Get_Parameters(configuration):
    with open('Models.json') as json_file:
        data = json.load(json_file)
        for p in data[configuration]:
            model = p["Engine"]
            max_tokens = p["Max tokens"]
            temperature = p["Temperature"]
            top_p = p["Top p"]
            frequency_penalty = p["Frequency penalty"]
            presence_penalty = p["Presence penalty"]

    json_file.close()
    return model, max_tokens, temperature, top_p, frequency_penalty, presence_penalty


def OpenAIConfig(prompt):
    prompt = prompt.replace(name, "").replace(name.lower(), "")
    prompt = ' '.join(map(str, removeCanYou(prompt).split()[:4]))
    # print(prompt)

    configuration = None
    with open("Commands.json") as json_file:
        data = json.load(json_file)

    json_file.close()
    VoiceActivatedCommands = tuple(data.items())
    for key in VoiceActivatedCommands:
        if " " + key[0] in " " + prompt.lower() + " ":
            # print(key[0])
            configuration = key[1]
            break

    # print(configuration)
    # print(type(configuration))
    if bool(configuration) is None:
        return None
    else:
        return configuration



def GeneralOpenAiResponse(prompt):
    # print(prompt)
    if len(nltk.word_tokenize(prompt)) >= 5:
        POSattributes =  []
        doc = nlp(prompt)
        for token in doc:
            if str(token.pos_) != "DET":
                POSattributes.append(str(token.pos_))

        if str(POSattributes[0]) == "VERB":
            if str(POSattributes[1]) == "PRON" or str(POSattributes[1]) == "PROPN" or str(POSattributes[1]) == 'ADP':
                return False
            else:
                return True
        else:
            return False
    else:
        return False




def OpenAIAnswer(prompt, configuration):
    print("OPEN-AI Tokens incresed")
    try:
        model, max_tokens, temperature, top_p, frequency_penalty, presence_penalty = Get_Parameters(configuration)

        response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
        )

        stud_obj = json.loads(str(response))
        return stud_obj["choices"][0]['text'].replace("\n", ", ").replace(", , ", "")

    except openai.error.RateLimitError as TokensExceeded:
        print(TokensExceeded)
        print("Update API Key in Settings file!")
        f"This advanced speech feature has not been configured. the current more is {mode}"

    except Exception as WifiError:
        print(WifiError)
        return "This feature is not accessible, connect to the Internet!"


def GPTResponse(text):

    question_words = ["what", "why", "when", "where", "how", "whose", "who", "which"]

    if len(nltk.tokenize.word_tokenize(text)) == 1:
        cant  = True
        for i in question_words:
            if i == text.lower():
                cant = False

        if cant:
            return None

    # print("GPTResponse")
    global step_
    global chat_history_ids
    new_user_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step_ > 0 else new_user_input_ids
    chat_history_ids = modelGPT.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)



# new_user_input_ids = tokenizer.encode("Hello" + tokenizer.eos_token, return_tensors='pt')
# chat_history_ids = modelGPT.generate(new_user_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
















#VAD Design
def Listener():
    global VAD_frames
    triggered = False
    ring_buffer_flags = [0] * windowChunk
    ring_buffer_index = 0
    ring_buffer_flags_end = [0] * windowChunkEnd
    ring_buffer_index_end = 0
    StartTime = time.time()
    start_point_time = time.time()

    while True:
        data = VAD_stream.read(VAD_chunk, exception_on_overflow = False)
        VAD_frames.append(data)

        TimeUse = time.time() - StartTime
        active = vad.is_speech(data, VAD_rate)
        ring_buffer_flags[ring_buffer_index] = 1 if active else 0
        ring_buffer_index += 1
        ring_buffer_index %= windowChunk
        ring_buffer_flags_end[ring_buffer_index_end] = 1 if active else 0
        ring_buffer_index_end += 1
        ring_buffer_index_end %= windowChunkEnd
        if not triggered:
            global global_start_point_time
            global_start_point_time = time.time()
            num_voiced = sum(ring_buffer_flags)
            if num_voiced > 0.8 * windowChunk:
                triggered = True

        else:
            global end_point_time
            end_point_time = time.time()
            num_unvoiced = windowChunkEnd - sum(ring_buffer_flags_end)
            if num_unvoiced > 0.8 * windowChunkEnd or TimeUse > 15: #15
                triggered = False
                break


    wf = wave.open("full.wav", 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(VAD_p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(VAD_rate)
    wf.writeframes(b''.join(VAD_frames))
    wf.close()
    startTime = (((end_point_time - start_point_time) - (end_point_time - global_start_point_time))-0.7)*1000
    endTime = +((end_point_time - start_point_time))*1000

    if end_point_time - start_point_time < 2.5:
        myfile = AudioSegment.from_file("full.wav")
        myfile.export("ending.wav", format="wav")
    else:
        myfile = AudioSegment.from_file("full.wav")
        extract = myfile[startTime:endTime]
        extract.export("ending.wav", format="wav")
        
    VAD_frames.clear()


#Speech recognition
def predict(file):
    try:
        harvard = Sr.AudioFile(file)
        with harvard as source:
            audio = recognition.record(source)
        initial =  recognition.recognize_google(audio)
    except:
        initial = None

    return initial


#Gender Recognition

def extract_gender_feature(file_name, **kwargs):
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    # print(mfcc, chroma, mel, contrast, tonnetz)
    X, sample_rate = librosa.core.load(file_name)
    if chroma or contrast:
        stft = np.abs(librosa.stft(X))
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
        result = np.hstack((result, mel))
    if contrast:
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, contrast))
    if tonnetz:
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
        result = np.hstack((result, tonnetz))
    return result

def predict_gender(audio_path):
    features = extract_gender_feature(audio_path, mel=True).reshape(1, -1)
    male_prob = gender_model.predict(features)[0][0]    
    # print(male_prob)  
    if male_prob > (1 - male_prob):
        return male_ref
    else:
        return female_ref



#Interupt functions
def SpeakerVerification(filePath):
    wav = preprocess_wav(filePath)
    embeds_b = encoder.embed_utterance(wav)
    utt_sim_matrix = (np.inner(embeds_a, embeds_b)).item()
    if testing_speaker_verification:
        print("speaker:", utt_sim_matrix)
    if utt_sim_matrix > float(verification_threshold):
        return user
    else:
        return "Unknown"

def createPartial(frames_for_file, filename):
    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(VAD_p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames_for_file))
    wf.close()

def rms(frame):
    count = len(frame) / 2
    format = "%dh" % (count)
    shorts = struct.unpack(format, frame)
    sum_squares = 0.0
    for sample in shorts:
        n = sample * SHORT_NORMALIZE
        sum_squares += n * n
    rms = math.pow(sum_squares / count, 0.5)
    return rms * 1000



#Pytorch Synthesis Wrapper
def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_checkpoint(path, model, optimizer, reset_optimizer):
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    return model
def inv_spectrogram(spectrogram):
    S = _db_to_amp(_denormalize(spectrogram) + ref_level_db)  # Convert back to linear
    processor = _lws_processor()
    D = processor.run_lws(S.astype(np.float64).T ** power_)
    y = processor.istft(D).astype(np.float32)
    return inv_preemphasis(y)

def _inv_preemphasis(x, coef=0.97):
    b = np.array([1.0], x.dtype)
    a = np.array([1.0, -coef], x.dtype)
    return signal.lfilter(b, a, x)

def inv_preemphasis(x):
    return _inv_preemphasis(x, preemphasis)
  
def _denormalize(S):
    return (np.clip(S, 0, 1) * -min_level_db) + min_level_db

def _lws_processor():
    return lws.lws(fft_size, hop_size, mode="speech")

def _db_to_amp(x):
    return np.power(10.0, x * 0.05)

def _tts(model, text, p=0, speaker_id=0, fast=False):
  model = model.to(device)
  model.eval()
  if fast:
      model.make_generation_fast_()

  sequence = np.array(_frontend.text_to_sequence(text, p=p))
  sequence = torch.from_numpy(sequence).unsqueeze(0).long().to(device)
  text_positions = torch.arange(1, sequence.size(-1) + 1).unsqueeze(0).long().to(device)
  speaker_ids = None if speaker_id is None else torch.LongTensor([speaker_id]).to(device)

  # Greedy decoding
  with torch.no_grad():
      mel_outputs, linear_outputs, alignments, done = model(
          sequence, text_positions=text_positions, speaker_ids=speaker_ids)

  linear_output = linear_outputs[0].cpu().data.numpy()

  # Predicted audio signal
  waveform = inv_spectrogram(linear_output.T)
  return waveform

def tts(model, text, file_path, p=0, speaker_id=0, fast=True, figures=False):
  waveform = _tts(model, text, p, speaker_id, fast)
  scipy.io.wavfile.write(file_path, rate=fs, data=waveform)

# End of Pytorch Synthesis Wrapper



#Main event
def removeConcatFiles():
    try:
        for filename in os.listdir("Concatenate/"):
            os.remove(f"Concatenate/{str(filename)}")
    except OSError:
        pass

def ConcatenateAudiofiles():
    combined = AudioSegment.from_file("Concatenate/sound0.wav", format="wav")
    for filename in sorted(os.listdir("Concatenate/")):
        if str(filename) != "sound0.wav":
            combined += AudioSegment.from_file(f"Concatenate/{str(filename)}", format="wav")

    combined.export("combined.wav", format="wav")



def inference_loop(initial):
    global mixer
    global last_thing_said_time

    text = myExceptions(' ' + initial[:1].lower() + initial[1:] +' ')
    text = ' '.join(map(str, text.split()))
    mixer.music.stop()

    print(f"{Fore.LIGHTCYAN_EX}{text}{Style.RESET_ALL}")

    try:
        answer = pipeline_data(text, text.replace(name,"").replace(name.lower(),""))
    except Exception as pipeline_error:
        import traceback
        traceback.print_exc()
        print(pipeline_error)
        answer = None



    if answer != None:
        print(f"{Fore.GREEN}answer: {answer}{Style.RESET_ALL}")

        # concatenate audio files 
        Answer_sentences = sent_tokens(str(answer))
        removeConcatFiles()
        for number, sentence in  enumerate(Answer_sentences):
            NNFileName = str(f"Concatenate/sound{number}.wav")
            tts(model, sentence, NNFileName, speaker_id=speakerid, figures=False)

        ConcatenateAudiofiles()
        file_path = "combined.wav"
        data, samplerate = sf.read(file_path)
        sf.write(file_path, data, samplerate)
        song = AudioSegment.from_wav("combined.wav")
        # song = song + 20
        sil_duration = 0.3 * 1000
        one_sec_segment = AudioSegment.silent(duration = sil_duration)
        final_song =  song + one_sec_segment
        final_song.export("combined.wav", format="wav")
        last_thing_said_time = time.time()
        mixer.music.load("combined.wav")
        mixer.music.play()
        
    return answer

#Pipeline Wrapper
def pipeline_data(Unchanged_text, initial):
    global scenario
    global step_
    global attention
    global starting_attention

    # x_phase = False

    object_rotation = predict_gender('ending.wav') #"Boss"
    speaker_rotation = PersonVerification('ending.wav')
    print(f"{Fore.LIGHTGREEN_EX}Speaker: {speaker_rotation}{Style.RESET_ALL}")

    #Text filtering
    text = pronounSolver(nlpInterpolation(exceptions(initial)))
    # print(text)
    initial = text
    # Acceptance(text) #putting things into memory if need be
    # text = question_index_filter(text)
    text = removeCanYou(text) #getting the wording intent right
    #end of text filtering


    #attention
    # print(Unchanged_text)
    attention = Attention(Unchanged_text)

    # print(text)
    print(Unchanged_text.lower(), name.lower(), str(AttentionHistory()), "False")
    if Unchanged_text.lower() == name.lower() and str(AttentionHistory()) == "False":

        with open("System_Memory/SpeakerInput.txt", "r",encoding='utf-8') as j:
            ans = j.readlines()
            j.close()

        ans.reverse()
        for i in ans:
            if i.lower() != name.lower():
                text = i.replace("\n", "")
                Unchanged_text = i.replace("\n", "")
                break
        print(f"{Fore.MAGENTA}{text}{Style.RESET_ALL}")

    log_attention_history(str(attention))
    print("Attention:", attention)

    #Intent Filtering
    text = pastTense(text, "")
    if isinstance(text, bytes):
        text = text.decode()

    argument_object = cortex(removeCanYou(Unchanged_text))
    scenario = argument_object #setting defult scenario

    # End of Final text filtering
    text = removeCanYou(text)

    print("argument:", argument_object)
    #Natural language user interface is engadged

    if attention_always_on:
        attention = True
        print(f"{Fore.GREEN}Attention is always on{Style.RESET_ALL}")


    if attention == True:

        #Response to Name
        print("::"+Unchanged_text+"::")
        if (Unchanged_text).lower() == name.lower() or Unchanged_text.lower() == name.lower():
            starting_attention = time.time()
            step_ = 0
            return f"Yes {object_rotation}?"


        # Designed for scientific purposes
        if str(argument_object).lower() == "none" or argument_object == "OpenAi Function":
            # if "read project" not in last_situation():
            if "open a page" in text.lower() or "open a webpage" in text.lower():
                if len(text.split(" ")) < 5:
                    Responses = ["I don't think i should be opening anything",
                                "I'm not sure I quite understand what you want me to open",
                                "Im not sure I quite catch your drift",
                                ]

                    spfa = random.choice(list(Responses))
                else:
                    try:
                        openpages(text)

                        spfa = f"Opening Webpage {object_rotation}"
                    except Exception as e:
                        Responses = ["Connect to the Internet!",
                                    "No Internet detected!",
                                    "Connect to Wifi!",
                                    "No Signal detected!"
                                    ]

                        spfa = random.choice(list(Responses))


        #might have to re-arange this at some point
        print(argument_object, "argument_object-982453984034")
        if spacer("not") in spacer(text.lower()) and last_situation() == "memory":
            print("check-5059e9ru340ru34083")
            description = conversation_generator(text, object_rotation)
            argument_object = description[0]
            spfa = description[1]

        elif argument_object == "Read project":
            print("check-5050495384538945")
            spfa = i_just_said(serial_println(Unchanged_text, initial, text, object_rotation, argument_object, speaker_rotation))

        
        #Second Case

        elif argument_object == None or "new project" in str(last_situation()).lower() or "x" in last_situation() or last_situation() == "read project4":


            print("1", argument_object == None)
            print("2", "new project" in str(last_situation()).lower())
            print("3", "x" in last_situation())
            print("4", last_situation() == "read project4")

            print(argument_object, "argument_object-42853443")
            description = conversation_generator(text, object_rotation)
            print("check-384538945734895") #for some reason if the arguemt from here is question, the arguemnt becomes none after entering the zone
            argument_object = description[0]
            spfa = description[1]

            if str(spfa).lower() =="none" and argument_object != None:
                print("check-505905955")
                print(argument_object, "argument_object-8798678")
                spfa = i_just_said(serial_println(Unchanged_text, initial, text, object_rotation, argument_object, speaker_rotation))


        #Best Case
        elif argument_object is not None: #or x_phase
            print("check-1-2-3")
            spfa = i_just_said(serial_println(Unchanged_text, initial, text, object_rotation, argument_object, speaker_rotation))


        #Logging Results to System Memory Files
        if str(spfa).lower() != "none":
            print("check-50590595eijer5")
            write_machine(spfa)
            if argument_object !="Question" or argument_object !="Assertion" and re.findall('[0-9]+', str(argument_object)) == []:
                write_answer_situation(Unchanged_text +" "+ spfa)

        #reference scenario in a global namespace, if situation needs to be changed; basicaly if i need to skip straight to a situation
        scenarios = {
            "google none":"question2",
            "memory":"memory",
            "memory1":"memory1",
            "assertion memory":"assertion memory",
            "NewUser":"newuser",
            "recommender1":"recommender1",
            "draw1":"draw1",
            "read project1":"read project1",
            "read project2":"read project2",
            "read project3":"read project3",
            "transformer content":"transformer content"
            }

        continueSet = True
        for i in scenarios.items():
            if scenario == i[0]:
               print(i[0], scenario)
               situation(i[1])
               continueSet = False
               break
            
        print(continueSet)
        if continueSet:
            situation(str(argument_object.lower()))
                
        # if last_situation() !=  "question1":
        write_last_thing_y(text)

        import traceback
        import sys
        #detect sarcasm here
        if str(spfa).lower() == "none":
            try:
                x = sarcasm_model.predict_sarcasm(text)
                print(text, "->", x)

                if x == "It's a sarcasm!":
                    print("True")
                    spfa = sarcastic_case_responses(text)
                    try:
                        print(text + " -> " + spfa)
                    except TypeError:
                        pass
                # spfa = transformerDetectSarcasm(text)
                    if spfa is not None:
                        argument_object = "sarcasm"
                        situation(str(argument_object.lower()))
            except Exception as e:
                print("sarcasm error",e)

                print(traceback.format_exc())
                # or
                print(sys.exc_info()[2])
                spfa = None


        #GPT-3 and OpenAi Wrapper
        if str(spfa).lower() == "none":
            # print("Open AI")
            # config = OpenAIConfig(text)

            # #Response to Name
            # if (Unchanged_text).lower() == name.lower() or Unchanged_text.lower() == name.lower():
            #     starting_attention = time.time()
            #     step_ = 0
            #     return f"Yes {object_rotation}?"

            #GPT Answer
            # else:
            spfa = GPTResponse(Unchanged_text.replace(name,""))

            step_ += 1
            # print(similar(spfa, Unchanged_text))
            if similar(spfa, read_last_line_y()) > 0.7 or similar(spfa, read_last_line_y_two()) > 0.7 or similar(spfa, read_last_line_y_three()) > 0.7 or similar(spfa, Unchanged_text) > 0.8:
                step_ = 0

            #if answer simmilar to something then remove

            annoying_answers = [
                "I'm not sure if you're serious or not, but I'm going to assume you're serious",
                "I'm not sure if you're serious or not, but I'm not sure either.",
                "but I'm not sure if you're being serious or not",
                "downvote",
                "upvote"
            ]

            if str(spfa).count("you're glad") >= 3 or str(spfa).count("I'm not sure if you're serious") >= 3:
                print("there was an annoying answer")
                spfa = None
                step_ = 0

            for i in annoying_answers:
                if i in str(spfa):
                    print("there was an annoying answer")
                    spfa = None
                    step_ = 0
                


            write_machine(spfa)
            if step_ == 0:
                spfa = None

            if step_ >= 4:
                step_ = 0

        return spfa

    write_last_thing_y(text)
    return

#End of Pipeline Wrapper






new_user_input_ids = tokenizer.encode("Hello" + tokenizer.eos_token, return_tensors='pt')
chat_history_ids = modelGPT.generate(new_user_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

removeConcatFiles()


#object oriented programming
mixer.init()
VAD_p = pyaudio.PyAudio()
vad = webrtcvad.Vad(3)
recognition = Sr.Recognizer()

with Sr.Microphone() as source:
    recognition.adjust_for_ambient_noise(source)

#Neural net DeepVoice model
use_cuda = torch.cuda.is_available()
_frontend = getattr(frontend, 'en')
model_ = torch.load(DeepVoiceModel)
model = load_checkpoint(checkpoint_path, model_, None, True)

#tensorflow models
gender_model  = keras.models.load_model("Neural_net_model/gender_model.h5")
predict_gender("combined.wav")

#stream
VAD_rate = 16000
chunkDuration = 30
VAD_chunk = int(VAD_rate * chunkDuration / 1000)
windowChunk = int(400 / chunkDuration)
windowChunkEnd = windowChunk * 2
VAD_stream = VAD_p.open(format=pyaudio.paInt16,
                channels=1,
                rate=VAD_rate,
                input=True,
                start=False,
                frames_per_buffer=VAD_chunk)

VAD_stream.start_stream()


encoder = VoiceEncoder()
fpath = Path("test2.wav")
p_wav = preprocess_wav(fpath)
embeds_a = encoder.embed_utterance(p_wav)

#use vad.py ad say: "Good afternoon Merlin"
# verification_threshold = 0.67
# verification_threshold_regression = 0.55
#Hparams synthisis
# speakerid = 98
# preemphasis=0.97
# min_level_db = -100
# ref_level_db = 20
# _power = 1.4
# fft_size = 1024
# hop_size = 256
# fs = 22050
# user = "Ben Knighton"


#most advanced Vad Speech Recognition Design
generated_1 = "SHORTTEST_1.wav"
partial = "speech_before_vad.wav"


SHORT_NORMALIZE = (1.0/32768.0)

do_speech_recognition = False
ending = False
global_stop = False
initialAttention = True

frames = []
VAD_frames_whilst = []
speaker_results = []
speaking_frames = []
VAD_frames = []

starting_attention = time.time()
step_ = 0


amount_of_too_long_time = 20
last_thing_said_time = time.time()


# class ceaseAudio(threading.Thread):
    
#     def __init__(self, name="ceaseAudio"):
#         threading.Thread.__init__(self, name=name)

#     def run(self):
#         global mixer
#         global global_stop

#         while True:
#             input_ = input("-> ")
#             if input_ == "stop":
#                 VAD_stream.stop_stream()
#                 print(f"{Fore.RED}Audio stream: OFF{Style.RESET_ALL}")
#                 mixer.music.stop()
#                 mixer.quit()
#                 global_stop = True

#                 while True:
#                     input_ = input("-> ")
#                     if input_ == "start":
#                         VAD_stream.start_stream()
#                         print(f"{Fore.GREEN}Audio stream: ON{Style.RESET_ALL}")
#                         mixer.init()
#                         global_stop = False
#                         break

# thread = ceaseAudio()
# thread.start()


# print("started!")
# while True:

#     try:
#         if not mixer.music.get_busy():
#             VAD_frames_whilst = [] #resets
#             speaker_results.clear()
#             speaking_frames.clear()

#         if not mixer.music.get_busy():
#             Listener() #listening
#             initial = predict("ending.wav")
#             if initial is not None:
#                 print("prediction:", initial)
#                 inference_loop(initial)
#                 # say_something()

#         else:
#             data = VAD_stream.read(3000, exception_on_overflow=False) #listening for interupts
#             frames.append(data)
#             VAD_frames_whilst.append(data)
#             if len(frames) >= 3:
#                 createPartial(frames[-3:], generated_1)
#                 f = sf.SoundFile(generated_1)
#                 total_time = len(f) / f.samplerate
#                 x = SpeakerVerification(generated_1)
#                 speaker_results.append(x)
#                 if len(speaker_results) > 2:
#                     if speaker_results[-1] is user and speaker_results[-2] is user:
#                         start_time = time.time()
#                         speaker_results.append(user)
#                         mixer.music.pause()
#                         active = True
#                         while active:
#                             data = VAD_stream.read(3000, exception_on_overflow = False)
#                             frames.append(data)
#                             createPartial(frames[-3:], generated_1)
#                             f = sf.SoundFile(generated_1)
#                             total_time = len(f) / f.samplerate
#                             speaker = SpeakerVerification(generated_1)
#                             speaker_results.append(speaker)
#                             VAD_data = VAD_stream.read(VAD_chunk, exception_on_overflow = False)
#                             active = vad.is_speech(VAD_data, VAD_rate)
#                             speaking_frames.append(active)
#                             VAD_frames_whilst.append(data)
#                             # print(speaking_frames.count(True))
#                             if speaking_frames.count(True) > 3:
#                                 do_speech_recognition = True


#             if do_speech_recognition:
#                 mixer.music.pause()
#                 createPartial(VAD_frames_whilst, partial)
#                 speaker_results.clear()
#                 f = sf.SoundFile(partial)
#                 total_time = len(f) / f.samplerate
#                 trim_time = total_time - (time.time() - start_time) - 0.7
#                 if trim_time > total_time:
#                     trim_time = 0

#                 myfile = AudioSegment.from_mp3(partial)
#                 extract = myfile[trim_time*1000:]
#                 extract.export("twoseconds.wav", format="wav")
#                 initial = predict("twoseconds.wav")
#                 db_f = wave.open("twoseconds.wav", "rb")
#                 db_frames = db_f.readframes(int(db_f.getnframes()))
#                 db = rms(db_frames)
#                 print("DECIBELS", db)
#                 if initial is None or db < db_threshold:
#                     mixer.music.unpause()
#                 else:
#                     print("INITIAL:", initial)
#                     mixer.music.stop()
#                     inference_loop(initial)
#                     # say_something()
            
#             else:
#                 mixer.music.unpause()

#             do_speech_recognition = False

#     except KeyboardInterrupt:
#         print("aaaahhhhh!!!")
#         VAD_stream.stop_stream()
#         #delete memory heavy variables in this space
#         state = True
#         while state:
#             definition = input(">")
#             if definition == "start":
#                 print("loading...")
#                 VAD_stream.start_stream()
#                 #re initiate memory heavy variables here
#                 print("started!")
#                 state = False
#             elif definition == "end":
#                 raise SystemExit("ending boss")

#     except Exception as error:
#         if global_stop is False:
#             raise Exception(error)
#         else:
#             print(error)


initialAttention = True

starting_attention = time.time()
step_ = 0


amount_of_too_long_time = 20
last_thing_said_time = time.time()







class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.setFixedSize(906, 513)
        self.History = ""
        font = QtGui.QFont()
        font.setFamily("OCR A Extended")
        # font.setFamily("0Arame")
        font.setPointSize(26)
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(0, 0, 906, 513))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("/Users/benkn/Downloads/6.jpg"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.scrollArea = QtWidgets.QScrollArea(Form)
        self.scrollArea.setGeometry(QtCore.QRect(130, 120, 681, 251))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollArea.setStyleSheet("background-color: transparent;\n")
        self.scrollAreaWidgetContents = QtWidgets.QLabel(Form)
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 679, 239))
        self.scrollAreaWidgetContents.setFont(font)
        self.scrollAreaWidgetContents.setStyleSheet("background-color: transparent;")
        self.scrollAreaWidgetContents.setWordWrap(True)
        self.scrollAreaWidgetContents.setScaledContents(True)
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.scrollAreaWidgetContents.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.textEdit = QtWidgets.QTextEdit(self.scrollAreaWidgetContents)
        self.textEdit.setGeometry(QtCore.QRect(0, 0, 681, 251))
        self.textEdit.setFont(font)
        self.textEdit.setStyleSheet("background-color: transparent;")
        self.textEdit.setObjectName("textEdit")
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.lineEdit = QtWidgets.QLineEdit(Form)
        self.lineEdit.setGeometry(QtCore.QRect(130, 370, 681, 31))
        self.lineEdit.setFont(font)
        self.lineEdit.setStyleSheet("background-color: transparent;")
        self.lineEdit.setAttribute(QtCore.Qt.WA_MacShowFocusRect, 0)
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit.returnPressed.connect(self.on_pushButtonOK_clicked)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "NLUI"))
        self.lineEdit.setText(_translate("Form", ""))

    def on_pushButtonOK_clicked(self):




        initial = self.lineEdit.text()

        mixer.music.stop()
        answer = inference_loop(initial)


        # global mixer
        # text = myExceptions(' ' + initial[:1].lower() + initial[1:] +' ')
        # text = ' '.join(map(str, text.split()))
        # mixer.music.stop()

        # print(f"{Fore.LIGHTCYAN_EX}{text}{Style.RESET_ALL}")

        # try:
        #     answer = pipeline_data(text, text.replace(name,"").replace(name.lower(),""))
        # except Exception as pipeline_error:
        #     import traceback
        #     traceback.print_exc()
        #     print(pipeline_error)
        #     answer = None



        # if answer != None:
        #     print(f"{Fore.GREEN}answer: {answer}{Style.RESET_ALL}")

        #     # concatenate audio files 
        #     Answer_sentences = sent_tokens(str(answer))
        #     removeConcatFiles()
        #     for number, sentence in  enumerate(Answer_sentences):
        #         NNFileName = str(f"Concatenate/sound{number}.wav")
        #         tts(model, sentence, NNFileName, speaker_id=speakerid, figures=False)

        #     ConcatenateAudiofiles()
        #     file_path = "combined.wav"
        #     data, samplerate = sf.read(file_path)
        #     sf.write(file_path, data, samplerate)
        #     song = AudioSegment.from_wav("combined.wav")
        #     # song = song + 20
        #     sil_duration = 0.3 * 1000
        #     one_sec_segment = AudioSegment.silent(duration = sil_duration)
        #     final_song =  song + one_sec_segment
        #     final_song.export("combined.wav", format="wav")
        #     last_thing_said_time = time.time()
        #     mixer.music.load("combined.wav")
        #     mixer.music.play()





        #Watch Design
        # self.History += f'<br /><br /><span style="color: red">{prompt}</span>'
        # self.History += f'<br /><br />{answer}'

        self.History += f'<br /><br /><span style="background-color:rgb(11, 19, 28);">{str(initial).upper()}</span>'
        self.History += f'<br /><br /><span style="background-color:rgb(57, 110, 126);">{str(answer).upper()}</span>'

        self.scrollAreaWidgetContents.setText(self.History)
        self.lineEdit.clear()       
        self.scrollArea.verticalScrollBar().rangeChanged.connect(self.scrollToBottom) #ranged changed

    def scrollToBottom(self):
        scrollBar = self.scrollArea.verticalScrollBar()
        scrollBar.rangeChanged.disconnect(self.scrollToBottom)
        scrollBar.setValue(scrollBar.maximum())


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())