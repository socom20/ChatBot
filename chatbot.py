#! /usr/bin/python3

import sys, os
import argparse
import glob
import re

import numpy as np
import pandas as pd
import json

import chatterbot
from chatterbot.trainers import ListTrainer
import chatterbot.response_selection
import chatterbot.comparisons

sys.path.append('../Mitsuku')
sys.path.append('../voice_handler')
sys.path.append('../Acapela_TTS_API')

from mitsuku import PandoraBot
from voice_handler import VoiceHandler
from Acapela_TTS_API import Acapela_API



def my_cf(a,b):
    dist = chatterbot.comparisons.levenshtein_distance(a,b)
##    print(d, a.text,'||',  b.text)
    return dist


class ChatBot:
    def __init__(
        self,
        cfg_file='./chatbot.cfg'):

        self.cfg_d = {
            'use_chatterbot':True,
            'use_mitsuku':True,
            
            'voice_input':True,
            'voice_output':True,

            'similarity_threshold':0.80,
            'default_response':'No entiendo tu pregunta.',

            'corpus_dir':'./corpus',
            'db_dir':'./DataBase',
            'verbose':False,

            'mitsuku_cfg_d':{
                'user_id':'558',
                'bot_name':'David',
                'is_male':True,
                'bot_lang':'es',
                'verbose':False},

            'acapela_cfg_d':{
                'username':None,
                'password':None,
                'credentials_path':'./acapela_credentials.json',
                'listen_speed':180, 'listen_shaping':100,
                'listen_voice':'Spanish - dnn-DavidPalacios_sps',
                'verbose':False},
            
            'voice_listener_cfg_d':{
                'device_idx':None,
                'sample_rate':22050,
                'sample_channels':2,
                'sample_width':2,
                'chunk_size':1024,
                'voice_threshold':0.03,
                'start_secs':0.10,
                'end_secs':0.80,
                'start_buffer_secs':0.50,
                'max_rec_secs':10,
                'voice_language':'es-EC',
                'verbose':False}
            }


        self.cfg_file = cfg_file
        
        if os.path.exists(self.cfg_file):
            try:
                self.load_config()
            except Exception as e:
                print(' - ERROR while reading: "{}".Please check or delete the file.'.format(self.cfg_file))
                sys.exit(1)
                
        else:
            self.save_config()

                        
        self.use_chatterbot = self.cfg_d['use_chatterbot']
        self.use_mitsuku = self.cfg_d['use_mitsuku']
        self.voice_input = self.cfg_d['voice_input']
        self.voice_output = self.cfg_d['voice_output']
        
        self.cb_similarity_threshold = self.cfg_d['similarity_threshold']
        self.cb_default_response = self.cfg_d['default_response']

        
        self.corpus_dir = self.cfg_d['corpus_dir']
        self.db_dir = self.cfg_d['db_dir']
        self.verbose = self.cfg_d['verbose']
        
        self.mitsuku_cfg_d = self.cfg_d['mitsuku_cfg_d']
        self.acapela_cfg_d = self.cfg_d['acapela_cfg_d']
        self.voice_listener_cfg_d = self.cfg_d['voice_listener_cfg_d']

        assert self.use_chatterbot or self.use_mitsuku, 'At least one use_chatterbot or use_mitsuku must be set.'
        
        
        if not os.path.exists(self.corpus_dir):
            os.makedirs(self.corpus_dir)

        if not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir)


        if self.use_chatterbot:
            self._build_cb_chatbot()
            
        if self.use_mitsuku:
            self._build_mk_chatbot()

        if self.voice_input:
            self._start_voice_handler()

        if self.voice_output:
            self._start_acapela()
            
        return None


    def load_config(self):
        if self.cfg_file is not None:
            print('- Loading existent config file:', self.cfg_file)
            
            with open(self.cfg_file, 'r') as f:
                cfg_d = json.loads(f.read())

            for k in self.cfg_d.keys():
                if k in cfg_d.keys():
                    if type(cfg_d[k]) == list:
                        cfg_d[k] = tuple(cfg_d[k])
                        
                    if self.cfg_d[k] != cfg_d[k]:
                        print(' - Default config updated: {} = {} >> {}'.format(k, self.cfg_d[k], cfg_d[k]))
                        
                    self.cfg_d[k] = cfg_d[k]
                else:
                    print(' - WARNING: key:"{}" not found in the "config.json" file.'.format(k), file=sys.stderr)

        return None

    def dump_and_format(self, var={'dict':{'num0':2}, 'num1':3}):
        var_str = json.dumps(var)

        for c in '({[':
            var_str = var_str.replace(c, c+'\n')
        for c in ')}]':
            var_str = var_str.replace(c, '\n'+c)

        var_str = var_str.replace(', ', ',\n')

        var_lines = var_str.split('\n')

        n_t = 0
        str_out_v = []
        for l in var_lines:
            if '}' in l or ')' in l or ']' in l:
                n_t -= 1

            str_out_v.append('\t'*n_t + l)

            if '{' in l or '(' in l or '[' in l:
                n_t += 1

        return '\n'.join(str_out_v)
    
    def save_config(self):

        if self.cfg_file is not None:
            print('- Saving config file:', self.cfg_file)
            with open(self.cfg_file, 'w') as f:
                to_write = self.dump_and_format(self.cfg_d)
                f.write(to_write)
                
        return None
    

    def _read_txt(self, filename='./corpus/conversacion_01.txt'):

        with open(filename, 'r') as f:
            ml = f.readlines()

        ml = [l.replace('\n', '') for l in ml]
        ml = [l for l in ml if len(l) > 0]

        if len(ml) % 2 == 1:
            print(' WARNING conversation has a odd number of lines = {}, filename={}. It appears to have a missing answer.\n Press ENTER to skip this. Or interrupt and fix the file and run again.'.format(len(ml), filename), file=sys.stderr)
            input()
            
        return ml


    def _read_csv(self, filename='./corpus/corpus1.csv'):

        df = pd.read_csv(filename, header=0)
        table = df.values


        conv_m = []
        resp_v = []
        for i in range(table.shape[0]):
            if (type(table[i,0]) == str and table[i,0] == '') or type(table[i,0]) == float:
                continue

            input_text = table[i,0]
            resp_pos_v = [ table[i,j] for j in range(1, table.shape[1]) if type(table[i,j]) == str and table[i,j] != '' ]
            if len(resp_pos_v) > 0:
                resp_v = resp_pos_v

            for resp in resp_v:
                conv_m.append( [input_text, resp] )
                
        return conv_m
    
    def _read_yml(self, fileame='./corpus_cb/conversations.yml'):
        c_d = chatterbot.corpus.read_corpus(fileame)

        if self.verbose:
            print(' - Reading {}:'.format(fileame) )
            print(' |-> Detected: {} conversations.'.format( len(c_d['conversations']) ) )

        conv_m = []
        for i_c in range(len(c_d['conversations'])):
            if type(c_d['conversations'][i_c]) == str:
                c_d['conversations'][i_c] = [ c_d['conversations'][i_c] ]

            n_lines = len(c_d['conversations'][i_c])
                
            conv_m.append( c_d['conversations'][i_c] )

        if self.verbose:
            print('Ok!!, Total number of conversations read:', len(conv_m))
            
        return conv_m
        


    def _start_voice_handler(self):
        self.voice_handler = VoiceHandler(**self.voice_listener_cfg_d)
        return None


    def _start_acapela(self):
        self.acapela = Acapela_API(
            download_dir=os.path.join(self.db_dir, 'acapela_cache'),
            verbose=self.acapela_cfg_d['verbose'],
            )

        if self.verbose:
            print(' Loggin to Acapela ...', end='')
            
        self.acapela.login(
            username=self.acapela_cfg_d['username'],
            password=self.acapela_cfg_d['password'],
            credentials_path=self.acapela_cfg_d['credentials_path'],
            )

        if self.verbose:
            print(' Ok !!')
            
        return None
    
        
    def _build_mk_chatbot(self):
        self.mk_chatbot = PandoraBot(**self.mitsuku_cfg_d)
        return None
    

    def _build_cb_chatbot(self):
        self.cb_chatbot = chatterbot.ChatBot(
            "ChatterBot",

            storage_adapter='chatterbot.storage.SQLStorageAdapter',
            database_uri='sqlite:///{}/ChatterBot_db.sqlite3'.format(self.db_dir),

            statement_comparison_function = my_cf,
            
            logic_adapters=[
                {
                    "import_path": "chatterbot.logic.BestMatch",
                    "response_selection_method": chatterbot.response_selection.get_random_response,
                    "maximum_similarity_threshold": self.cb_similarity_threshold,
                    "default_response": self.cb_default_response,
                },
                
            ],
            
            preprocessors=[
                'chatterbot.preprocessors.clean_whitespace',
                'chatterbot.preprocessors.convert_to_ascii'
            ],
            
            read_only=True,
        )

        self.cb_chatbot.storage.tagger.language = chatterbot.languages.SPA
        
        # Building Trainer
        self.trainer = ListTrainer(self.cb_chatbot, show_training_progress=False)
        return None

        
    def _make_cache_from_text(self, text):
        if not self.voice_output:
            self._start_acapela()
            self.voice_output = True
            
        print(' - Makeing Acapela Cache for: "{}"'.format(text) )
        self._make_output_voice(text)
        return None

    
    def train(self, conv_m, do_acapela_cache=False):
        
        conv_m = list(conv_m)
        if len(conv_m) > 0 and type(conv_m[0]) == str:
            conv_m = [ conv_m ]

        if do_acapela_cache:
            self._make_cache_from_text(self.cb_default_response)
            cached_v = [self.cb_default_response]
            
        for conv_v in conv_m:
            if do_acapela_cache:
                for i_t, text in enumerate(conv_v):
                    if (i_t%2 == 1) and text not in cached_v:
                        self._make_cache_from_text(text)
                        cached_v.append(text)
                
            self.trainer.train(conv_v)
            
        return None


    def get_response(self, input_text='Hola'):
        resp = ''

        if input_text == '':
            resp = self.cb_default_response

        else:
            if self.use_chatterbot:
                resp = self.cb_chatbot.get_response(input_text).text
                if self.verbose:
                    print(' ChatterBot:', resp)

            if self.use_mitsuku and (resp == '' or resp == self.cb_default_response):
                resp = self.mk_chatbot.get_response(input_text)
                if self.verbose:
                    print(' Mitsuku:', resp)
            
        return resp


    def _make_output_voice(self, resp_text):
        self.acapela.speak_and_download_wav(
            text=resp_text,
            listen_speed=self.acapela_cfg_d['listen_speed'],
            listen_shaping=self.acapela_cfg_d['listen_shaping'],
            listen_voice=self.acapela_cfg_d['listen_voice'],
            convert_to_ogg=False,
            when_ogg_rm_wav_file=False,
            use_cache=True)

        return None

    
    def _retrieve_input_text(self):

        if self.voice_input:
            print(' Please talk ...')
            text = ''
            while text == '':
                text = self.voice_handler.rec_speech2text(
                    return_audio_data=False)

            print(' >>>', text)
            
        else:
            text = input(' >>> ')
            
        return text
    

    def _show_output_text(self, resp_text):
        if self.voice_output:
            self._make_output_voice(resp_text)
            print(' <<<', resp_text)
            self.acapela.play()
                
        else:
            print(' <<<', resp_text)
            
        return None

    def _format_text(self, text):
        
        text_ = text.lower()
        text_ = text_.translate( str.maketrans("áéíóú", "aeiou") )

        text_ = re.sub(r'[:.,;\[\]\(\)!¡¿?$%&+-/*="]', ' ', text_)
        
        text_ = re.sub(r'  +', ' ', text_).strip()
        
        return text_
    
    def start_conversation(self):
        print(' Starting conversation')
        
        while True:
            text = self._retrieve_input_text()
            resp_text = self.get_response(text)
            self._show_output_text(resp_text)

        return None


    def clear_chatterbot_db(self):
        for f in glob.glob( os.path.join(self.db_dir, '*.sqlite3') ):
            os.remove(f)
            if self.verbose:
                print(' - Deleted file: "{}"'.format(f))

        f_tokenizer = './sentence_tokenizer.pickle'
        if os.path.exists(f_tokenizer):
            os.remove(f_tokenizer)
            if self.verbose:
                print(' - Deleted file: "{}"'.format(f_tokenizer))


        if self.use_chatterbot:
            self._build_cb_chatbot()

        return None


    def train_on_corpus(self):

        print(' ChatBot Training', end='\n\n')
        do_clear = None
        while do_clear is None:
            print(' Do you want to clear the model database? (y/n): ', end='')
            r = input()
            if len(r) > 0 and r[0].lower() in ['s', 'y']:
                do_clear = True
            elif len(r) > 0 and r[0].lower() in ['n']:
                do_clear = False
            
        
        if do_clear:
            self.clear_chatterbot_db()
        
        
        corpus_files_v = []
        for f in glob.glob( os.path.join(self.corpus_dir, '*.csv') ):
            corpus_files_v.append(f)

        for f in glob.glob( os.path.join(self.corpus_dir, '*.txt') ):
            corpus_files_v.append(f)

        for f in glob.glob( os.path.join(self.corpus_dir, '*.yml') ):
            corpus_files_v.append(f)

        idx_sel = -2
        while not (-1 <= idx_sel < len(corpus_files_v)):
            print(' Please select the file to train on:')
            
            for i_f, f in enumerate(corpus_files_v):
                print(' {:3d} - {}'.format(i_f, os.path.split(f)[-1] ) )

            print(' {:3d} - {}'.format( -1, 'Train on all files !!!') )

            try:
                idx_sel = int( input(' >>> ') )
            except:
                  pass

        if idx_sel != -1:
            corpus_files_v = corpus_files_v[idx_sel:idx_sel+1]


        do_acapela_cache = None
        while do_acapela_cache is None:
            print(' Do you want to make the Acapela Voice cache while training the model? (y/n): ', end='')
            r = input()
            if len(r) > 0 and r[0].lower() in ['s', 'y']:
                do_acapela_cache = True
            elif len(r) > 0 and r[0].lower() in ['n']:
                do_acapela_cache = False

        
        for f in corpus_files_v:
            conv_m = []
            try:            
                print(' Training on: "{}" ...'.format(f))
                    
                ext = os.path.splitext( f )[-1]
                if ext == '.csv':
                    conv_m = self._read_csv(f)
                elif ext == '.txt':
                    conv_m = self._read_txt(f)
                elif ext == '.yml':
                    conv_m = self._read_yml(f)
                else:
                    raise TypeError('Bad FileExtension.')
                    
            except Exception as e:
                print(' - WARNING, Problem while reading "{}": {}'.format(f, e ), file=sys.stderr)
                continue

            if len(conv_m) > 0:
                try:
                    self.train(conv_m, do_acapela_cache)
                except Exception as e:
                    print('\n - WARNING, Problem while training with "{}": {}'.format(f, e ), file=sys.stderr)
                    
        return None

    def start_test(self):
        print(' Starting Mic Test')

        if self.voice_input:
            self.voice_handler.config_device_idx()
            self.voice_handler.autoset_voice_threshold()

            save_config = None
            while save_config is None:
                print(' Do you want to save new config? (y/n): ', end='')
                r = input()
                if len(r) > 0 and r[0].lower() in ['s', 'y']:
                    save_config = True
                elif len(r) > 0 and r[0].lower() in ['n']:
                    save_config = False

            self.cfg_d['voice_listener_cfg_d']['device_idx'] = self.voice_handler.device_idx
            self.cfg_d['voice_listener_cfg_d']['voice_threshold'] = self.voice_handler.voice_threshold
            
            if save_config:
                self.save_config()

        print()
        
        print(' Starting Echo Test')
        while True:
            text = self._retrieve_input_text()
            self._show_output_text(text)
            
        return None


def main(start_mode='conversation'):
    global chatbot
    chatbot = ChatBot()
    
    if start_mode.lower() == 'conversation':
        chatbot.start_conversation()
    elif start_mode.lower() == 'train':
        chatbot.train_on_corpus()
    elif start_mode.lower() == 'test':
        chatbot.start_test()
    else:
        raise ValueError('Chatbot_mode not not recognized, possible modes: "conversation", "train", "test".')
    
    return chatbot


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='conversation', help='Chatbot Start Mode: "conversation", "train", "test"', choices=["conversation", "train", "test"])
    args = parser.parse_args()

    chatbot = main(start_mode=args.mode)

    
    
        
