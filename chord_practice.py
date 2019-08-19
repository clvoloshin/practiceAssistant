#!/usr/bin/env python3
import argparse
import math
import numpy as np
from collections import OrderedDict

import sounddevice as sd
from pychord import Chord
from pychord.constants import QUALITY_DICT


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

def compose_peak(idx,seq_len,big_list):
    peak_base = big_list[idx-seq_len-1:idx]
    indexes = np.array([x[0] for x in peak_base])
    values =  np.array([x[1] for x in peak_base])
    peak_location = sum(indexes*(values/sum(values)))
    weights = abs(indexes-peak_location)
    peak_value = sum(values*(weights/sum(weights)))
    return peak_location,peak_value

def find_peaks(a,num_deviations=3):
    E = sum(a)/len(a)
    V = sum(a*a)/len(a) - E*E
    s = np.sqrt(V)
    l = enumerate(a)
    big = [x for x in l if x[1] > E+num_deviations*s]
    cur_len = 0
    joint = []
    big+=[(0,0)]
    for i in range(1,len(big)):
        if big[i][0]==big[i-1][0]+1:
            cur_len+=1
        else:
            if cur_len>0:
                joint.append(compose_peak(i,cur_len,big))
            else:
                joint.append(big[i-1])
            cur_len = 0
    return joint

columns = 256

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-l', '--list-devices', action='store_true',
                    help='list audio devices and exit')
parser.add_argument('-b', '--block-duration', type=float,
                    metavar='DURATION', default=50,
                    help='block size (default %(default)s milliseconds)')
parser.add_argument('-c', '--columns', type=int, default=columns,
                    help='width of spectrogram')
parser.add_argument('-d', '--device', type=int_or_str,
                    help='input device (numeric ID or substring)')
parser.add_argument('-g', '--gain', type=float, default=10,
                    help='initial gain factor (default %(default)s)')
parser.add_argument('-r', '--range', type=float, nargs=2,
                    metavar=('LOW', 'HIGH'), default=[10, 2000],
                    help='frequency range (default %(default)s Hz)')
parser.add_argument('-q', '--qualities', type=str, nargs='+',
                    default=[''],
                    help='frequency range (default %(default)s Hz)')
args = parser.parse_args()

low, high = args.range
if high <= low:
    parser.error('HIGH must be greater than LOW')

try:

    if args.list_devices:
        print(sd.query_devices())
        parser.exit(0)

    samplerate = sd.query_devices(args.device, 'input')['default_samplerate']

    delta_f = (high - low) / (args.columns - 1)
    fftsize = math.ceil(samplerate / delta_f)
    low_bin = math.floor(low / delta_f)

    NOTE_MIN = 21       # A0
    NOTE_MAX = 108      # C8
    
    # NOTE_NAMES = 'A A#/Bb B C C#/Db D D#/Eb E F F#/Gb G G#/Ab'.split()
    NOTE_NAMES = 'A A# B C C# D D# E F F# G G#'.split()

    POSSIBILITES = 'A B C D E F G F# C# Gb Db Ab Eb Bb'.split()

    def freq_to_number(f): return 69 + 12 * np.log2(f / 440)
    def number_to_freq(n): return 440. * 2.0**((n - 69) / 12.0)
    def note_name(n): return NOTE_NAMES[(n - NOTE_MIN) % len(NOTE_NAMES)] + str(int(n / 12 - 1))
    # FREQ_STEP = 69/12
    # def note_to_fftbin(n): return number_to_freq(n) / FREQ_STEP

    print()
    if len(args.qualities):
        QUALITIES_ = [x for x in args.qualities]
        QUALITIES = []
        for x in QUALITIES_:
            if x == 'maj':
                x = ''
            if x == 'min':
                x = 'm'
            QUALITIES.append(x)
        assert set(QUALITIES).issubset(set(QUALITY_DICT.keys())), 'Please make sure that your qualities are among: %s' % [x for x in QUALITY_DICT.keys()]
    else:
        QUALITIES = ['', 'm'] #

    print('*'*20)
    print('Choosing Chord Quality among: ', QUALITIES)
    print('*'*20)
    print()
    
    def getNewChord():
        return np.random.choice(POSSIBILITES) + np.random.choice(QUALITIES)

    def chord_to_notes(chord, sharpify=True):
        notes = 'A B C D E F G'.split()
        notes_dic = {key:val for val,key in enumerate(notes)}

        xs = Chord(chord).components()
        out = []
        for x in xs:
            if sharpify and 'b' in x:
                out.append(notes[notes_dic[x[:-1]]-1 % len(notes)] + '#')
            else:
                out.append(x)
        return out

    correct = {x:0 for x in np.hstack([[x + y for y in QUALITIES] for x in POSSIBILITES])}
    def reset():
        global CURRENTCHORD, K
        try: 
            correct[CURRENTCHORD] += 1
        except:
            pass
        CURRENTCHORD = getNewChord()
        K = 0
        print()
        print('Play: ', CURRENTCHORD)
        print('How many correct: ', correct)
        print(chord_to_notes(CURRENTCHORD, False))


    CURRENTCHORD = None
    K = None
    reset() 

    def callback(indata, frames, time, status):
        global K, CURRENTCHORD
        if any(indata):
            fft = np.abs(np.fft.rfft(indata[:, 0], n=fftsize))
            freq = fft.argmax() * delta_f

            out = [x for x in find_peaks(fft, 5) if x[0] >= 10]
            ordering = np.array([x[1] for x in out]).argsort()[::-1]
            notes = np.array([note_name(int(round(freq_to_number(x[0] * delta_f)))) for x in out])

            if len(notes):
                if set(chord_to_notes(CURRENTCHORD)).issubset(set([x[:-1] for x in notes])): 
                    K += 1

                if K > 10:
                    reset()
                    K = 0
        else:
            print('no input')

    with sd.InputStream(device=args.device, channels=1, callback=callback,
                        blocksize=int(samplerate * args.block_duration / 1000),
                        samplerate=samplerate):
        while True:
            response = input()
            if response.lower() in ('', 'q', 'quit'):
                break
except KeyboardInterrupt:
    parser.exit('Interrupted by user')
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))

