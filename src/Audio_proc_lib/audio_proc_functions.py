import soundfile as sf 
import librosa
import librosa.display
import numpy as np
from scipy.linalg import toeplitz
#from scipy.fftpack import fft,ifft
from numpy.fft import fft,ifft,rfft,irfft
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import youtube_dl
from Plotting_funcs.Plotting_util_and_other import *
import time
import requests
import re
import os





def load_music():

    flag = int(input('Press 4 if you want to download the glockenspiel signal\nPress 3 if you want to download a sound file from web\nPress 2 if you want to load a song segment of youtube\nPress 1 if you want to load a song segment of your pc\n: ') )

    del_flag = int(input('Press 1 if you want to delete the downloaded file else 0\n:' ) )        

    s = input("Give the desired samplerate\n:")  


    if flag==4:
        url = "https://www.univie.ac.at/nonstatgab/signals/glockenspiel.wav"
        r = requests.get(url, allow_redirects=True)
        filename = 'glockenspiel.wav' 
        open( filename , 'wb').write(r.content)
        audio, s = librosa.load(filename, sr=int(s), mono=True)
        if del_flag:
            os.remove(filename)    





    elif flag==3:
        url = input("Give the url of the sound_file you want to be downloaded\n:") 
        r = requests.get(url, allow_redirects=True)
        #matches the last /...
        filename = re.findall("/(?:.(?!/))+$", url)[0][1:]
        open(filename, 'wb').write(r.content)        
        audio, s = librosa.load(filename, sr=int(s), mono=True)
        if del_flag:
            os.remove(filename)            


    elif flag==2:
        #GETTING TRACK FROM YOUTUBE-------------------------------------
        def my_hook(d):
            if d['status'] == 'finished':
                print('Done downloading...')        

        url = input("Give the url of the song in youtube:") 
        start = int(input("Give the start sec:")) 
        stop = int(input("Give the stop sec:")) 

        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '44100',
            }],
            'outtmpl': '%(title)s.wav',
            'progress_hooks': [my_hook],
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            status = ydl.download([url])


        filename = info.get('title', None) + '.wav'
        audio, s = librosa.load(filename, sr=int(s), mono=True)
        if del_flag:
            os.remove(filename)            

    elif flag==1:
        song_path = input("Give the path of the .wav file that you want to process\n: ")
        audio , s = librosa.load(song_path,sr=int(s))



    if (flag==1 or flag==3 or flag==2):
        start = int(input("Give the start sec:")) 
        stop = int(input("Give the stop sec:"))             
        if not(stop>len(audio)*(1/s)):
            audio = audio[ start*s:stop*s]        
    


    return audio , s
    #------------------------------------------------------------------

def timeis(func):
    '''Decorator that reports the execution time.'''
  
    def wrap(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
          
        print(func.__name__, end-start)
        return result
    return wrap

def plot(x=None,y=None):

    if not(x):
        plt.plot(x,y)
    else:
        plt.plot(y)
    plt.show()


def euclidian_dist(x,y,normalize=1):

    if normalize:
        out = np.linalg.norm( ( x/np.linalg.norm(x) ) - ( y/np.linalg.norm(y) ) )
    else:
        out = np.linalg.norm(x-y)

    return out

def dot(x,y,normalize=1):
    if normalize:
        out = np.dot( ( x/np.linalg.norm(x) ) , ( y/np.linalg.norm(y) ) )
    else:
        out = np.dot(x,y)

    return out


def create_desiered_window(win_type,nb_samples):
    if(win_type=="Hamming"):
        window = signal.hamming(nb_samples)

    if(win_type=="Blackman"):
        window = signal.blackman(nb_samples)

    if(win_type=="Hann"):
        window = signal.blackman(nb_samples)

    if(win_type=="Rectangular"):
        window = signal.boxcar(nb_samples)

    if(win_type=="Gaussian"):
        window = create_gaussian_kernel(shape=(nb_samples,1),sigma=1,typ="1D")

    return window

#analog SINUSOID-----------------------------------------
def get_cos(f,phase_off,dur,s):
    n = np.arange(0,dur, 1/s)
    W = (np.pi*2*f)
    y = np.cos(W*n+phase_off)

    return y




def get_sin(f,phase_off,dur,s):
    n = np.arange(0,dur, 1/s)
    W = (np.pi*2*f)
    y = np.sin(W*n+phase_off)
   
    return y

def complex_exp(f,phase_off,dur,s):
    cos = get_cos(f,phase_off,dur,s)
    sin = get_sin(f,phase_off,dur,s)

    return cos + 1j*sin
#--------------------------------------------------



def interpolate_zeros_in_the_spectrum(x,num_zeros):
    #interpolating zeros at the start and the end of the spectrum
    #TRICK USED : we can do this easily by padding zeros in the cenenter of the spectrum (because we cut the periodic sequence in a nice way)
    #num_zeros better be a divizor of two

    xf = fft(x)
    tmp = xf[:len(xf)//2]    
    tmp = np.concatenate( (tmp,np.zeros(num_zeros)) )
    tmp = np.concatenate( ( tmp, np.flip(tmp)) )    

    return np.real( ifft(tmp) )


#MEL SPECTROGRAM TEST
'''
X_stft = librosa.stft(x)
X_phase = np.angle(X_stft)
spect = librosa.feature.melspectrogram(y=x, sr=s)
mel_spect = librosa.power_to_db(spect, ref=np.max)

S = librosa.feature.inverse.mel_to_stft(spect)

X_recon = create_complex_spectr(S,X_phase)
x_hat = librosa.istft(X_recon)
#x_hat = librosa.griffinlim(S)
'''




def compute_and_plot_fft_1(y,fs):
    yf = fft(y)
    xf = np.linspace(0.0, (fs)/2.0 , len(y)//2)
    plt.figure(figsize=(10,5))

    plt.plot(xf,  np.abs(yf[0:len(y)//2]))


def create_gaussian_kernel(shape=(5,1),sigma=1,typ="1D"):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh

    if typ=="1D":
        h = h.reshape(-1)

    #h = np.exp( -(1/2)*( ( np.arange(-3,4) - 0 )**2 ) /5**2 )/5*np.sqrt(2*np.pi)

    return h

#FITLERING--------------------------------------

#REMEZ-----------------------------------------
def remez(numtaps,fs,cutoff = 32,trans_width = 100):
    #fs = 44100       # Sample rate, Hz

    #cutoff = 32    # Desired cutoff frequency, Hz

    #trans_width = 100  # Width of transition from pass band to stop band, Hz

    #numtaps = 400      # Size of the FIR filter.

    taps = signal.remez(numtaps, [0, cutoff, cutoff + trans_width, 0.5*fs], [1, 0], Hz=fs)

    w, h = signal.freqz(taps, [1], worN=2000)

    h[len(h)//2:] = np.flip(h[:len(h)//2])

    #plot_response(fs, w, h, "Low-pass Filter")
    return h
#-----------------------------------------------------------

def LTI_filtering_toepliz(h,x):
    h_padded = np.concatenate((h,np.zeros(len(x)-len(h))))
    Conv_mtx = scipy.linalg.toeplitz(h_padded,np.zeros(len(h_padded)))


    y = np.dot(Conv_mtx,x)

    return  y 

def get_conv_mtx(h,sig_len):
    h_padded = np.concatenate((h,np.zeros(sig_len-len(h))))
    Conv_mtx = scipy.linalg.toeplitz(h_padded,np.zeros(len(h_padded)))

    return Conv_mtx


def divide_and_conquer_toeplitz(h,x,block_len):
    #BLOCK LEN SHOULD BE POWER OF 2

    #WE DIVIDE THE PROBLEM INT0 M SMALLER PROBLEMS
    x_power2 = PadRight(x)
    M = len(x_power2)//block_len

    x_blokcs = np.reshape(x_power2,[block_len,M],"F")


    Conv_mtx = get_conv_mtx(h,len(x_blokcs.T[0]))
    #out_blocks = np.array( list( map( lambda seg : np.dot(Conv_mtx,seg)   , x_blokcs.T ) ) )
    out_blocks = np.dot(Conv_mtx,x_blokcs)
    
    #out_blocks = np.array( list( map( lambda seg : LTI_filtering_toepliz(h,seg)   , x_blokcs.T ) ) )

    return np.reshape(out_blocks,len(x_power2),"F")
   

def LTI_filtering(filter_kernel,x):
    return signal.convolve(x,filter_kernel,mode='same')

def IIR_filtering(x,coefs=[1,-2.737,3.746,-2.629,0.922]):

    #example
    #model parameteres for the all pole IIR system
    # a = [1,-2.737,3.746,-2.629,0.922]
    # x = signal.lfilter([1],a,noise,0)

    return signal.lfilter([1],coefs,x,0)

def custom_LTI_filtering(filter_kernel,x):
    xf = fft(x,len(x)+len(filter_kernel)-1)
    hf = fft(filter_kernel,len(xf))
    outf = hf*xf

    return np.real(ifft(outf))

def custom_LTI_REAL_filtering(filter_kernel,x):

    #neccesary padding to obtain Linear convolution
    N = len(x)+len(filter_kernel)-1
    xf = rfft(x,N)
    hf = rfft(filter_kernel,N)
    outf = hf*xf

    return np.real(irfft(outf))    

def filtering_using_wind(x,fs,fc):
    #filtering using real valued window

    L=len(x)
    fk = np.arange(L)*fs/L

    inds = fk<=fc
    H = inds*1

    xf = fft(x)
    out = np.real( ifft(xf*H) )

    return out,H

def FIR_filt_design(fs,trans_width,fc,ripple_db):

    # The Nyquist rate of the signal.
    nyq_rate = fs / 2.0

    # The desired width of the transition from pass to stop,
    # relative to the Nyquist rate.  We'll design the filter
    # with a 5 Hz transition width.
    width = trans_width

    # The desired attenuation in the stop band, in dB.
    ripple = ripple_db

    # Compute the order and Kaiser parameter for the FIR filter.
    N, beta = signal.kaiserord(ripple_db, width)

    # The cutoff frequency of the filter.
    cutoff_hz = fc

    # Use firwin with a Kaiser window to create a lowpass FIR filter.
    taps = signal.firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))

    return taps
#------------------------------------------------------------------


#SAMPLERATE CONVERSION------------------------------------------------
def upsample(arr, factor=2):
    n = len(arr)
    return np.interp(np.linspace(0, n, factor*n+1), np.arange(n), arr)

def upsample_v2(x,L):
    #upsample : inserting L zeroes between each sample
    n = len(x)
    out = np.zeros(L*n,dtype=x.dtype)
    out[::L] = x

    return out

def downsample(arr,factor=2):
    return signal.decimate(arr, factor)

def downsample_v2(x,M):
    #downasmple: getting every Mth element starting from 0
    return x[::M]

def get_downsampling_mtx(signal_length,factor):

    D_N = np.eye(signal_length//factor)

    eN = np.zeros(factor)
    eN[0] = 1 
    Downsampling_matrix = np.kron(D_N,eN)

    return Downsampling_matrix



def prefilter_and_downsampling(x,M):

    #DECIMATION FILTER!!!

    #THE FILTERING IS DONE IN THE FREQUENCY DOMAIN
    #we implement an ideal low pass filter with a cuttof frequency depending on the signal to be filtered 
    #this is because to achieve a so steep transition region we have to find the apropriate cutoff (index to start the zeroes) and this varies depending 
    # on the sampling rate  
    
    len_1s = len(x)//(2*M)
    len_0s = len(x)//2 - len_1s
    H = np.concatenate( ( np.ones(len_1s,dtype=complex) , np.zeros(len_0s,dtype=complex)) )
    H = np.concatenate( ( H , np.flip(H)) )

    xf = fft(x)
    outf = xf*H
    x_tmp = np.real(ifft(outf))
    
    x_decimated = downsample_v2(x,M)

    return x_decimated

def interpolation_filter(x,L):

    x_expanded = upsample_v2(x,L)

    len_1s = len(x_expanded)//(2*L)
    len_0s = len(x_expanded)//2 - len_1s
    H = np.concatenate( ( np.ones(len_1s,dtype=complex) , np.zeros(len_0s,dtype=complex)) )
    H = np.concatenate( ( H , np.flip(H)) )

    xf = fft(x_expanded)
    x_interpolated = np.real( ifft( xf*H*L ) )
    

    return x_interpolated

#TEST LEAST SQUARES METHOD FIR DESIGN
def fir(fr_bins,impulse_resp_len,):

    cos = lambda w : np.cos(w*np.arange(impulse_resp_len//2))
    Wn_k = (2*np.pi/fr_bins)*np.arange(fr_bins)
    A = np.array( list( map( lambda Wn : cos(Wn) , Wn_k ) ) )
    # A = np.delete(A,0,0)
    # A = A[:,1:]*2

    #desiered response 
    tmp = np.concatenate((np.ones(100),np.zeros(100)))
    D = np.concatenate((tmp,np.flip(tmp)))
    #D = np.delete( D, len(D)//2 ) 


    A_psuedo = np.linalg.pinv(A)
    h = np.dot(A_psuedo,D)

    h_full = np.concatenate((h,np.flip(h)))

    return h_full

#---------------------------------------------------------------------------------



def sound_write(sound,s,path='/home/nnanos/Desktop/sounds/test.wav'):
    sf.write(path,sound,s)

def create_complex_spectr(X_mag,X_phase):
    #creates a complex spectrogram givven the magnitude spectrogram and the phase
    tmp = []
    for rho , phi in zip(X_mag , X_phase):
        tmp .append( list(map(lambda x, y: np.complex(x*np.cos(y),x*np.sin(y)) , rho, phi ))  )    

    return np.array(tmp)

def create_spectrum(Amp,phase):
    #creates a complex spectrum givven the magnitude response and the phase response
    return np.array( list(map(lambda x, y: np.complex(x*np.cos(y),x*np.sin(y)) , Amp, phase )) )

    

def estimate_autocor_seq(x):
    #estimating the autocorellation sequence assuming a mean ergodic random process x(n)---------------------------------------------------------

    #autocorellation (deterministic) can be obtained by convolving x(n) and x(-n) and dividing by N
    flipped_x = np.flip(x)

    #normalizing the signals to have energy 1 before convolving
    #flipped_x = flipped_x/np.sqrt((np.dot(flipped_x,flipped_x)/len(flipped_x)))
    #x = x/np.sqrt(np.dot(x,x)/len(x)) 

    var_est = np.std(x)**2
    rx_hat_conv_case = (1/(len(x)*var_est))*(signal.convolve(x,flipped_x,mode='same'))

    return rx_hat_conv_case

def estimate_crosscor_seq(x,y):
    #estimating the autocorellation sequence assuming a mean ergodic random process x(n)---------------------------------------------------------

    #autocorellation (deterministic) can be obtained by convolving x(n) and x(-n) and dividing by N
    flipped_y = np.flip(y)

    #normalizing the signals to have energy 1 before convolving
    #flipped_x = flipped_x/np.sqrt((np.dot(flipped_x,flipped_x)/len(flipped_x)))
    #x = x/np.sqrt(np.dot(x,x)/len(x)) 

    std_est_x = np.std(x)
    std_est_y = np.std(y)
    rx_hat_conv_case = (1/(len(x)*std_est_x*std_est_y))*(signal.convolve(x,flipped_y,mode='same'))

    return rx_hat_conv_case

def estimate_PSD(x):
    #obtaining the PSD by the periodogram 
    normalized_freq, Pxx_den = signal.periodogram(x)

    return normalized_freq, Pxx_den

def periodic_extension(x,nb_periods,win_type,zero_padd):

    silent_region = np.zeros(zero_padd)
    w = create_desiered_window(win_type,len(x)) 
    x_per = np.concatenate((w*x,silent_region,w*x))
    for i in range(nb_periods):
        x_per = np.concatenate((x_per,silent_region,w*x))
    
    return x_per

def triangle(length, amplitude):
     section = length // 4
     for direction in (1, -1):
         for i in range(section):
             yield i * (amplitude / section) * direction
         for i in range(section):
             yield (amplitude - (i * (amplitude / section))) * direction

def translation_operator(x,n):
    #using the modulation property of the fourier transform

    #n negative -> translation right
    #n positive -> translation left 

    xf = fft(x)
    comp_exp = complex_exp(n,0,1,len(x))
    tmp = ifft( comp_exp*xf )

    #w = create_desiered_window("Blackman",4096)
    #np.pad(w, (1024,1024), 'constant', constant_values=(0, 0))
    #w_pad = np.concatenate((w,np.zeros(len(x)-len(w))))
    #periodization to length M = L/b :  


    return tmp



def estimate_f0_of_a_note(amp_response,fs):
    
    N = len(amp_response)

    #frequency sampling whith freqency increments m/N*Ts
    m = np.arange(0,N-1 )
    freq_incr = fs/N
    m = m*freq_incr

    #up to nyiquist frequancy
    m = m[:N//2]    
    amplitude = amp_response[:N//2]

    ind = np.argmax( amplitude )

    max_energy_bin = m[ind]

    return max_energy_bin

def estimate_f0_of_a_note_autocor(x):

    Rx = estimate_autocor_seq(x)
    Rx = Rx[len(Rx)//2:]
    Rx.max()

    
def change_pitch_of_note(x,f,fs):

    xf = fft(x)
    #xf = np.fft.fftshift(fft(x))
    phase_response = np.angle(xf)
    amp_response = np.abs( xf )
    max_freq_bin = estimate_f0_of_a_note(amp_response,fs)

    #find the apropriate frequency for the carier
    #m = len(x)*(1/fs)*max_freq_bin #m is the ind of the max freq bin

    true_carier_freq = f-max_freq_bin
    #cos = get_cos(true_carier_freq,0,len(x)/fs,fs)
    #modulating based on the fourier property 
    cos = complex_exp(true_carier_freq,0,len(x)/fs,fs)
    #plt.plot(np.arange(0,len(out_f))*(fs/len(out_f)),np.abs(out_f))
    out_f = fft(cos*x)
    '''
    #modulating based on the impulse response delay trick
    h = np.zeros(44100)
    h[len(h)-1] = 1
    #delay the spectrum
    amp_response_out = LTI_filtering(h,np.abs(fft(x)))
    amp_response_out = amp_response_out[:(len(amp_response_out)//2)-1]
    amp_response_out = np.concatenate( ( amp_response_out , np.array([0]) ,np.flip( amp_response_out )[:len(amp_response_out)-1] ) )
    #plt.plot(np.arange(0,len(amp_response_out)/fs , 1/fs) ,amp_response_out)
    plt.plot(np.arange(0,len(amp_response_out)  ) * (fs/len(amp_response_out)) ,amp_response_out)
    '''

    #phase_response = np.angle(out_f)
    amp_response_out = np.abs(out_f)
    

    #zeroing the symmetric (even) copy that have been created from the modulation
    m = len(x)*(1/fs)*true_carier_freq
    amp_response_out[:int(np.floor(m))] = 0
    #coppying the true spectrum flipped in order to reconstruct----------------------------
    #amp_response_out[len(amp_response_out)//2:] = np.flip( amp_response_out[:len(amp_response_out)//2] )
    amp_response_out = amp_response_out[:(len(amp_response_out)//2)-1]
    amp_response_out = np.concatenate( ( amp_response_out , np.array([0]) ,np.flip( amp_response_out )[:len(amp_response_out)-1] ) ) 

    result = create_spectrum(amp_response_out,phase_response)

    return np.real(result)

def change_pitch_of_note1(x,f,fs):

    xf = fft(x)
    amp_response = np.abs( xf )
    max_freq_bin = estimate_f0_of_a_note(amp_response,fs)

    true_carier_freq = f-max_freq_bin
    #cos = get_cos(true_carier_freq,0,len(x)/fs,fs)
    cos = complex_exp(true_carier_freq,0,len(x)/fs,fs)
    out_f = fft(cos*x)

    return out_f

def chenge_pitch_by_cqt(x,num_semitones,s):

    ksi_min = 16.35
    ksi_max = 7902.13
    B = 12
    nsgt = instantiate_NSGT( x , s , "oct", ksi_min , ksi_max , B )
    X = NSGT_forword(x,nsgt,pyramid_lvl = 0)
    out = NSGT_backward(np.roll(X,num_semitones,axis=0),nsgt,pyramid_lvl = 0)

    return out


def get_frequency_notes(fmin):
    #get frequency notes equal tempered scale (tunning A4=440)
    k = np.arange(-50,41)
    F = fmin*2**(k/12)
    return F

def estimate_fundamental_if_it_exists(x,s):
    
    #convert to freq
    #(2**(37/12))*32.7

    nsgt = instantiate_NSGT( x , s , "oct", 32.7 , s//2-1 , 12 )
    X = NSGT_forword(x,nsgt,pyramid_lvl = 6)
    a = list(map(lambda x: np.argmax(np.abs(x)),X.T))
    #plt.hist(a)

    tmp0,tmp1 = np.histogram(a,X.shape[0])
    f0_ind = tmp1[np.argmax(tmp0)-1]

    return f0_ind

def periodization(x,M):
    b = len(x)//M

    #for i in range(1,b):
        #x = x/np.sqrt(b) + np.roll(x, i*M)
        #x = x/np.sqrt(b)
        #x += np.real(translation_operator(x, i*M))



    #IN ORDER TO OBTAIN A FOURIER TRANSFORM OF LENGTH M you should do fft(x[:M]/np.sqrt(b))

    return x

def periodization_v2(x,M):



    #segmenting the x into blocks of M and adding them
    S = np.reshape(x, (-1, int(M))).T

    return S.sum(1)

def periodization_v3(x,b):
    M = len(x)//b

    for i in range(0,b):

        #tmp_translation = np.roll(x, i*M)
        #tmp_translation = translation_operator(x, i*M)

        if ( ( np.abs(tmp_translation[:M-1]) > 0 )*1).sum() :
            return tmp_translation[:M-1] , i




def create_chord():
    chord = ['D3','F3','A3']
    chord = librosa.note_to_hz(chord)

    dur = 2
    num_samples = s*dur
    ch = np.zeros(num_samples)
    for f in chord:
        ch += get_cos(f,0,dur,s)

    return ch


if __name__ =='__main__':
    #load music
    x,s = load_music()
    # x = x.astype(np.float64)

    #IF IT is odd then check if its even then we have to do smthing
    h = FIR_filt_design(44100,0.003,200.0,60.0)
    out1=custom_LTI_REAL_filtering(h,x[:len(x)-1])
    out2 = custom_LTI_filtering(h,x[:len(x)-1])
    norm = lambda x: np.sqrt(np.sum(np.abs(np.square(x))))
    print(norm(out2-out1)/norm(out2))

    #even case:
    hf = fft(h)
    #output of rfft:
    hf_half = hf[:len(hf)//2+1]
    #this is what irfft calculates given the hf_half:
    hf_rec = np.concatenate((hf_half,np.flip(np.conj(hf_half[1:len(hf_half)-1]))))



    h_odd = h[:len(h)-1]
    hf_odd = fft(h_odd)
    hf_odd_half = hf_odd[:len(hf_odd)//2+1]
    hf_odd_rec = np.concatenate((hf_odd_half,np.flip(np.conj(hf_odd_half[1:len(hf_odd_half)-1]))))






    # def irfft_custom(xf_half,flag):
        
    #     #N = len(x)
        
    #     #checking if its odd
    #     if flag:
    #         #odd case:
    #         xf_rec = np.concatenate((xf_half,np.flip(np.conj(xf_half[1:len(xf_half)-1]))))
    #         x_rec = np.real(ifft(xf_rec))

    #     else:
    #         #even case:
    #         #this is what irfft calculates given the xf_half:
    #         xf_rec = np.concatenate((xf_half,np.flip(np.conj(xf_half[1:len(xf_half)-1]))))
    #         x_rec = np.real(ifft(xf_rec))

    #     return x_rec    

    # def rfft_custom(x):
        
    #     N = len(x)
        
    #     #checking if its odd
    #     if N%2:
    #         #odd case:
    #         flag=1
    #         #converting to even
    #         x_even = x[:N-1]
    #         xf_even = fft(x_even)
    #         xf_half = xf_even[:N//2+1]
            

    #     else:
    #         #even case (painless):
    #         flag=0

    #         xf = fft(x)

    #         xf_half = xf[:N//2+1]

    #     return xf_half,flag   
        
    def rfft_custom(x):
        N = len(x)
        N_new = 1 + N//2 
        xf_half = fft(x)[:N_new]

        return xf_half                   

    def irfft_custom(xf_half):
        
        xf_full =  np.concatenate((xf_half,np.flip(np.conj(xf_half[1:len(xf_half)-1]))))

        return np.real(ifft(xf_full))

        

    def rfft_custom(x):
        N = len(x)
        N_new = 1 + N//2 
        xf_half = fft(x,N_new)

        return xf_half                   

    def irfft_custom(xf_half):
        
        xf_full =  np.concatenate((xf_half,np.flip(np.conj(xf_half[1:len(xf_half)-1]))))

        return np.real(ifft(xf_full))



    xf = rfft_custom(x[:len(x)-1])
    x_rec = irfft_custom(xf)

    xf1 = rfft(x[:len(x)-1])
    x_rec1 = irfft(xf1)


    print(norm(x_rec1-x_rec)/norm(x_rec1))

    xf = rfft_custom(x)
    x_rec = irfft_custom(xf)

    xf1 = rfft(x)
    x_rec1 = irfft(xf1)

    print(norm(x_rec1-x_rec)/norm(x_rec1))      



    a = 0


def get_filter_bank():
    F = get_frequency_notes()

    for i in range(len(F)):
        w = create_desiered_window()
        cos = complex_exp(16,0,len(w)/1024,1024)


def ALIAS_L(x,L):

    '''
    if len(x)mod(2):
        x = np.concatenate((x,0))

    x.reshape((,L))
    '''
    M = len(x)/L
    mod = len(x)%M
    S = []
    if mod :
        x = x[:len(x)-mod]
        #getting the segments of the signal (the columns contain the segments of nfft size )
        S = np.reshape(x, (-1, M)).T

    S = np.reshape(x, (-1, int(M))).T

    return S

