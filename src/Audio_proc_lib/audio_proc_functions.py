import soundfile as sf 
import librosa
import librosa.display
import numpy as np
from scipy.linalg import toeplitz
#from scipy.fftpack import fft,ifft
from numpy.fft import fft,ifft
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import youtube_dl
#from nsgt import NSGT, LogScale, LinScale, MelScale, OctScale
from Plotting_funcs.Plotting_util_and_other import *
import time




def load_music():

    ytbe_flag = int(input('Press 2 if you want to load song segment of youtube\n Press 1 if you want to load a song segment of your pc\n: ') )

    #GETTING TRACK FROM YOUTUBE-------------------------------------
    def my_hook(d):
        if d['status'] == 'finished':
            print('Done downloading...')

    if ytbe_flag==2:

        url = input("Give the url of the song in youtube:") #@param {type:"string"}
        start = int(input("Give the start sec:")) #@param {type:"number"}
        stop = int(input("Give the stop sec:")) #@param {type:"number"}

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

        s = input("Give the desired samplerate (recommended 11025 for faster processing) :")

        audio, s = librosa.load(info.get('title', None) + '.wav', sr=int(s), mono=True)

    elif ytbe_flag==1:
        song_path = input("Give the path of the .wav file that you want to create the pyramids for : ")
        s = input("Give the desired samplerate (recommended 11025 for faster processing) :")  
        start = int(input("Give the start sec:")) #@param {type:"number"}
        stop = int(input("Give the stop sec:")) #@param {type:"number"}
        audio , s = librosa.load(song_path,sr=int(s))

    
    # else:
    # #    audio,s = librosa.load('/home/nnanos/Desktop/ThinkDSP-master-20200928T154642Z-001/musicradar-bass-guitar-samples/Bass/bass_tones_long&tremolo/long002.wav',sr=44100) 
    #     url = "https://www.youtube.com/watch?v=qY2WHqvRlFY"
    #     ydl_opts = {
    #         'format': 'bestaudio/best',
    #         'postprocessors': [{
    #             'key': 'FFmpegExtractAudio',
    #             'preferredcodec': 'wav',
    #             'preferredquality': '44100',
    #         }],
    #         'outtmpl': '%(title)s.wav',
    #         'progress_hooks': [my_hook],
    #     }
    #     with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    #         info = ydl.extract_info(url, download=False)


    if ytbe_flag:
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

def filtering_using_wind(x,fs,fc):
    #filtering using real valued window

    L=len(x)
    fk = np.arange(L)*fs/L

    inds = fk<=fc
    H = inds*1

    xf = fft(x)
    out = np.real( ifft(xf*H) )

    return out,H


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

    nsgt = instantiate_NSGT( x , s , "oct", 190 , s//2 , 12 )
    #reconstruction error with upasmpling---------------------------
    X = NSGT_forword(x,nsgt,pyramid_lvl = 3)


    S = ALIAS_L(np.zeros(6),2)

    tmp = get_vocals_drums_harmonic_hpss(x,s)

    #sound_write(chenge_pitch_by_cqt(x,-1,s),s)

    #mtp = librosa_features_extract('/home/nnanos/Desktop/ThinkDSP-master-20200928T154642Z-001/ThinkDSP-master/code/audio_processing/ISA/J Dilla - BillyBrooksBaatin.wav')

    tmp = HPSS_ALG_orig(x,44100,1024,256,filter_len_per=31,filter_len_harm=31)

    (2**(37/12))*32.7
    nsgt = instantiate_NSGT( x , s , "oct", 32.7 , s//2-1 , 12 )
    X = NSGT_forword(x,nsgt,pyramid_lvl = 0)
    a = list(map(lambda x: np.argmax(np.abs(x)),X.T))
    #plt.hist(a)

    tmp0,tmp1 = np.histogram(a)
    
    # s = 1000
    # x = get_cos(124,0,1,s)

    #x,s = librosa.load('/home/nnanos/Downloads/334538__teddy-frost__c4(1).wav',sr=44100)
    #x,s = librosa.load('/home/nnanos/Desktop/ThinkDSP-master-20200928T154642Z-001/ThinkDSP-master/code/low-funky-bass-line_96bpm.wav',sr=44100)  
    #change_pitch_of_note(x,440,s)
    

    norm = lambda x: np.sqrt(np.sum(np.abs(np.square(x))))

    #compare with librosa CQT----------------------------------
    X1 = librosa.cqt(y=x, sr=s, hop_length=64, n_bins=7*12,bins_per_octave=12)
    shape_lib = X1.shape
    s_r_lib = librosa.icqt(C=X1, sr=s, hop_length=64,bins_per_octave=12)
    rec_err_lib = norm(x[:len(s_r_lib)]-s_r_lib)/norm(x)
    #---------------------------------------------------------

    nsgt = instantiate_NSGT( x , s , "oct", 190 , s//2 , 12 )
    #reconstruction error with upasmpling---------------------------
    X = NSGT_forword(x,nsgt,pyramid_lvl = 3)
    shape_sub = X.shape
    #h = create_gaussian_kernel(sigma=1)
    #s_r = LTI_filtering(h,NSGT_backward(X,nsgt))
    s_r_sub = NSGT_backward(X,nsgt,pyramid_lvl = 3)
    rec_err_sub = norm(x-s_r_sub)/norm(x)
    #--------------------------------------------------------------

    #comparing the reconstruction error with full (without subsampling)-------
    c_full = np.array( nsgt.forward(x) )
    shape_full = c_full.shape
    #reconstruct
    s_r_full = nsgt.backward(c_full)
    rec_err_full = norm(x-s_r_full)/norm(x)
    #--------------------------------------------------

    print("Reconstruction error full: %.3e \t shape: %s \nReconstruction error pyramid: %.3e \t shape: %s \nReconstruction error librosa: %.3e \t shape: %s "%(rec_err_full,shape_full,rec_err_sub,shape_sub,rec_err_lib,shape_lib))

    change_pitch_of_note1(x,240,s)
    
    X,_ = get_spectrogram_custom1(x,4096,1024)
    y_recon = get_inverse_stft1(X,1024)

    y = get_cos(100,1,4096)
    D = get_dft_mtx(4096)


    ksi_min = 16.35
    ksi_max = 7902.13
    B = 12
    M = np.ceil(12*np.log2(ksi_max/ksi_min)+1)
    m = np.arange(M)+1
    fr_range = ksi_min*(2**((m-1)/B))

    nsgt = instantiate_NSGT( x , s , "oct", ksi_min , ksi_max , B )
    X = NSGT_forword(x,nsgt,pyramid_lvl = 0)

    plt.imshow(librosa.amplitude_to_db(np.abs(X)),aspect="auto",origin="lower",extent=[0,X.shape[1],ksi_min,ksi_max])



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


'''
x1,s = librosa.load('/home/nnanos/Desktop/ThinkDSP-master-20200928T154642Z-001/ThinkDSP-master/code/low-funky-bass-line_96bpm.wav',sr=44100) 
nsgt = instantiate_NSGT( x , s , "oct", 30 , s//2 , 12 )
X = NSGT_forword(x,nsgt,pyramid_lvl = 0)
sound_write(NSGT_backward(np.roll(X1,-1,axis=0),nsgt,pyramid_lvl = 0,s)
,s)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(np.roll(X1,20,axis=0)), ref=np.max),
                        sr=s, x_axis='time', y_axis='cqt_hz')
plt.colorbar(format='%+2.0f dB')
plt.title('Constant-Q power spectrum')
plt.tight_layout()



    tmp1 = np.eye(N_samples//2)
    tmp2 = np.zeros((N_samples//2,N_samples//2))

    D_N = np.concatenate((tmp1,tmp2) , axis=0)
    e2 = np.array([1,0])
    Downsampling_matrix = np.kron(D_N,e2)   
    D_N = np.concatenate((tmp1,tmp2) , axis=0)
    e2 = np.array([1,0])
    Downsampling_matrix = np.kron(D_N,e2)
'''

'''
basis, lengths = librosa.filters.constant_q(22050, filter_scale=0.5)
for filt in basis:
    #plot_magnitude_fft(fft(filt),0,22050)
    plt.semilogx(np.abs(fft(filt)[:(len(filt)//2)]))

basis, lengths = librosa.filters.constant_q(44100, filter_scale=0.5)
bas_tmp = basis[:12]
for filt in bas_tmp:
    #plot_magnitude_fft(fft(filt),0,22050)
    plt.semilogx(np.abs(fft(filt)[:(len(filt)//2)]))


next((i for i, x in enumerate(basis[50,:]) if x),None)
C = np.array( list( map(lambda filt: LTI_filtering(filt,x),basis) ) )

'''

'''
nsgt = instantiate_NSGT( x , s , 'oct',32.7,s//2,12,multithreading=False)
C_nsgt = NSGT_forword(x,nsgt)

#tmp contains a list of (cA,cD) (for each row freq bin)
tmp = list( map( lambda row: pywt.dwt(row, 'db2') , C_nsgt) )

#subsampled_C contains the subsampled (rowwise) CQT NSGT (each row is a cA)
subsampled_C = np.array( list( map( lambda x: x[0] , tmp ) ) )

#we can reconstruct with only the subsampled_C (the only artifact is some high frequency)
#h = create_gaussian_kernel(sigma=1)
#x_recon = LTI_filtering(h,NSGT_backward(subsampled_C,nsgt))


high_freq_C = np.array( list( map( lambda x: x[1] , tmp ) ) )

#reconstruction needs a list (tmp) of (cA_processed,cD) (for each row freq bin)
tmp_recon = np.array( list( map( lambda coefs: pywt.idwt(coefs[0],coefs[1], 'db2') , tmp ) ) )
x_recon = NSGT_backward(tmp_recon,nsgt)




xf = fft(xf)
xf_pad = np.concatenate((np.zeros(len(hf)//2),xf,np.zeros((len(hf)+1)//2)))
h = create_gaussian_kernel((50,1),50)


h = signal.firwin(11, 100,fs=44100)
hf = fft(h)
hf_pad = np.concatenate( (np.zeros( (len(xf)-len(hf))//2 ),hf, np.zeros( (len(xf)-len(hf))//2+1) ) ) 
hf_pad = np.concatenate( (np.zeros( (len(xf))//2 ),hf, np.zeros( (len(xf))//2+1) ) ) 
outf = hf_pad*xf
sound_write(np.real(ifft(outf)),s)


'''
