import os
import argparse
import psrchive
import numpy as np
from scipy.fftpack import fft
import scipy.signal as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
from scipy.stats import norm
 

########################
# LOAD PSRCHIVE Utils  #
######################## 
def load_psrchive(fname):
    archive = psrchive.Archive_load(fname)
    archive.set_dedispersed(False)
    archive.update()
    archive.pscrunch()
    archive.tscrunch()
    archive.remove_baseline()
    w = archive.get_weights().squeeze()
    return archive, w

###############################################
# Get Waterfall Plot from archive @ Single DM #
###############################################
def get_waterfall(archive, DM, w=None):
    archive.set_dispersion_measure(DM)
    archive.dedisperse()
    waterfall = archive.get_data().squeeze()
    if w is not None: waterfall *= w[:, np.newaxis]
    return waterfall

###################################
# Get Coherent Spect of Waterfall #
###################################
def get_Spect(waterfall):
    FT = fft(waterfall)
    amp = np.abs(FT)
    amp[amp == 0] = 1
    spect = np.sum(FT / amp, axis=0)
    return spect

###################################
# Get Coherent Power of Waterfall #
###################################
def get_Pow(waterfall):
    spect = get_Spect(waterfall)
    Pow = np.abs(spect)**2
    return Pow



###################################################
# Cal best F.freq cutoff base on (peak error)^-2 #
###################################################
def get_fact(Pow_list,MEAN,STD):
    s   = np.max(Pow_list,axis=1)
    #SN_r= float(norm.ppf(1-1/float(np.size(Pow_list))))
    SN  = (s-MEAN)/STD 
    Kern= np.round( Window(SN*(SN>5))/2)
    if Kern < 5:
       Kern = 5
    return Kern

###################
# Turing Point SN #
###################
def Tp(series):
  Ntp=np.float(np.sum(np.absolute(np.diff(np.sign(np.diff(series))))==2));
  N=np.float(np.size(series));
  sig=np.sqrt((16*N-29)/90);
  E=2*(N-2)/3;
  z=(Ntp-E)/sig;
  return z
    
##################
# Polynomial Fit #
##################
def Poly_Max(x,y,Err):
  m   = np.float(np.size(y))
  Ntp = np.float(np.sum(np.absolute(np.diff(np.sign(np.diff(y))))==2))
  Pows= np.arange(2,m-1)
  Met = []
  SN  = []
  for j in Pows:
    fit = np.polyfit(x,y,j,full="False")
    Res = y-np.polyval(fit[0],x)
    SN.append(np.absolute(Tp(Res)))    
    Met.append(np.std(Res))
  Val = np.extract(SN==np.min(SN),Met)
  loc = np.where(Met==np.min(Val))[0]
  n   = loc[0]+2
  fit = np.polyfit(x,y,n,full="False")
  p   = fit[0]  
  #Err = np.std(y-np.polyval(p,x))
  Fac = np.std(y)/Err
  #Err = Err*(1+1/np.sqrt(len(x)))
  
  dp      = np.polyder(p)
  ddp     = np.polyder(dp)
  cands   = np.roots(dp)
  r_cands = np.polyval(ddp,cands)
  Indx    = np.where( (cands.imag==0) & (cands.real>=min(x)) & (cands.real<=max(x)) & (r_cands<0) )[0]
  if len(Indx)>0:
    first_cut = cands[Indx]
    Value     = np.polyval(p,first_cut)
    Best      = first_cut[np.where(Value==np.max(Value))]
    delta_x   = np.sqrt(np.absolute(2*Err/np.polyval(ddp,Best)))
  else:
    Best    = 0
    delta_x = 0

  return (np.real(Best), delta_x, p , Fac)

###########################################################
#              Coherent Power vs DM Plotting              #
###########################################################
def Graphics(DM_Map,X,Y,Range,Returns_Poly,x,y,SN,filename):
  ar = psrchive.Archive_load(filename)
  DM_len=DM_Map.shape[1]
  FT_len=DM_Map.shape[0]
  Bin_len=ar.get_nbin()
  #Buff = np.int(np.round(DM_len/6))+1
  Buff=3
  fig = plt.figure(num=None, figsize=(6,8.5),facecolor='black')
  grid = plt.GridSpec(16, 1, hspace=0.0, wspace=0.0)
  main_ax = fig.add_subplot(grid[:-1, 1:])
  plt.ioff()
    
  plt.subplot(grid[0:4,0])
  plt.plot(X,Y,'-w',linewidth=2)
  plt.plot(X[Range],np.polyval(Returns_Poly[2],X[Range]),color='0.5',linewidth=3)
  Title='{0:} \n Best DM = {1:.3f} $\pm$ {2:.3f}\n SN = {3:.1f}'.format(os.path.basename(filename), Returns_Poly[0][0], Returns_Poly[1][0],SN)
  plt.suptitle(Title,color='w')  
  plt.axis([np.min(X), np.max(X), np.min(Y[Buff:-Buff-1]), np.max(Y[Buff:-Buff-1])])
  plt.axis('off')

  axr=plt.subplot(grid[4,0],axisbg='k')
  Res = y-np.polyval(Returns_Poly[2],x)
  plt.plot(x,Res,'xw',linewidth=2)
  plt.axis([np.min(X), np.max(X), np.min(Res)-np.std(Res)/2, np.max(Res)+np.std(Res)/2])
  plt.ylabel('$\Delta$')
  axr.yaxis.label.set_color('w')
  axr.tick_params(axis='x', colors='w',labelbottom='off')
  axr.tick_params(axis='y', colors='w',labelleft='off')
    
  ax=plt.subplot(grid[5:,0])
  dt=ar.get_first_Integration().get_duration()
  plt.imshow(DM_Map,origin='lower',aspect='auto', \
             cmap=cm.YlOrBr_r,extent=[np.min(X),np.max(X),0,2*np.pi*FT_len/Bin_len/dt])
  ax.tick_params(axis='x', colors='w')
  ax.tick_params(axis='y', colors='w')
  plt.xlabel('DM')
  plt.ylabel('w (Rad / ms)')
  ax.xaxis.label.set_color('w')
  ax.yaxis.label.set_color('w')
  return fig 

##################
# ACF Windowing  #
##################
def Window(Pro):
  arr=sp.detrend(Pro)
  X=np.correlate(arr,arr,"same")
  n=np.where(X==np.max(X))[0]
  X[n]=(X[n-1]+X[n+1])/2
  W=np.max(np.diff(np.where(X<0)))
  
  return W

def W_Check(Pro,W):
  SM=np.convolve(Pro,np.ones(W),'same')
  Peak = np.mean(np.where(SM==max(SM)))
  Max= np.where(Pro==np.max(Pro))
  if (Peak-Max)**2>W**2:
    W=W+np.absolute(Peak-Max)/2
    Peak=(Peak+Max)/2
  Start=np.int(Peak-np.round(1.25*W))
  End=np.int(Peak+np.round(1.25*W))
  if Start<0:
    Start=0
  if End>np.size(Pro)-1:
    End=np.size(Pro)-1
  
  return Start,End

####################################
#      Dynamic Spec. Plotting      #
#################################### 
def Graphics1(Returns_Poly,fname,archive,w,Cut_off):

  DMs=Returns_Poly[0][0]+5*Returns_Poly[1][0]*np.array([-1,0,1])
  fig = plt.figure(num=None, figsize=(8.5,6),facecolor='black')
  grid = plt.GridSpec(8, 3, hspace=0.0, wspace=0.2)
  main_ax = fig.add_subplot(grid[:-1, 1:])
  plt.ioff()
  Title='{0:} \n Best DM = {1:.3f} $\pm$ {2:.3f}'.format(os.path.basename(fname), Returns_Poly[0][0], Returns_Poly[1][0])
  plt.suptitle(Title,color='w')
  
  
  for j,dm in enumerate(DMs):
    DS=get_waterfall(archive,dm,w)
    
    plt.subplot(grid[1,j])
    subtitle='{0:.3f}'.format(dm)
    Nty=1
    pro=np.swapaxes(np.mean(DS[Nty:-Nty,:],0),0,1)
    pro=np.ravel(pro)
    
    DS_sig=np.std(DS[Nty:-Nty,:])
    #Clip=np.where(DS>6*DS_sig)
    #DS[Clip]=np.mean(DS)+6*DS_sig
    MAX_DS = np.max(DS[Nty:-Nty,:])
    MIN_DS = np.mean(DS[Nty:-Nty,:])-DS_sig
    
    if j==0: 
      W=Window(pro)
    Spect = get_Spect(get_waterfall(archive,dm,w))
    Filter= np.ones(np.shape(Spect))
    Filter[Cut_off:-Cut_off]=0
    Spike = np.real(np.fft.ifft(Spect*Filter))
    Spike[0]=0
    Win=W_Check(Spike,W)

    plt.plot(np.arange(Win[0],Win[1]),pro[Win[0]:Win[1]],'w')
    plt.axis('tight')
    plt.axis('off')
    plt.title(subtitle,color='w')
    dt=archive.get_first_Integration().get_duration()
    ax=plt.subplot(grid[2:,j])
    chans=archive.get_nchan()
    ch=np.arange(0,chans)
    
    freq=[archive.get_first_Integration().get_centre_frequency(i) for i in range(archive.get_nchan())]
    df=np.mean(np.diff(freq))
    if df<0:
      im=np.flipud(DS[:,Win[0]:Win[1]])
    else:
      im=DS[:,Win[0]:Win[1]]
    plt.imshow(im,origin='lower',aspect='auto', vmin=MIN_DS, vmax=MAX_DS,\
               cmap=cm.YlOrBr_r,extent=[0,dt*(Win[1]-Win[0]),np.min(freq),np.max(freq)])
    ax.xaxis.set_ticks(np.arange(0,dt*(Win[1]-Win[0]), 0.5*np.ceil(2*dt*(Win[1]-Win[0])/5)))
    if j==1:
      plt.xlabel('Time (ms)')
      ax.xaxis.label.set_color('w')
      SN=(chans**0.5)*(np.max(pro)-np.mean(DS))/DS_sig
   
    if j>0:
      plt.tick_params(axis='both', which='both', bottom='on',\
       top='off', labelbottom='on', right='off',\
       left='off', labelleft='off',colors='w')
    else:
      plt.ylabel('Freq. (MHz)')
      plt.tick_params(axis='both', which='both', bottom='on',\
       top='off', labelbottom='on', right='off',\
       left='on', labelleft='on',colors='w')
      ax.yaxis.label.set_color('w')
     
  return fig,SN

#####################################
#           Main Work Horse         #
#####################################
def main(fname, DM_s, DM_e, DM_step):
    archive, w = load_psrchive(fname)
    DM_list = np.arange(np.float(DM_s), np.float(DM_e), np.float(DM_step))
    Pow_list = np.zeros([DM_list.size, archive.get_nbin()])
    for i, DM in enumerate(DM_list):
        waterfall = get_waterfall(archive, DM, w=w)
        #if len(waterfall.shape) != 2: raise 
        Pow = get_Pow(waterfall)
        Pow_list[i] = Pow
    Pow_list = Pow_list.T
    
    
    xv, yv = np.meshgrid(np.arange(0,archive.get_nbin()/2,1),DM_list, sparse=False, indexing='ij')
    dPow_list=Pow_list[:archive.get_nbin()/2]*xv**2
    #DM_curve = Pow_list[:fact_idx+1].sum(axis=0)
    
    Mean = archive.get_nchan() # Base on Gamma(2,)
    STD  = archive.get_nchan()/np.sqrt(2) # Base on Gamma(2,)
    fact_idx = np.round(get_fact(Pow_list[:archive.get_nbin()/2],Mean,STD))
    DM_curve =  (dPow_list[:fact_idx]).sum(axis=0)

    Max   = np.max(DM_curve)
    dMean = (fact_idx*(fact_idx+1)*(2*fact_idx+1)/6)*Mean
    dSTD = STD * np.sqrt(fact_idx*(fact_idx+1)*(2*fact_idx+1)*(3*fact_idx**2 + 3*fact_idx-1)/30)
    SN   = (Max-dMean)/dSTD
     
    Thres= (Max+np.mean(DM_curve))/2
    Top  = np.where(DM_curve>Thres)
    Peak = np.where(DM_curve==Max)
    St   = np.min(Top)
    Ed   = np.max(Top)
    #Range= np.arange(St,Ed)
    
    Range= np.arange(Peak[0]-2,Peak[0]+2)
    y=DM_curve[Range]
    x=DM_list[Range]
    Returns_Poly=Poly_Max(x,y,dSTD)
    
    
    max_idx = np.round(fact_idx) 
    Fig1=Graphics(Pow_list[:max_idx],DM_list,DM_curve,Range,Returns_Poly,x,y,SN,fname)
    (Fig2,SN)=Graphics1(Returns_Poly,fname,archive,w,fact_idx)

    
    Fig1.savefig(os.path.basename(fname)+"_DM_Search.eps", dpi=None, facecolor='k', edgecolor='k',
        orientation='portrait', papertype='a4', format='eps',
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None)
    
    Fig2.savefig(os.path.basename(fname)+"_DS_5sig.eps", dpi=None, facecolor='k', edgecolor='k',
        orientation='landscape', papertype='a4', format='eps',
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None)

    return DM_list, Pow_list, DM_curve, fact_idx


def plot(DM_list, Pow_list, DM_curve, fact_idx):
    
    plt.imshow(np.log(Pow_list[:fact_idx]), cmap=cm.YlOrBr_r, origin="lower", aspect='auto', interpolation='nearest', 
               extent=[DM_list[0], DM_list[-1], 0, fact_idx])
    plt.show()
    plt.plot(DM_list, DM_curve)
    plt.show()
    return


def parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Search for best DM based on FFT phase angles.")
    parser.add_argument('fname', help="Filename of the PSRCHIVE file.")
    parser.add_argument('DM_s', help="Start DM.")
    parser.add_argument('DM_e', help="End DM.")
    parser.add_argument('DM_step', help="Step DM.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parser()
    DM_list, Pow_list, DM_curve, fact_idx = main(args.fname, args.DM_s, args.DM_e, args.DM_step)
    #plot(DM_list, Pow_list, DM_curve, fact_idx)

