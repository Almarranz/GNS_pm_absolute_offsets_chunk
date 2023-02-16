#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 12:40:08 2023

@author: amartinez
"""

# Compares GNS´ pm with Hosek propermotions

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
import sys
from astropy.table import Table
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler
from astropy import stats
import pandas as pd
import time
from astropy.time import Time
from astropy.stats import sigma_clip
import astropy.coordinates as ap_coor
from matplotlib.ticker import FormatStrFormatter
from astropy.stats import sigma_clip
from astropy.stats import sigma_clipped_stats
from matplotlib.ticker import FormatStrFormatter

# %%plotting parametres
from matplotlib import rc
from matplotlib import rcParams
rcParams.update({'xtick.major.pad': '7.0'})
rcParams.update({'xtick.major.size': '7.5'})
rcParams.update({'xtick.major.width': '1.5'})
rcParams.update({'xtick.minor.pad': '7.0'})
rcParams.update({'xtick.minor.size': '3.5'})
rcParams.update({'xtick.minor.width': '1.0'})
rcParams.update({'ytick.major.pad': '7.0'})
rcParams.update({'ytick.major.size': '7.5'})
rcParams.update({'ytick.major.width': '1.5'})
rcParams.update({'ytick.minor.pad': '7.0'})
rcParams.update({'ytick.minor.size': '3.5'})
rcParams.update({'ytick.minor.width': '1.0'})
rcParams.update({'font.size': 20})
rcParams.update({'figure.figsize':(10,5)})
rcParams.update({
    "text.usetex": False,
    "font.family": "sans",
    "font.sans-serif": ["Palatino"]})
plt.rcParams["mathtext.fontset"] = 'dejavuserif'
rc('font',**{'family':'serif','serif':['Palatino']})
plt.rcParams.update({'figure.max_open_warning': 0})# 


catal='/Users/amartinez/Desktop/PhD/Arches_and_Quintuplet_Hosek/'
pm_abs = '/Users/amartinez/Desktop/PhD/HAWK/pm_gns1_gns2_off_chunk/'



choosen_cluster = 'Arches'#TODO
# choosen_cluster = 'Quintuplet'#TODO

# ref_frame = 'relative'#TODO
ref_frame = 'absolute'#TODO
prob_lim = 0#TODO
pm_ok = 1#TODO
align_degree =2#TODO
vel_cut = pm_ok#TODO this is for the vel uncertainty in gns
max_sig = 0.1#TODO
max_sep = 0.1*u.arcsec#TODO
cluster_gone = 'yes'
radio = 15#TODO radio around the cluster to delete (in arcsec)

center_arc = SkyCoord(ra = '17h45m50.65020s', dec = '-28d49m19.51468s', equinox = 'J2000') if choosen_cluster =='Arches' else SkyCoord('17h46m14.68579s', '-28d49m38.99169s', equinox = 'J2000')#Quintuplet
#Name 0	F127M 1	e_F127M 2	F139M 3	e_F139M 4	F153M 5	e_F153M 6	dRA 7	e_dRA 8	dDE 9	e_dDE 10	pmRA 11	e_pmRA 12	pmDE 13	e_pmDE 14	t0 15	Nobs 16	chi2RA 17	chi2DE 18	Pclust 19	
col = np.arange(0,34)
if choosen_cluster == 'Arches':
    
    arches = pd.read_fwf(catal + 'Arches_from_Article.txt',sep = " ", header = None,skiprows =col,
                         names =['F127M',	'e_F127M',	'F139M',	'e_F139M',	'F153M',	'e_F153M',	
                                 'dRA',	'e_dRA',	'dDE',	'e_dDE',	'pmRA',	'e_pmRA',	'pmDE',	
                                 'e_pmDE',	't0',	'Nobs',	'chi2RA',	'chi2DE',	'Pclust'])
if choosen_cluster == 'Quintuplet' :
    arches = pd.read_fwf(catal + 'Quintuplet_from_Article.txt',sep = " ", header = None,skiprows =col,
                         names =['F127M',	'e_F127M',	'F139M',	'e_F139M',	'F153M',	'e_F153M',	
                                 'dRA',	'e_dRA',	'dDE',	'e_dDE',	'pmRA',	'e_pmRA',	'pmDE',	
                                 'e_pmDE',	't0',	'Nobs',	'chi2RA',	'chi2DE',	'Pclust'])




prob = arches['Pclust'] < prob_lim
# arches = arches[prob]

t_inter = Time(['2022-05-27T00:00:00','2011-02-15T00:00:00'],scale='utc')
d_time = (t_inter[0]-t_inter[1]).to(u.yr)

# lets move Hosek stars to GNS2 epoch (2022) before the matching
arches['dRA'] = arches['dRA'] + arches['pmRA']*d_time.value/1000
arches['dDE'] = arches['dDE'] + arches['pmDE']*d_time.value/1000

RA_DEC = center_arc.spherical_offsets_by(arches['dRA']*u.arcsec, arches['dDE']*u.arcsec)
RA = RA_DEC.ra
DEC = RA_DEC.dec

# arches = np.c_[arches,RA.value,DEC.value]
arches.insert(0,'RA',RA.value)
arches.insert(1,'DEC', DEC.value)

# take away the foregroud stars
# center = np.where((arches['F127M'] - arches['F153M'] > 1.7))
# arches = arches.iloc[center]

#select only pm motion under a certain uncertainty

epm_lim = np.where((arches['e_pmRA']<pm_ok)&(arches['e_pmDE']<pm_ok))
arches = arches.iloc[epm_lim]

hos_coord = SkyCoord(ra = arches['RA'], dec = arches['DEC'], unit = 'degree', equinox = 'J2000')

if cluster_gone == 'yes':
    center_clus = SkyCoord(ra  = [center_arc.ra.value], dec = [center_arc.dec.value], unit ='degree')
    idxc, group_md, d2d,d3d =  ap_coor.search_around_sky(center_clus, hos_coord, radio*u.arcsec)
    arches = arches.drop(arches.index[group_md],axis = 0)
    hos_coord = SkyCoord(ra = arches['RA'], dec = arches['DEC'], unit = 'degree', equinox = 'J2000')

fig, ax = plt.subplots(1,1,figsize=(10,10))
ax.set_title('%s'%(choosen_cluster))
ax.scatter(arches['RA'], arches['DEC'],s=100)
ax.scatter(arches['RA'][prob],arches['DEC'][prob],s=20,c='r',label ='Prob > %s'%(prob_lim))
ax.legend()

if choosen_cluster == 'Quintuplet':
    gns_clus = [10,2,4,3]
elif choosen_cluster == 'Arches':    
    gns_clus = [7,4,7,1]
field_one = gns_clus[0]
chip_one = gns_clus[1]
field_two = gns_clus[2]
chip_two = gns_clus[3]


if ref_frame == 'absolute':
    dmax = 100 #TODO
    # x_dis  0,y_dis 1,dvx 2,dvy 3,x1 4,y1 5,x2 6,y2 7,H1 8,dH1 9,Ks1 10,dKs1 11,H2 12,dH2 13,RaH1 14,DecH1 15,raH2 16,decH2 17
    gns = np.loadtxt(pm_abs + 'pm_GaiaRF_ep1_f%sc%s_ep2_f%sc%sdeg%s_dmax%s_sxy%s.txt'%(field_one, chip_one, field_two, chip_two,align_degree,dmax,max_sig))
# elif ref_frame == 'relative':
#     dmax = 1#TODO
#     # x_dis  0,y_dis 1,dvx 2,dvy 3,x1 4,y1 5,x2 6,y2 7,H1 8,dH1 9,Ks1 10,dKs1 11,H2 12,dH2 13,RaH1 14,DecH1 15,raH2 16,decH2 17
#     gns = np.loadtxt(pm_rel + 'pm_ep1_f%sc%s_ep2_f%sc%sdeg%s_dmax%s_sxy%s.txt'%(field_one, chip_one, field_two, chip_two,align_degree,dmax,max_sig))

unc_cut = np.where((gns[:,2]<vel_cut) &(gns[:,3]<vel_cut))
gns = gns[unc_cut]
mag_cut =np.where((gns[:,12]<20) & (gns[:,12]>12))
gns=gns[mag_cut]
color_cut =np.where((gns[:,8]- gns[:,10])>1.3)
gns=gns[color_cut]


# Choose your GNS epoch
# gns_coor = SkyCoord(ra= gns[:,-2]*u.degree, dec=gns[:,-1]*u.degree, frame = 'fk5',equinox = 'J2000',obstime='J2022.43')
gns_coor = SkyCoord(ra= gns[:,-4]*u.degree, dec=gns[:,-3]*u.degree, frame = 'fk5',equinox = 'J2000',obstime='J2015.43')


idx,d2d,d3d = hos_coord.match_to_catalog_sky(gns_coor,nthneighbor=1)# ,nthneighbor=1 is for 1-to-1 match
sep_constraint = d2d < max_sep
hos_match = arches.iloc[sep_constraint]
gns_match = gns[idx[sep_constraint]]

d_pmra = hos_match['pmRA']  - gns_match[:,0]
d_pmdec = hos_match['pmDE'] - gns_match[:,1]



# d_pmra_ls =[]
# d_pmdec_ls =[]
# for i in range(len(d_pmra)):
#     d_pmra_ls.append(float(d_pmra[i]))
# d_pmdec_ls =[]
# for i in range(len(d_pmdec)):
#     d_pmdec_ls.append(float(d_pmdec[i]))

# sig = 3
# pmra_sta = stats.sigma_clipped_stats(d_pmra_ls, sigma=sig, maxiters=10)  
# pmdec_sta = stats.sigma_clipped_stats(d_pmdec_ls, sigma=sig, maxiters=10)  
# print('Pm_Ra clip(sig = %s): mean = %.2f, sigma = %.2f'%(sig,pmra_sta[0],pmra_sta[2]))
# print('Pm_Dec clip(sig = %s): mean = %.2f, sigma = %.2f'%(sig,pmdec_sta[0],pmdec_sta[2]))

# sig_pmra=sigma_clip(d_pmra_ls,sigma=30,maxiters=20,cenfunc='mean',masked=True)
# sig_pmdec=sigma_clip(d_pmdec_ls,sigma=30,maxiters=20,cenfunc='mean',masked=True)
# d_pmra=d_pmra[sig_pmra.mask==False]
# d_pmdec=d_pmdec[sig_pmdec.mask==False]


# %
fig, ax = plt.subplots(1,2,figsize=(20,10))
ax[0].set_title('#Matches = %s. Degree %s'%(len(hos_match),align_degree))
ax[0].scatter(arches['RA'], arches['DEC'],s = 20, c='k',alpha=0.1 )
ax[0].scatter(gns[:,-2],gns[:,-1],s=20, alpha=0.1)
ax[0].scatter(hos_match['RA'],hos_match['DEC'],s=100)
ax[0].scatter(gns_match[:,-2],gns_match[:,-1],s=20)
ax[0].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))




ax[1].set_title('%s'%(choosen_cluster))
ax[1].hist(d_pmra, histtype = 'step',linewidth = 10,label = '$\overline{\Delta\mu_{ra}}$ = %.1f $\pm$ %.1f'%(np.mean(d_pmra),np.std(d_pmra)))
ax[1].hist(d_pmdec,histtype = 'step',linewidth = 10,label = '$\overline{\Delta\mu_{dec}}$ = %.1f $\pm$ %.1f'%(np.mean(d_pmdec),np.std(d_pmdec)))
ax[1].legend(loc=2)
ax[1].set_xlabel('$\Delta\mu$ (mas/yr)')
ax[1].set_xlim(-6,6)



# %%
# hos_match_ra = []
# hos_match_dec = []
# for j in range(len(hos_match)):
#     hos_match_ra.append(hos_match['pmRA'][j])
#     hos_match_dec.append(hos_match['pmDEC'][j])
pmra_fit = np.polyfit(hos_match['pmRA'], gns_match[:,0],1)
pmdec_fit = np.polyfit(hos_match['pmDE'], gns_match[:,1],1)


# %%
# marking strange vertical feature in the pm vs pm plots
# feature = np.where((hos_match['pmDE']>-2)&(hos_match['pmDE']<-1.7))
# feature_A = -1
# feature_B = -1.001
feature_A = -1
feature_B = -0.7
feature = np.where((hos_match['pmRA']>feature_A)&(hos_match['pmRA']<feature_B))
fig, ax = plt.subplots(1,2,figsize = (20,10))
ax[0].scatter(hos_match['pmRA'], gns_match[:,0])
ax[0].plot(hos_match['pmRA'], pmra_fit[1] + pmra_fit[0]*hos_match['pmRA'],  color = 'red'
           ,label ='m = %.2f \nyo = %.2f'%(pmra_fit[0],pmra_fit[1]))
# ax[0].plot(gns_match[:,0], gns_match[:,0], color = 'red')
ax[0].set_title('RA')
ax[1].set_title('DEC')

ax[0].set_xlabel('$\mu_{Hos}$ (mas/yr)')
ax[0].set_ylabel('$\mu_{gns}$ (mas/yr)')
ax[0].legend()
ax[1].scatter(hos_match['pmDE'], gns_match[:,1])
ax[1].plot(hos_match['pmDE'], pmdec_fit[1] + pmdec_fit[0]*hos_match['pmDE'], color = 'red'
           ,label ='m = %.2f \nyo = %.2f'%(pmdec_fit[0],pmdec_fit[1]))
ax[1].legend()
ax[1].set_xlabel('$\mu_{Hos}$ (mas/yr)')
ax[1].set_ylabel('$\mu_{gns}$ (mas/yr)')
ax[0].scatter(hos_match['pmRA'][list(feature[0])],gns_match[:,0][feature],color ='r',marker='x')
ax[1].scatter(hos_match['pmDE'][list(feature[0])],gns_match[:,1][feature],color ='r',marker='x')
ax[0].set_xticks(np.arange(-10,5))
ax[0].grid()
# %%
fig, ax = plt.subplots(1,2,figsize=(20,10))
ax[0].scatter(gns_match[:,-2][feature],gns_match[:,-1][feature],color = 'r', marker = 'x',s =100)
gns_match = np.delete(gns_match, feature, axis =0)
hos_match = hos_match.drop(hos_match.index[feature],axis = 0)

d_pmra_match =  hos_match['pmRA']  - gns_match[:,0]
d_pmdec_match = hos_match['pmDE'] - gns_match[:,1]


ax[0].scatter(arches['RA'], arches['DEC'],s = 20, c='k',alpha=0.01 )
ax[0].scatter(gns[:,-2],gns[:,-1],s=20, alpha=0.01)
ax[0].scatter(hos_match['RA'],hos_match['DEC'],s=100)
ax[0].scatter(gns_match[:,-2],gns_match[:,-1],s=20)


ax[0].xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
# %



ax[1].set_title('%s'%(choosen_cluster))
ax[1].hist(d_pmra_match, histtype = 'step',linewidth = 10,label = '$\overline{\Delta\mu_{ra}}$ = %.1f $\pm$ %.1f'%(np.mean(d_pmra_match),np.std(d_pmra_match)))
ax[1].hist(d_pmdec_match,histtype = 'step',linewidth = 10,label = '$\overline{\Delta\mu_{dec}}$ = %.1f $\pm$ %.1f'%(np.mean(d_pmdec_match),np.std(d_pmdec_match)))
ax[1].legend(loc=2)
ax[1].set_xlabel('$\Delta\mu$ (mas/yr)')
ax[1].set_xlim(-6,6)

sys.exit('285')
# %%
# =============================================================================
# This part deletes the points that fall at certain distance from the fitiing 
# line
# =============================================================================
bad = []
sig_dis = 3
# dist_line = abs(((-1)*pmdec_fit[0]*hos_match[:,13] + gns_match[:,1]
#         + (-1)*(pmdec_fit[1]))/np.sqrt(1**2 + pmdec_fit[0]**2))

dist_line = abs(((-1)*pmra_fit[0]*hos_match['pmRA'] + gns_match[:,0]
        + (-1)*(pmra_fit[1]))/np.sqrt(1**2 + pmra_fit[0]**2))

fig, ax  = plt.subplots(1,2,figsize =(20,10))
for k in range(len(gns_match)):
    if abs(((-1)*pmra_fit[0]*hos_match['pmRA'][k] + gns_match[:,0][k]
            + (-1)*(pmra_fit[1]))/np.sqrt(1**2 + pmra_fit[0]**2)) >np.mean(dist_line) + sig_dis*np.std(dist_line):
      
        bad.append(k)

dist_line_B = abs(((-1)*pmdec_fit[0]*hos_match['pmDE'] + gns_match[:,1]
        + (-1)*(pmdec_fit[1]))/np.sqrt(1**2 + pmdec_fit[0]**2))
for k in range(len(gns_match)):
    if abs(((-1)*pmdec_fit[0]*hos_match['pmDE'][k] + gns_match[:,1][k]
            + (-1)*(pmdec_fit[1]))/np.sqrt(1**2 + pmdec_fit[0]**2)) >np.mean(dist_line_B) + sig_dis*np.std(dist_line_B):    
       
        bad.append(k)
bad = np.unique(bad)
 

ax[0].scatter(hos_match['pmRA'], gns_match[:,0],alpha = 0.2)
ax[0].scatter(hos_match['pmRA'][bad], gns_match[:,0][bad],color = 'r')
ax[1].scatter(hos_match['pmDE'], gns_match[:,1],alpha = 0.2)
ax[1].scatter(hos_match['pmDE'][bad], gns_match[:,1][bad],color = 'r')
ax[0].plot(hos_match['pmRA'], pmra_fit[1] + pmra_fit[0]*hos_match['pmRA'],color = 'r')
ax[1].plot(hos_match['pmDE'], pmdec_fit[1] + pmdec_fit[0]*hos_match['pmDE'],color = 'r')

# %%
# %%


gns_del = np.delete(gns_match, bad, axis =0)
hos_del = hos_match.drop(hos_match.index[bad],axis = 0)
# fig, ax = plt.subplots(1,1,figsize = (10,10))

d_pmra_del =  hos_del['pmRA']  - gns_del[:,0]
d_pmdec_del = hos_del['pmDE'] - gns_del[:,1]


sig = 3#TODO
pmra_sta = stats.sigma_clipped_stats(d_pmra_del, sigma=sig, maxiters=10)  
pmdec_sta = stats.sigma_clipped_stats(d_pmdec_del, sigma=sig, maxiters=10)  
print('Pm_Ra clip(sig = %s): mean = %.2f, sigma = %.2f'%(sig,pmra_sta[0],pmra_sta[2]))
print('Pm_Dec clip(sig = %s): mean = %.2f, sigma = %.2f'%(sig,pmdec_sta[0],pmdec_sta[2]))

sig_ra=sigma_clip(d_pmra_del,sigma=sig,maxiters=10,cenfunc='mean',masked=True)
d_pmra_del_ls=np.array(d_pmra_del)[sig_ra.mask==False]
sig_dec=sigma_clip(d_pmdec_del,sigma=sig,maxiters=10,cenfunc='mean',masked=True)
d_pmdec_del_ls=np.array(d_pmdec_del)[sig_dec.mask==False]

fig, ax = plt.subplots(1,2,figsize=(20,10))
ax[0].set_title('#Matches = %s ($\sigma_{clipp}$=%s). Degree %s'%(len(d_pmdec_del_ls),sig,align_degree))
ax[0].scatter(arches['RA'], arches['DEC'],s = 20, c='k',alpha=0.01 )
ax[0].scatter(gns[:,-2],gns[:,-1],s=20, alpha=0.01)
ax[0].scatter(hos_match['RA'],hos_match['DEC'],s=100)
ax[0].scatter(gns_match[:,-2],gns_match[:,-1],s=20)
ax[0].scatter(gns_match[:,-2][bad],gns_match[:,-1][bad],color = 'r')

ax[0].xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
# %



ax[1].set_title('%s'%(choosen_cluster))
ax[1].hist(d_pmra_del_ls, histtype = 'step',linewidth = 10,label = '$\overline{\Delta\mu_{ra}}$ = %.1f $\pm$ %.1f'%(np.mean(d_pmra_del_ls),np.std(d_pmra_del_ls)))
ax[1].hist(d_pmdec_del_ls,histtype = 'step',linewidth = 10,label = '$\overline{\Delta\mu_{dec}}$ = %.1f $\pm$ %.1f'%(np.mean(d_pmdec_del_ls),np.std(d_pmdec_del_ls)))
ax[1].legend(loc=2)
ax[1].set_xlabel('$\Delta\mu$ (mas/yr)')
ax[1].set_xlim(-6,6)


# %%

pmra_fit_del = np.polyfit(hos_del['pmRA'][sig_ra.mask==False], gns_del[:,0][sig_ra.mask==False],1)
pmdec_fit_del = np.polyfit(hos_del['pmDE'][sig_dec.mask==False], gns_del[:,1][sig_dec.mask==False],1)
# %%
fig, ax = plt.subplots(1,2,figsize = (20,10))
ax[0].scatter(hos_del['pmRA'][sig_ra.mask==False], gns_del[:,0][sig_ra.mask==False])
ax[0].plot(hos_del['pmRA'][sig_ra.mask==False], pmra_fit_del[1] + pmra_fit_del[0]*hos_del['pmRA'][sig_ra.mask==False],  color = 'red'
           ,label ='m = %.2f \nyo = %.2f'%(pmra_fit_del[0],pmra_fit_del[1]))
# ax[0].plot(gns_match[:,0], gns_match[:,0], color = 'red')
ax[0].set_title('RA (deleted)')
ax[1].set_title('DEC (deleted)')

ax[0].set_xlabel('$\mu_{Hos}$ (mas/yr)')
ax[0].set_ylabel('$\mu_{gns}$ (mas/yr)')
ax[0].legend()
ax[1].scatter(hos_del['pmDE'][sig_dec.mask==False], gns_del[:,1][sig_dec.mask==False])
ax[1].plot(hos_del['pmDE'][sig_dec.mask==False], pmdec_fit_del[1] + pmdec_fit_del[0]*hos_del['pmDE'][sig_dec.mask==False], color = 'red'
           ,label ='m = %.2f \nyo = %.2f'%(pmdec_fit_del[0],pmdec_fit_del[1]))
ax[1].legend()
ax[1].set_xlabel('$\mu_{Hos}$ (mas/yr)')
ax[1].set_ylabel('$\mu_{gns}$ (mas/yr)')



# %%
d_pmra_del = hos_del['pmRA']  - gns_del[:,0]
d_pmdec_del = hos_del['pmDE'] - gns_del[:,1]

pmra_sta = stats.sigma_clipped_stats(d_pmra_del, sigma=sig, maxiters=10)  
pmdec_sta = stats.sigma_clipped_stats(d_pmdec_del, sigma=sig, maxiters=10)  
print('Pm_Ra clip(sig = %s): mean = %.2f, sigma = %.2f'%(sig,pmra_sta[0],pmra_sta[2]))
print('Pm_Dec clip(sig = %s): mean = %.2f, sigma = %.2f'%(sig,pmdec_sta[0],pmdec_sta[2]))
# %%
sig_pmra_del=sigma_clip(d_pmra_del,sigma=sig,maxiters=20,cenfunc='mean',masked=True)
sig_pmdec_del=sigma_clip(d_pmdec_del,sigma=sig,maxiters=20,cenfunc='mean',masked=True)

d_pmra_del = d_pmra_del[sig_pmra_del.mask == False]
d_pmdec_del = d_pmdec_del[sig_pmdec_del.mask == False]

bad_sig = np.where((sig_pmra_del.mask == True)&(sig_pmra_del.mask == True))
gns_del_sig = gns_del[bad_sig]
# %%
fig, ax = plt.subplots(1,2,figsize=(20,10))
ax[0].set_title('Clipped at $\sigma$ = %s'%(sig))
ax[0].scatter(arches['RA'], arches['DEC'],s = 20, c='k',alpha=0.01 )
ax[0].scatter(gns[:,-2],gns[:,-1],s=20, alpha=0.01)
ax[0].scatter(gns_del[:,-2],gns_del[:,-1],s=100, color ='b')
ax[1].set_title('# Stars = %s'%(len(d_pmra_del)))
ax[0].scatter(gns_del_sig[:,-2],gns_del_sig[:,-1],s=20, color = 'red')
ax[1].hist(d_pmra_del, histtype = 'step',linewidth = 10,label = '$\overline{\Delta\mu_{ra}}$ = %.1f $\pm$ %.1f'%(np.mean(d_pmra_del),np.std(d_pmra_del)))
ax[1].hist(d_pmdec_del,histtype = 'step',linewidth = 10,label = '$\overline{\Delta\mu_{dec}}$ = %.1f $\pm$ %.1f'%(np.mean(d_pmdec_del),np.std(d_pmdec_del)))
ax[1].legend()
ax[1].set_xlim(-6,6)


# good_ra = np.where((gns_del[sig_pmra_del.mask==False]) & (gns_del[sig_pmdec_del.mask==False]))


# %%

# =============================================================================
# # HERE we are selecting only stars far from GNS borders
# =============================================================================
# =============================================================================
# if choosen_cluster == 'Arches':
#     
#     arches = pd.read_fwf(catal + 'Arches_from_Article.txt',sep = " ", header = None,skiprows =col,
#                          names =['F127M',	'e_F127M',	'F139M',	'e_F139M',	'F153M',	'e_F153M',	
#                                  'dRA',	'e_dRA',	'dDE',	'e_dDE',	'pmRA',	'e_pmRA',	'pmDE',	
#                                  'e_pmDE',	't0',	'Nobs',	'chi2RA',	'chi2DE',	'Pclust'])
# if choosen_cluster == 'Quintuplet' :
#     arches = pd.read_fwf(catal + 'Quintuplet_from_Article.txt',sep = " ", header = None,skiprows =col,
#                          names =['F127M',	'e_F127M',	'F139M',	'e_F139M',	'F153M',	'e_F153M',	
#                                  'dRA',	'e_dRA',	'dDE',	'e_dDE',	'pmRA',	'e_pmRA',	'pmDE',	
#                                  'e_pmDE',	't0',	'Nobs',	'chi2RA',	'chi2DE',	'Pclust'])
# 
# 
# 
# 
# prob_lim = 0.3#TODO
# prob = arches['Pclust'] > prob_lim
# # arches = arches[prob]
# 
# t_inter = Time(['2022-05-27T00:00:00','2011-02-15T00:00:00'],scale='utc')
# d_time = (t_inter[0]-t_inter[1]).to(u.yr)
# 
# # lets move Hosek stars to GNS2 epoch (2022) before the matching
# arches['dRA'] = arches['dRA'] + arches['pmRA']*d_time.value/1000
# arches['dDE'] = arches['dDE'] + arches['pmDE']*d_time.value/1000
# 
# RA_DEC = center_arc.spherical_offsets_by(arches['dRA']*u.arcsec, arches['dDE']*u.arcsec)
# RA = RA_DEC.ra
# DEC = RA_DEC.dec
# 
# # arches = np.c_[arches,RA.value,DEC.value]
# arches.insert(0,'RA',RA.value)
# arches.insert(1,'DEC', DEC.value)
# hos_coord = SkyCoord(ra = arches['RA'], dec = arches['DEC'], unit = 'degree', equinox = 'J2000')
# 
# 
# # take away the foregroud stars
# # center = np.where((arches['F127M'] - arches['F153M'] > 1.7))
# # arches = arches.iloc[center]
# 
# #select only pm motion under a certain uncertainty
# pm_ok = 10#TODO
# epm_lim = np.where((arches['e_pmRA']<pm_ok)&(arches['e_pmDE']<pm_ok))
# arches = arches.iloc[epm_lim]
# 
# 
# if ref_frame == 'absolute':
#    
#     # x_dis  0,y_dis 1,dvx 2,dvy 3,x1 4,y1 5,x2 6,y2 7,H1 8,dH1 9,Ks1 10,dKs1 11,H2 12,dH2 13,RaH1 14,DecH1 15,raH2 16,decH2 17
#     gns = np.loadtxt(pm_abs + 'pm_GaiaRF_ep1_f%sc%s_ep2_f%sc%sdeg%s_dmax%s_sxy%s.txt'%(field_one, chip_one, field_two, chip_two,align_degree,dmax,max_sig))
# # elif ref_frame == 'relative':
# #     dmax = 1#TODO
# #     # x_dis  0,y_dis 1,dvx 2,dvy 3,x1 4,y1 5,x2 6,y2 7,H1 8,dH1 9,Ks1 10,dKs1 11,H2 12,dH2 13,RaH1 14,DecH1 15,raH2 16,decH2 17
# #     gns = np.loadtxt(pm_rel + 'pm_ep1_f%sc%s_ep2_f%sc%sdeg%s_dmax%s_sxy%s.txt'%(field_one, chip_one, field_two, chip_two,align_degree,dmax,max_sig))
# 
# unc_cut = np.where((gns[:,2]<vel_cut) &(gns[:,3]<vel_cut))
# gns = gns[unc_cut]
# mag_cut =np.where((gns[:,12]<18) & (gns[:,12]>12))
# gns=gns[mag_cut]
# color_cut =np.where((gns[:,8]- gns[:,10])>0)
# gns=gns[color_cut]
# gns_coor = SkyCoord(ra= gns[:,-2]*u.degree, dec=gns[:,-1]*u.degree, frame = 'fk5',equinox = 'J2000',obstime='J2022.43')
# 
# 
# # m1_ra, m2_ra =266.46, 266.47
# # m1_dec, m2_dec = -28.827, -28.840
# 
# point_ra = 266.47
# point_dec = -28.830
# 
# m1_ra, m2_ra =point_ra, point_ra
# m1_dec, m2_dec = point_dec, point_dec
# 
# 
# m_point = SkyCoord(ra =[np.mean([m1_ra, m2_ra])], dec = [np.mean([m1_dec, m2_dec])],unit = 'degree')
# idxc, group_md, d2d,d3d =  ap_coor.search_around_sky(m_point,gns_coor, 20*u.arcsec)
# 
# gns_cent = gns_coor[group_md]
# gns_cent_all = gns[group_md]
# 
# idx,d2d,d3d = hos_coord.match_to_catalog_sky(gns_cent,nthneighbor=1)# ,nthneighbor=1 is for 1-to-1 match
# sep_constraint = d2d < max_sep
# hos_match = arches[sep_constraint]
# gns_match = gns_cent_all[idx[sep_constraint]]
# 
# fig, ax = plt.subplots(1,2,figsize = (20,10))
# ax[0].set_title('#Matches = %s. Degree %s'%(len(hos_match),align_degree))
# ax[0].scatter(arches['RA'], arches['DEC'],s = 20, c='k',alpha=0.1 )
# ax[0].scatter(gns[:,-2],gns[:,-1],s=20, alpha=0.1)
# # ax[0].scatter(gns_cent[:,-2], gns_cent[:,-1])
# ax[0].scatter(gns_cent.ra,gns_cent.dec)
# ax[0].scatter(hos_match['RA'],hos_match['DEC'],s=100, color = 'green')
# ax[0].scatter(gns_match[:,-2],gns_match[:,-1],s=20, color = 'pink')
# ax[0].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# 
# ax[0].axvline(m_point.ra.value,color = 'red')
# # ax[0].axvline(m2_ra,color = 'red')
# ax[0].axhline(m_point.dec.value,color = 'red')
# # ax[0].axhline(m2_dec,color = 'red')
# 
# 
# 
# d_pmra = hos_match['pmRA']  - gns_match[:,0]
# d_pmdec = hos_match['pmDE'] - gns_match[:,1]
# 
# 
# 
# ax[1].set_title('%s'%(choosen_cluster))
# ax[1].hist(d_pmra, histtype = 'step',linewidth = 10,label = '$\overline{\Delta\mu_{ra}}$ = %.1f $\pm$ %.1f'%(np.mean(d_pmra),np.std(d_pmra)))
# ax[1].hist(d_pmdec,histtype = 'step',linewidth = 10,label = '$\overline{\Delta\mu_{dec}}$ = %.1f $\pm$ %.1f'%(np.mean(d_pmdec),np.std(d_pmdec)))
# ax[1].legend(loc=2)
# ax[1].set_xlabel('$\Delta\mu$ (mas/yr)')
# ax[1].set_xlim(-6,6)
# 
# # %
# 
# pmra_fit = np.polyfit(hos_match['pmRA'], gns_match[:,0],1)
# pmdec_fit = np.polyfit(hos_match['pmDE'], gns_match[:,1],1)
# 
# bad = []
# dist_line =  abs(((-1)*pmra_fit[0]*hos_match['pmRA'] + gns_match[:,0]
#         + (-1)*(pmra_fit[1]))/np.sqrt(1**2 + pmra_fit[0]**2))
# for k in range(len(gns_match)):
#     if abs(((-1)*pmra_fit[0]*hos_match['pmRA'][k] + gns_match[:,0][k]
#             + (-1)*(pmra_fit[1]))/np.sqrt(1**2 + pmra_fit[0]**2)) >np.mean(dist_line) + 2*np.std(dist_line):
#        
#         bad.append(k)
#         
# 
# 
# # %
# fig, ax = plt.subplots(1,2,figsize = (20,10))
# ax[0].scatter(hos_match['pmRA'], gns_match[:,0])
# ax[0].plot(hos_match['pmRA'], pmra_fit[1] + pmra_fit[0]*hos_match['pmRA'],  color = 'red'
#            ,label ='m = %.2f \nyo = %.2f'%(pmra_fit[0],pmra_fit[1]))
# # ax[0].plot(gns_match[:,0], gns_match[:,0], color = 'red')
# ax[0].set_title('RA')
# ax[1].set_title('DEC')
# 
# ax[0].set_xlabel('$\mu_{Hos}$ (mas/yr)')
# ax[0].set_ylabel('$\mu_{gns}$ (mas/yr)')
# ax[0].legend()
# ax[1].scatter(hos_match['pmDE'], gns_match[:,1])
# ax[1].plot(hos_match['pmDE'], pmdec_fit[1] + pmdec_fit[0]*hos_match['pmDE'], color = 'red'
#            ,label ='m = %.2f \nyo = %.2f'%(pmdec_fit[0],pmdec_fit[1]))
# ax[1].legend()
# ax[1].set_xlabel('$\mu_{Hos}$ (mas/yr)')
# ax[1].set_ylabel('$\mu_{gns}$ (mas/yr)')
# ax[0].scatter(hos_match['pmRA'][bad], gns_match[:,0][bad],color = 'r')
# ax[1].scatter(hos_match['pmDE'][bad], gns_match[:,1][bad],color = 'r')
# 
# # %
# fig, ax = plt.subplots(1,2,figsize=(20,10))
# ax[0].set_title('#Matches = %s. Degree %s'%(len(hos_match),align_degree))
# ax[0].scatter(arches['RA'], arches['DEC'],s = 20, c='k',alpha=0.01 )
# ax[0].scatter(gns[:,-2],gns[:,-1],s=20, alpha=0.01)
# ax[0].scatter(hos_match['RA'],hos_match['DEC'],s=100)
# ax[0].scatter(gns_match[:,-2],gns_match[:,-1],s=20)
# ax[0].scatter(gns_match[:,-2][bad],gns_match[:,-1][bad],color = 'r')
# # %
# gns_del = np.delete(gns_match, bad, axis =0)
# hos_del = hos_match.drop(hos_match.index[bad] , axis = 0)
# # fig, ax = plt.subplots(1,1,figsize = (10,10))
# 
# d_pmra_del =  hos_del['pmRA']  - gns_del[:,0]
# d_pmdec_del = hos_del['pmDE'] - gns_del[:,1]
# 
# # d_pmra_del =  d_pmra[bad]
# # d_pmdec_del = d_pmdec[bad]
# 
# ax[1].set_title('%s'%(choosen_cluster))
# ax[1].hist(d_pmra_del, histtype = 'step',linewidth = 10,label = '$\overline{\Delta\mu_{ra}}$ = %.1f $\pm$ %.1f'%(np.mean(d_pmra_del),np.std(d_pmra_del)))
# ax[1].hist(d_pmdec_del,histtype = 'step',linewidth = 10,label = '$\overline{\Delta\mu_{dec}}$ = %.1f $\pm$ %.1f'%(np.mean(d_pmdec_del),np.std(d_pmdec_del)))
# ax[1].legend(loc=2)
# ax[1].set_xlabel('$\Delta\mu$ (mas/yr)')
# ax[1].set_xlim(-6,6)
# 
# =============================================================================



