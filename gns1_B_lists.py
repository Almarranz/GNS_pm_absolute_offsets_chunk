#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 13:30:42 2023

@author: amartinez
"""

# Generates the GNS1 second reduction with the Ks and H magnitudes
# IN this script we wont cutoff GNS2 over GNS1

import numpy as np
import matplotlib.pyplot as plt
import astroalign as aa
from astropy.io.fits import getheader
from astropy.io import fits
from scipy.spatial import distance
import pandas as pd
import sys
import time
from astropy.wcs import WCS
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
# %% 
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
plt.rcParams.update({'figure.max_open_warning': 0})# a warniing for matplot lib pop up because so many plots, this turining it of
# %%
#%%
field_one = '7'
chip_one = '4'
field_two = '7'
chip_two = '1'

GNS_1='/Users/amartinez/Desktop/PhD/HAWK/GNS_1/lists/%s/chip%s/'%(field_one, chip_one)
GNS_2='/Users/amartinez/Desktop/PhD/HAWK/GNS_2/lists/%s/chip%s/'%(field_two, chip_two)

GNS_2off = '/Users/amartinez/Desktop/PhD/HAWK/GNS_2off_chunk/lists/%s/chip%s/'%(field_two, chip_two)
GNS_1off = '/Users/amartinez/Desktop/PhD/HAWK/GNS_1off_chunk/lists/%s/chip%s/'%(field_one, chip_one)

pruebas1 = '/Users/amartinez/Desktop/PhD/HAWK/GNS_1off_chunk/pruebas/'
pruebas2 = '/Users/amartinez/Desktop/PhD/HAWK/GNS_2off_chunk/pruebas/'

# 0     1    2   3   4   5   6    7    8    9   
# ra1, dec1, x1, y1, f1, H1, dx1, dy1, df1, dH1 = np.loadtxt(GNS + 'stars_calibrated_H_chip1.txt', unpack = True)
gns1_H = np.loadtxt(GNS_1 + 'stars_calibrated_H_chip%s.txt'%(chip_one))
gns1_K = np.loadtxt(GNS_1 + 'stars_calibrated_Ks_chip%s.txt'%(chip_one))

print(len('check the time')*'*'+'\n','CHECK THE TIME'+'\n'+len('check the time')*'*')
gns1H_coor = SkyCoord(ra = gns1_H[:,0],dec = gns1_H[:,1], unit = 'degree', frame = 'fk5',equinox ='J2000',obstime='J2015.4')
if field_one == 7:
    gns1K_coor = SkyCoord(ra = gns1_K[:,0],dec = gns1_K[:,1], unit = 'degree', frame = 'fk5',equinox ='J2000',obstime='J2018.4')
else:
    gns1K_coor = SkyCoord(ra = gns1_K[:,0],dec = gns1_K[:,1], unit = 'degree', frame = 'fk5',equinox ='J2000',obstime='J2015.4')

max_sep = 0.05 * u.arcsec
idx,d2d,d3d = gns1H_coor.match_to_catalog_sky(gns1K_coor)# ,nthneighbor=1 is for 1-to-1 match
sep_constraint = d2d < max_sep
gns1H_match = gns1_H[sep_constraint]
gns1K_match = gns1_K[idx[sep_constraint]]




gns1_all = np.c_[gns1H_match, gns1K_match[:,5],gns1K_match[:,9]]
# max_sig = 0.3
max_sig = 0.1
# unc_cut = np.where(np.sqrt(gns1_all[:,1]**2 + gns1_all[:,3]**2)<max_sig)
unc_cut = np.where((gns1_all[:,6]<max_sig) & (gns1_all[:,7]<max_sig))
gns1 = gns1_all[unc_cut]
# np.savetxt(GNS_1off +'stars_calibrated_HK_chip%s_sxy%s.txt'%(chip_one,max_sig), gns1, fmt ='%.8f',
#                  header = 'ra1 0, dec1 1, x1 2, y1 3, f1 4, H1 5, dx1 6, dy1 7, df1 8, dH1 9 ,Ks 10, dKs 11')

#Cut GNS1 on GNS2 and GNS2 on GNS1. This way we will aligning the exact same
# areas in both surveys and using the same Gaia stars, so both alignment will
# suffer the same kind of distorsions

gns1_gal = SkyCoord(ra = gns1[:,0], dec = gns1[:,1], 
                    unit = 'degree', frame = 'fk5', equinox = 'J2000',
                    obstime = 'J2015.43').galactic
# %%
gns2_all = np.loadtxt(GNS_2 + 'stars_calibrated_H_chip%s.txt'%(chip_two))
unc_cut2 = np.where((gns2_all[:,6]<max_sig) & (gns2_all[:,7]<max_sig))
gns2 = gns2_all[unc_cut2]

np.savetxt(GNS_1off +'stars_calibrated_HK_chip%s_on_gns2_f%sc%s_sxy%s.txt'%(chip_one,field_two,chip_two,max_sig), 
           gns1, fmt ='%.8f',
                  header = 'ra1 0, dec1 1, x1 2, y1 3, f1 4, H1 5, dx1 6, dy1 7, df1 8, dH1 9 ,Ks 10, dKs 11')


fig, ax = plt.subplots(1,1,figsize =(10,10))
ax.scatter(gns2[:,0], gns2[:,1])
ax.scatter(gns1[:,0], gns1[:,1])
ax.set_xlabel('Ra(deg)')
ax.set_ylabel('Dec(deg)')


np.savetxt(GNS_2off +'stars_calibrated_H_chip%s_on_gns1_f%sc%s_sxy%s.txt'%(chip_two,field_one,chip_one,max_sig)
           ,gns2, fmt ='%.8f',
            header = 'ra1 0, dec1 1, x1 2, y1 3, f1 4, H1 5, dx1 6, dy1 7, df1 8, dH1 9')








