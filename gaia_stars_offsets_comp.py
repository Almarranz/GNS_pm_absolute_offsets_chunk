#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 21:27:15 2022

@author: amartinez
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 17:31:55 2022

@author: amartinez
"""

# Compares the pm for common stars in GNS and Gaia

# %%imports
from astropy.table import Table

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import astropy.coordinates as ap_coor
import pandas as pd
import sys
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import NullFormatter
from scipy.stats import sigmaclip


# %%plotting parametres
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
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Palatino']})
plt.rcParams.update({'figure.max_open_warning': 0})# a warniing for matplot lib pop up because so many plots, this turining it of
# %%
field_one = 7
chip_one = 4
field_two = 7
chip_two =1
degree  =2
GNS_2='/Users/amartinez/Desktop/PhD/HAWK/GNS_2off_chunk/lists/%s/chip%s/'%(field_two, chip_two)
GNS_1='/Users/amartinez/Desktop/PhD/HAWK/GNS_1off_chunk/lists/%s/chip%s/'%(field_one, chip_one)
           
pm_cat = '/Users/amartinez/Desktop/PhD/HAWK/pm_gns1_gns2_off_chunk/'
pruebas1 = '/Users/amartinez/Desktop/PhD/HAWK/GNS_1off_chunk/pruebas/'
pruebas2 = '/Users/amartinez/Desktop/PhD/HAWK/GNS_2off_chunk/pruebas/'

GA_epoch = 2

if GA_epoch ==1:
    bad1 = np.loadtxt(pruebas1 + 'bad1_f%sc%s.txt'%(field_one,chip_one))
elif GA_epoch ==2:
    bad1 = np.loadtxt(pruebas2 + 'bad2_f%sc%s.txt'%(field_two,chip_two))
if GA_epoch ==1:
    # ra, dec, dra(mas), ddec(mas), x, y, pmra(mas/yr), pmdec, dpmra(mas/yr), dpmdec, dradec
    gaia = np.loadtxt(GNS_1 + 'gaia_refstars_on_gns1_f%sc%s.txt'%(field_one,chip_one))
    # gaia = np.loadtxt(GNS_1 + 'ALL_gaia_refstars_on_gns1_f%sc%s.txt'%(field_one,chip_one))
if GA_epoch ==2:
    # ra, dec, dra(mas), ddec(mas), x, y, pmra(mas/yr), pmdec, dpmra(mas/yr), dpmdec, dradec
    gaia = np.loadtxt(GNS_2 + 'gaia_refstars_on_gns2_f%sc%s_gns1_f%sc%s.txt'%(field_two,chip_two,field_one,chip_one))
    # gaia =np.loadtxt(GNS_2 + 'ALL_gaia_refstars_on_gns2_f%sc%s_gns1_f%sc%s.txt'%(field_two,chip_two,field_one,chip_one))

dmax =100#TODO   
max_sig = 0.1#TODO
# x_dis,y_dis,dvx,dvy,x1,y1,x2,y2,H1,dH1,Ks1,dKs1,H2,dH2,RaH1,DecH1,raH2,decH2
gns = np.loadtxt(pm_cat + 'pm_GaiaRF_ep1_f%sc%s_ep2_f%sc%sdeg%s_dmax%s_sxy%s.txt'%(field_one,chip_one,field_two,chip_two,degree,dmax,max_sig))
# gns = np.loadtxt(pm_cat + 'pm_ep1_f%sc%s_ep2_f%sc%sdeg%s_dmax%s.txt'%(field_one,chip_one,field_two,chip_two,degree,dmax))

# %%
bins = 20
fig, ax = plt.subplots(1,2, figsize = (20,10))
ax[0].hist(gns[:,0],bins=bins,label = '$\overline{\mu_{ra}}$ =%.2f\n$\sigma_{ra}$ =%.2f'%(np.mean(gns[:,0]),np.std(gns[:,0])))
ax[1].hist(gns[:,1],bins=bins,label = '$\overline{\mu_{dec}}$ =%.2f\n$\sigma_{dec}$ =%.2f'%(np.mean(gns[:,1]),np.std(gns[:,1])))
ax[0].legend()
ax[1].legend()
ax[0].set_xlabel('$\mu_{ra}$ (mas/yr)')
ax[1].set_xlabel('$\mu_{dec}$ (mas/yr)')
ax[0].set_title('Degree = %s'%(degree))
ax[1].set_title('GNS1:f%s_c%s, GNS2: f%s_c%s'%(field_one,chip_one,field_two,chip_two))
# gns = np.loadtxt(pruebas2 + 'fg_pm_ep1_f%sc%s_ep2_f%sc%sdeg%s_dmax%s.txt'%(field_one,chip_one,field_two,chip_two,degree,dmax))
gns_coord = SkyCoord(ra = gns[:,-2], dec = gns[:,-1], unit = 'degree',equinox ='J2000', frame = 'fk5', obstime='J2022.4' )
gns_coord = gns_coord.transform_to('icrs')
gaia_coord = SkyCoord(ra = gaia[:,0], dec = gaia[:,1], unit = 'degree',frame ='icrs',obstime='J2016.0' )
# gaia_ALL_coord = SkyCoord(ra = gaia_ALL[:,0], dec = gaia_ALL[:,1], unit = 'degree',frame ='icrs',obstime='J2016.0' )


# %
max_sep = 0.1 * u.arcsec
idx,d2d,d3d = gaia_coord.match_to_catalog_sky(gns_coord,nthneighbor=1)# ,nthneighbor=1 is for 1-to-1 match
sep_constraint = d2d < max_sep
gaia_match = gaia[sep_constraint]
gns2_match = gns[idx[sep_constraint]]

# idxA,d2dA,d3dA = gaia_ALL_coord.match_to_catalog_sky(gns_coord,nthneighbor=1)# ,nthneighbor=1 is for 1-to-1 match
# sep_constraintA = d2dA < max_sep
# gaia_ALL_match = gaia_ALL[sep_constraintA]
# gns2_ALL_match = gns[idxA[sep_constraintA]]
# print(len(gaia_match))


# %
# 'ra, dec, dra(mas), ddec(mas), x, y, pmra(mas/yr), pmdec, dpmra(mas/yr), dpmdec,dradec'
mul_res = gaia_match[:,6]-gns2_match[:,0]
mub_res = gaia_match[:,7]-gns2_match[:,1]

# mul_res_ALL = gaia_ALL_match[:,6]-gns2_ALL_match[:,0]
# mub_res_ALL = gaia_ALL_match[:,7]-gns2_ALL_match[:,1]

# %%
a,b = -4,4
# =============================================================================
# fig, ax = plt.subplots(1,2, figsize = (20,10))
# ax[0].scatter(gns2_match[:,12],mul_res)
# ax[1].scatter(gns2_match[:,12],mub_res )
# ax[0].set_ylabel(r'$res_x {(mas\ yr^{-1})}$',fontsize =30) 
# ax[1].set_ylabel(r'$res_y {(mas\ yr^{-1})}$',fontsize =30) 
# ax[0].set_xlabel('[H]')
# ax[1].set_xlabel('[H]')
# ax[0].axhline(0, color='b', alpha = 0.5)
# ax[0].axhline(np.mean(mul_res), color = 'r')
# ax[0].axhline(np.mean(mul_res) + np.std(mul_res), color = 'r', linestyle = 'dashed')
# ax[0].axhline(np.mean(mul_res) - np.std(mul_res), color = 'r', linestyle = 'dashed')
# ax[1].axhline(np.mean(mub_res), color = 'r')
# ax[1].axhline(np.mean(mub_res) + np.std(mub_res), color = 'r', linestyle = 'dashed')
# ax[1].axhline(np.mean(mub_res) - np.std(mub_res), color = 'r', linestyle = 'dashed')
# =============================================================================
bins ='auto'
fig, ax = plt.subplots(1,2, figsize = (20,10))
ax[0].set_title('%s stars. Degree = %s'%(len(mul_res),degree))
ax[1].set_title('GNS1: f%s,c%s. GNS2: f%s,c%s'%(field_one,chip_one,field_two,chip_two))
ax[0].hist(mul_res, bins =bins)
ax[1].hist(mub_res, bins=bins)

ax[0].set_xlabel(r'$\Delta \mu_{ra} {(mas\ yr^{-1})}$',fontsize =30) 
ax[1].set_xlabel(r'$\Delta \mu_{dec} {(mas\ yr^{-1})}$',fontsize =30) 
ax[0].axvline(np.mean(mul_res), color = 'r', label = 'mean = %.3f'%(np.mean(mul_res)))
ax[0].axvline(np.mean(mul_res) + np.std(mul_res), color = 'r', linestyle = 'dashed',label = '1$\sigma$ = %.3f'%(np.std(mul_res)))
ax[0].axvline(np.mean(mul_res) - np.std(mul_res), color = 'r', linestyle = 'dashed')
ax[1].axvline(np.mean(mub_res), color = 'r',label = 'mean = %.3f'%(np.mean(mub_res)))
ax[1].axvline(np.mean(mub_res) + np.std(mub_res), color = 'r', linestyle = 'dashed',label = '1$\sigma$ = %.3f'%(np.std(mub_res)))
ax[1].axvline(np.mean(mub_res) - np.std(mub_res), color = 'r', linestyle = 'dashed')
ax[0].set_xlim(a,b)
ax[1].set_xlim(a,b)
ax[0].legend()
ax[1].legend()
print(mub_res)
plt.show()
# sys.exit('140') 

# %%
# %

# =============================================================================
# nullfmt = NullFormatter()         # no labels
# 
# # definitions for the axes
# left, width = 0.1, 0.50
# bottom, height = 0.1, 0.50
# bottom_h = left_h = left + width + 0.02
# 
# rect_scatter = [left, bottom, width, height]
# rect_histx = [left, bottom_h, width, 0.2]
# rect_histy = [left_h, bottom, 0.2, height]
# 
# # start with a rectangular Figure
# plt.figure(1, figsize=(10, 10))
# 
# 
# axScatter = plt.axes(rect_scatter)
# axScatter.grid()
# # plt.xlim(a,b)
# # plt.ylim(a,b)
# axHistx = plt.axes(rect_histx)
# # plt.xlim(a,b)
# axHisty = plt.axes(rect_histy)
# # plt.ylim(a,b)
# # no labels
# axHistx.xaxis.set_major_formatter(nullfmt)
# axHisty.yaxis.set_major_formatter(nullfmt)
# 
# axScatter.scatter(mul_res, mub_res)
# axScatter.set_xlabel(r'$\Delta \mu_{ra} {(mas\ yr^{-1})}$',fontsize =20)
# axScatter.set_ylabel(r'$\Delta \mu_{dec} {(mas\ yr^{-1})}$',fontsize =20)
# axScatter.axvline(np.median(mul_res), color = 'r')
# axScatter.axhline(np.median(mub_res), color = 'r')
# axHistx.hist(mul_res, bins=bins)
# axHisty.hist(mub_res, bins=bins, orientation='horizontal')
# axScatter.set_xlim(a,b)
# axScatter.set_ylim(a,b)
# axHistx.set_xlim(a,b)
# axHisty.set_ylim(a,b)
# =============================================================================

# %%
a,b = -4,4
bad =[]
bad_b =[]
clip_fac =2.5#TODO
fig, ax = plt.subplots(1,1,figsize=(8,8))
ax.set_title('Gaia%s stars matched = %s, Degree = %s'%(GA_epoch,len(gaia_match), degree,))
dmul_res = np.sqrt(gaia_match[:,8]**2 + gns2_match[:,2]**2)
dmub_res = np.sqrt(gaia_match[:,9]**2 + gns2_match[:,3]**2)
ax.scatter(mul_res, mub_res)
ax.errorbar(mul_res, mub_res,dmul_res,dmub_res,fmt ='None')
mul_res_c, low, upp = sigmaclip(mul_res,clip_fac, clip_fac)    
for i in range(len(mul_res)):
    if (mul_res[i] <low) or (mul_res[i] >upp):
        ax.annotate('%.i(%s)'%(gaia_match[i,-1],i),(mul_res[i], mub_res[i]))
        bad.append(int(gaia_match[i,-1]))
mul_res_b, lowb, uppb = sigmaclip(mub_res,clip_fac, clip_fac) 
for i in range(len(mul_res)):
    if (mub_res[i] <lowb) or (mub_res[i] >uppb):
        ax.annotate('%.i(%s)'%(gaia_match[i,-1],i),(mul_res[i], mub_res[i]))
        bad.append(int(gaia_match[i,-1]))
ax.set_xlabel(r'$\Delta \mu_{ra} {(mas\ yr^{-1})}$',fontsize =20)
ax.set_ylabel(r'$\Delta \mu_{dec} {(mas\ yr^{-1})}$',fontsize =20)
ax.axvline(np.median(mul_res), color = 'r')
ax.axhline(np.median(mub_res), color = 'r')
ax.set_xticks(np.arange(-4,4))
ax.set_xlim(a,b)
ax.set_ylim(a,b)
ax.axvline(np.mean(mul_res) + np.std(mul_res), color = 'r', linestyle = 'dashed')
ax.axvline(np.mean(mul_res) - np.std(mul_res), color = 'r', linestyle = 'dashed')
ax.axhline(np.mean(mub_res) + np.std(mub_res), color = 'r', linestyle = 'dashed')
ax.axhline(np.mean(mub_res) - np.std(mub_res), color = 'r', linestyle = 'dashed')
ax.grid()
print('GA_epoch',GA_epoch)
print(set(bad))
max_e = np.where(dmub_res == max(dmub_res))
print(max_e)
print(gaia_match[max_e, -3])

# %
fig, ax = plt.subplots(1,1,figsize =(10,10))
ax.set_title('Stars %s clipped before alignment'%(bad1))
ax.scatter(gns[:,-2],gns[:,-1], alpha = 0.1)
# ax.quiver(gaia_match[:,0],gaia_match[:,1], gaia_match[:,6],gaia_match[:,7],color ='grey',label='Gaia')
# ax.quiver(gns2_match[:,-2],gns2_match[:,-1], gns2_match[:,0],gns2_match[:,1],label ='GNS')
ax.quiver(gns2_match[:,-2],gns2_match[:,-1],mul_res,mub_res,label ='pm residual', color ='r')
ax.legend(loc=2)
for i in range(len(gaia_match)):
    # ax.annotate('%.i'%(gaia_match[i,-1]),(gaia_match[i,0],gaia_match[i,1]+0.001))
    ax.annotate('%.i'%(gaia_match[i,-1]),(gns2_match[i,-2],gns2_match[i,-1]+0.001))
# %
# Gaia
# ra, dec, dra(mas), ddec(mas), x, y, pmra(mas/yr), pmdec, dpmra(mas/yr), dpmdec, dradec
# GNS
# x_dis,y_dis,dvx,dvy,x1,y1,x2,y2,H1,dH1,Ks1,dKs1,H2,dH2,RaH1,DecH1,raH2,decH2
if GA_epoch == 1:
    ra_res =  gaia_match[:,4]-gns2_match[:,4]
    dec_res = gaia_match[:,5]-gns2_match[:,5]
if GA_epoch == 2:
    ra_res =  gaia_match[:,4]-gns2_match[:,6]
    dec_res = gaia_match[:,5]-gns2_match[:,7]
    
# %
# =============================================================================
# fig, ax = plt.subplots(1,2, figsize = (20,10))
# ax[0].set_title('%s stars. Degree = %s'%(len(mul_res),degree))
# ax[1].set_title('GNS1: f%s,c%s. GNS2: f%s,c%s'%(field_one,chip_one,field_two,chip_two))
# ax[0].hist(ra_res, bins =bins)
# ax[1].hist(dec_res, bins=bins)
# 
# ax[0].set_xlabel(r'$\Delta ra(mas)$',fontsize =30) 
# ax[1].set_xlabel(r'$\Delta dec(mas)$',fontsize =30) 
# ax[0].axvline(np.mean(ra_res), color = 'r', label = 'mean = %.3f'%(np.mean(ra_res)))
# ax[0].axvline(np.mean(ra_res) + np.std(ra_res), color = 'r', linestyle = 'dashed',label = '1$\sigma$ = %.3f'%(np.std(ra_res)))
# ax[0].axvline(np.mean(ra_res) - np.std(ra_res), color = 'r', linestyle = 'dashed')
# ax[1].axvline(np.mean(dec_res), color = 'r',label = 'mean = %.3f'%(np.mean(dec_res)))
# ax[1].axvline(np.mean(dec_res) + np.std(dec_res), color = 'r', linestyle = 'dashed',label = '1$\sigma$ = %.3f'%(np.std(dec_res)))
# ax[1].axvline(np.mean(dec_res) - np.std(dec_res), color = 'r', linestyle = 'dashed')
# # ax[0].set_xlim(a,b)
# # ax[1].set_xlim(a,b)
# 
# ax[0].legend()
# ax[1].legend()
# =============================================================================
# %
# Gaia
# ra, dec, dra(mas), ddec(mas), x, y, pmra(mas/yr), pmdec, dpmra(mas/yr), dpmdec, dradec
# GNS
# x_dis,y_dis,dvx,dvy,x1,y1,x2,y2,H1,dH1,Ks1,dKs1,H2,dH2,RaH1,DecH1,raH2,decH2
bad =[]
bad_b =[]
# clip_fac = 3
fig, ax = plt.subplots(1,1,figsize=(8,8))
ax.set_title('Gaia%s stars matched = %s, Degree = %s'%(GA_epoch,len(gaia_match), degree,))
dra_res = np.sqrt(gaia_match[:,2]**2 )
ddec_res = np.sqrt(gaia_match[:,3]**2)
ax.scatter(ra_res, dec_res)
ax.errorbar(ra_res, dec_res,dra_res,ddec_res,fmt ='None')
ra_res_c, low, upp = sigmaclip(ra_res,clip_fac, clip_fac)    
for i in range(len(ra_res)):
    if (ra_res[i] <low) or (ra_res[i] >upp):
        ax.annotate('%.i(%s)'%(gaia_match[i,-1],i),(ra_res[i], dec_res[i]))
        bad.append(int(gaia_match[i,-1]))
dec_res_c, lowb, uppb = sigmaclip(dec_res,clip_fac, clip_fac) 
for i in range(len(mul_res)):
    if (dec_res[i] <lowb) or (dec_res[i] >uppb):
        ax.annotate('%.i(%s)'%(gaia_match[i,-1],i),(ra_res[i],dec_res[i]))
        bad.append(int(gaia_match[i,-1]))
ax.set_xlabel(r'$\Delta ra(mas)$',fontsize =30)
ax.set_ylabel(r'$\Delta dec(mas)$',fontsize =30)
ax.axvline(np.median(ra_res), color = 'r')
ax.axhline(np.median(dec_res), color = 'r')
# ax.set_xticks(np.arange(-4,4))
# ax.set_xlim(a,b)
# ax.set_ylim(a,b)
ax.axvline(np.mean(ra_res) + np.std(ra_res), color = 'r', linestyle = 'dashed')
ax.axvline(np.mean(ra_res) - np.std(ra_res), color = 'r', linestyle = 'dashed')
ax.axhline(np.mean(dec_res) + np.std(dec_res), color = 'r', linestyle = 'dashed')
ax.axhline(np.mean(dec_res) - np.std(dec_res), color = 'r', linestyle = 'dashed')
ax.grid()
print('GA_epoch',GA_epoch)
print(bad)
# %%
fig, ax = plt.subplots(1,2, figsize = (20,10))
ax[0].set_title('%s stars. Degree = %s'%(len(dec_res),degree))
ax[1].set_title('GNS1: f%s,c%s. GNS2: f%s,c%s'%(field_one,chip_one,field_two,chip_two))
ax[0].hist(ra_res, bins =bins)
ax[1].hist(dec_res, bins=bins)

ax[0].set_xlabel(r'$\Delta ra (mas)$',fontsize =30) 
ax[1].set_xlabel(r'$\Delta dec (mas)$',fontsize =30) 
ax[0].axvline(np.mean(ra_res), color = 'r', label = 'mean = %.3f'%(np.mean(ra_res)))
ax[0].axvline(np.mean(ra_res) + np.std(ra_res), color = 'r', linestyle = 'dashed',label = '1$\sigma$ = %.3f'%(np.std(ra_res)))
ax[0].axvline(np.mean(ra_res) - np.std(ra_res), color = 'r', linestyle = 'dashed')
ax[1].axvline(np.mean(dec_res), color = 'r',label = 'mean = %.3f'%(np.mean(dec_res)))
ax[1].axvline(np.mean(dec_res) + np.std(dec_res), color = 'r', linestyle = 'dashed',label = '1$\sigma$ = %.3f'%(np.std(dec_res)))
ax[1].axvline(np.mean(dec_res) - np.std(dec_res), color = 'r', linestyle = 'dashed')
ax[0].set_xlim(a,b)
ax[1].set_xlim(a,b)
ax[0].legend()
ax[1].legend()
print(dec_res)
plt.show()






