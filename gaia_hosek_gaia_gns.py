#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 18:37:25 2023

@author: amartinez
"""

# Here we ara going to compute the residuals of the Gaia stars present in 
# Hosek and the one present in GNS

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
from matplotlib import rcParams
from astroquery.gaia import Gaia
from compare_lists import compare_lists
import pandas as pd
import time
from astropy.time import Time
from matplotlib.ticker import FormatStrFormatter
# %%plotting parametres
from matplotlib import rc
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

# Arches and Gaia stars around arches.
choosen_cluster = 'Arches'#TODO
# choosen_cluster = 'Quintuplet'#TODO

pm_ok = 1#TODO
prob_lim = 0#TODO
align_degree =1#TODO
max_sig = 0.1#TODO
vel_cut = 100#TODO this is for the vel uncertainty in gns
ref_frame = 'absolute'#TODO


# center_arc = SkyCoord(ra = '17h45m50.4769267s', dec = '-28d49m19.16770s') if choosen_cluster =='Arches' else SkyCoord('17h46m15.13s', '-28d49m34.7s', frame='icrs',obstime ='J2016.0')#Quintuplet
center_arc = SkyCoord(ra = '17h45m50.65020s', dec = '-28d49m19.51468s', equinox = 'J2000') if choosen_cluster =='Arches' else SkyCoord('17h46m15.13s', '-28d49m34.7s', frame='icrs',obstime ='J2016.0')#Quintuplet

# names=('Name','F127M','e_F127M','F153M','e_F153M','ra*','e_ra*','dec','e_dec','pm_ra*','e_pm_ra*','pm_dec','e_pm_dec','t0','n_epochs','dof','chi2_ra*','chi2_dec','Orig_name','Pclust')>
# arches=Table.read(catal + 'Arches_cat_H22_Pclust.fits') if choosen_cluster =='Arches' else Table.read(catal + 'Quintuplet_cat_H22_Pclust.fits')

#NName 0	 F127M 1 e_F127M 2	F139M 3	e_F139M 4	F153M 5	e_F153M 6	dRA 7	e_dRA 8	dDE 9	e_dDE 10 	pmRA 11	e_pmRA 12	pmDE 13	e_pmDE 14	t0 15	Nobs 16	chi2RA 17	chi2DE 18	Pclust 19	pmDE 20	e_pmDE 21	t0 22	Nobs 23	chi2RA 24	chi2DE 25	Pclust 26
col = np.arange(0,34)
if choosen_cluster == 'Arches':
    arches = pd.read_fwf(catal + 'Arches_from_Article.txt',sep = " ", header = None,skiprows =col,
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



hose_coord = SkyCoord(ra = arches['RA'], dec = arches['DEC'], unit = 'degree', frame = 'icrs', obstime = 'J2016.0')

Gaia.MAIN_GAIA_TABLE = "gaiaedr3.gaia_source" # Select early Data Release 3
Gaia.ROW_LIMIT = -1  # Ensure the default row limit.

rad = 150*u.arcsec
j = Gaia.cone_search_async(center_arc, rad)
gaia_ = j.get_results()

e_pm = 0.3
# WARNING: np.where was giving me porblems when I set many conditions in one go.
selec1 = np.where((gaia_['astrometric_params_solved']==31)&(gaia_['duplicated_source'] ==False))
gaia_good1 = gaia_[selec1]
selec2 = np.where((gaia_good1['parallax_over_error']>=-10)&(gaia_good1['astrometric_excess_noise_sig']<=2))
gaia_good2 = gaia_good1[selec2]
selec3 = np.where((gaia_good2['phot_g_mean_mag']>13)&(gaia_good2['pm']>0))
gaia_good3 = gaia_good2[selec3]
selec4 = np.where((gaia_good3['pmra_error']<e_pm)&(gaia_good3['pmdec_error']<e_pm))   
gaia_good4 = gaia_good3[selec4] 
gaia_good = gaia_good4
gaia_coord =  SkyCoord(ra=gaia_good['ra'], dec=gaia_good['dec'], unit = 'degree',frame = 'icrs',obstime='J2016.0')

gaia_off_ra, gaia_off_dec = center_arc.spherical_offsets_to(gaia_coord.frame)
gaia_off_ra = gaia_off_ra.to(u.arcsec)
gaia_off_dec = gaia_off_dec.to(u.arcsec)

if choosen_cluster == 'Quintuplet':
    gns_clus = [10,2,4,3]
elif choosen_cluster == 'Arches':    
    gns_clus = [7,4,7,1]
field_one = gns_clus[0]
chip_one = gns_clus[1]
field_two = gns_clus[2]
chip_two = gns_clus[3]
GNS_1off='/Users/amartinez/Desktop/PhD/HAWK/GNS_1off/lists/%s/chip%s/'%(field_one, chip_one)
GNS_2off='/Users/amartinez/Desktop/PhD/HAWK/GNS_2off/lists/%s/chip%s/'%(field_two, chip_two)


if ref_frame == 'absolute':
    dmax = 100 #TODO
    # x_dis  0,y_dis 1,dvx 2,dvy 3,x1 4,y1 5,x2 6,y2 7,H1 8,dH1 9,Ks1 10,dKs1 11,H2 12,dH2 13,RaH1 14,DecH1 15,raH2 16,decH2 17
    gns = np.loadtxt(pm_abs + 'pm_GaiaRF_ep1_f%sc%s_ep2_f%sc%sdeg%s_dmax%s_sxy%s.txt'%(field_one, chip_one, field_two, chip_two,align_degree,dmax,max_sig))
elif ref_frame == 'relative':
    dmax = 1#TODO
    # x_dis  0,y_dis 1,dvx 2,dvy 3,x1 4,y1 5,x2 6,y2 7,H1 8,dH1 9,Ks1 10,dKs1 11,H2 12,dH2 13,RaH1 14,DecH1 15,raH2 16,decH2 17
    gns = np.loadtxt(pm_rel + 'pm_ep1_f%sc%s_ep2_f%sc%sdeg%s_dmax%s_sxy%s.txt'%(field_one, chip_one, field_two, chip_two,align_degree,dmax,max_sig))

# unc_cut = np.where((gns[:,2]<vel_cut) &(gns[:,3]<vel_cut))
# gns = gns[unc_cut]
# mag_cut =np.where((gns[:,12]<18) & (gns[:,12]>12))
# gns=gns[mag_cut]
# color_cut =np.where((gns[:,8]- gns[:,10])>1.3)
# gns=gns[color_cut]

# %%
fig, ax = plt.subplots(1,1,figsize = (10,10))
ax.scatter(arches['RA'],arches['DEC'],alpha = 0.1, label = 'Hosek [Arches]')
ax.scatter(gns[:,-2],gns[:,-1], alpha = 0.1, label = 'GNS')
ax.scatter(gaia_coord.ra, gaia_coord.dec, color = 'r',s = 100, label = 'Gaia')
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
leg = ax.legend()
for lh in leg.legendHandles:
    lh.set_alpha(1)
    lh._sizes = [100]
# ax.legend()
# ax.scatter(arches[:,7], arches[:,9])
# ax.scatter(gaia_off_ra, gaia_off_dec)
# %%
# Now we look for the Gaia stras in Hosek distributions

max_sep = 0.08*u.arcsec#TODO
idx,d2d,d3d = gaia_coord.match_to_catalog_sky(hose_coord,nthneighbor=1)# ,nthneighbor=1 is for 1-to-1 match
sep_constraint = d2d < max_sep
hose_match = arches.iloc[idx[sep_constraint]]
gaia_match_hos = gaia_good[sep_constraint]
# %%
RA_match = hose_match['RA']
DEC_match = hose_match['DEC']

d_pmra = gaia_match_hos['pmra'] - np.array(hose_match['pmRA'])
d_pmdec = gaia_match_hos['pmdec'] -np.array(hose_match['pmDE'])
d_pmra = np.array(d_pmra)
d_pmdec = np.array(d_pmdec)

d_ra = gaia_match_hos['ra'] - np.array(hose_match['RA'])
d_dec = gaia_match_hos['dec'] - np.array(hose_match['DEC'])
d_ra = np.array(d_ra)*3600
d_dec = np.array(d_dec)*3600

# fig, ax = plt.subplots(1,1,figsize = (10,10))
# ax.scatter(gaia_match_hos['ra'],gaia_match_hos['dec'])
# ax.scatter(RA_match,DEC_match)

# %%
fig, ax = plt.subplots(1,2,figsize = (20,10))
ax[0].set_title('%s Gaia stars'%(len(gaia_match_hos)))
ax[0].hist(d_pmra, label = '$\overline{\Delta \mu_{dec}}$ = %.1f \nsig = %.2f'%(np.mean(d_pmra),np.std(d_pmra)))
ax[0].hist(d_pmdec,alpha =0.7, label = '$\overline{\Delta \mu_{dec}}$ = %.2f \nsig = %.2f'%(np.mean(d_pmdec),np.std(d_pmdec)))
ax[0].set_xlim(-1,1)
ax[1].set_title('HOSEK & Gaia')
ax[1].hist(d_ra, label = '$\overline{\Delta ra}$ = %.2f \nsig = %.2f'%(np.mean(d_ra),np.std(d_ra)))
ax[1].hist(d_dec, alpha =0.7,label = '$\overline{\Delta dec}$  = %.2f \nsig = %.2f'%(np.mean(d_dec),np.std(d_dec)))
ax[1].set_xlim(-0.2,0.2)
ax[0].set_xlabel('$\Delta\mu$ (mas/yr)')
ax[1].set_xlabel('$\Delta position$ (arcsec)')

ax[0].legend() 
ax[1].legend() 
ax[0].set_xlim(-3,3)
# %%

# We can use GNS1 or GNS2 coordinates (seven year difference)
gns_coord = SkyCoord(ra = gns[:,-4], dec = gns[:,-3], unit = 'degree', frame ='fk5', obstime = 'J2015.4')
# gns_coord = SkyCoord(ra = gns[:,-2], dec = gns[:,-1], unit = 'degree', frame ='fk5', obstime = 'J2022.4')


# Here we are going to try with the Gaia stars desplaced to the GNS2 epoch
# We load this list that already has the Gaia position moved to GNS2 time
# and selected the good Gaia stars
# 'ra 0, dec 1, dra(mas) 2, ddec(mas) 3, x 4, y 5, pmra(mas/yr) 6, pmdec 7, dpmra(mas/yr) 8, dpmdec 9,dradec 10' )
# gaia = np.loadtxt(GNS_2off + 'ALL_gaia_refstars_on_gns2_f%sc%s_gns1_f%sc%s.txt'%(field_two,chip_two,field_one,chip_one)) 
# gaia_coord =  SkyCoord(ra=gaia[:,0], dec=gaia[:,1], unit = 'degree',frame = 'icrs',obstime='J2022.4')


# max_sep = 0.20*u.arcsec
idx,d2d,d3d = gaia_coord.match_to_catalog_sky(gns_coord,nthneighbor=1)# ,nthneighbor=1 is for 1-to-1 match
sep_constraint = d2d < max_sep
gaia_match_gns = gaia_good[sep_constraint]
gns_match = gns[idx[sep_constraint]]

d_pmra = gaia_match_gns['pmra'] - gns_match[:,0]
d_pmdec = gaia_match_gns['pmdec'] - gns_match[:,1]
d_pmra = np.array(d_pmra)
d_pmdec = np.array(d_pmdec)

d_ra = gaia_match_gns['ra'] - gns_match[:,-2]
d_dec = gaia_match_gns['dec'] - gns_match[:,-1]
d_ra = np.array(d_ra)*3600
d_dec = np.array(d_dec)*3600


fig, ax = plt.subplots(1,2,figsize = (20,10))
ax[0].set_title('%s Gaia stars. Degree = %s'%(len(gns_match), align_degree))
ax[0].hist(d_pmra, histtype = 'step',linewidth = 4, label = '$\overline{\Delta \mu_{dec}}$ = %.2f \nsig = %.2f'%(np.mean(d_pmra),np.std(d_pmra)))
ax[0].hist(d_pmdec, histtype = 'step',linewidth = 4, label = '$\overline{\Delta \mu_{dec}}$ = %.2f \nsig = %.2f'%(np.mean(d_pmdec),np.std(d_pmdec)))
ax[0].legend()
ax[0].set_xlabel('$\Delta\mu$ (mas/yr)')
ax[0].set_xlim(-3,3)

ax[1].set_title('GNS & Gaia')
ax[1].hist(d_ra, histtype = 'step',linewidth = 4, label = '$\overline{\Delta ra}$  = %.2f \nsig = %.2f'%(np.mean(d_ra),np.std(d_ra)))
ax[1].hist(d_dec, histtype = 'step',linewidth = 4, label = '$\overline{\Delta dec}$ = %.2f \nsig = %.2f'%(np.mean(d_dec),np.std(d_dec)))
ax[1].legend()
ax[1].set_xlabel('$\Delta position$ (arcsec)')
ax[1].set_xlim(-0.2,0.2)

# %%
# Now we look for matches between gns´ gaia stars and Hosek´s gaia stars

gns_match_coor = SkyCoord(ra = gns_match[:,-4], dec = gns_match[:,-3], 
                          unit = 'degree', frame = 'fk5', obstime = 'J2015.4')
hos_match_coor = SkyCoord(ra = RA_match, dec = DEC_match, unit = 'degree')
# %
max_sep = 0.08*u.arcsec
idx,d2d,d3d = hos_match_coor.match_to_catalog_sky(gns_match_coor,nthneighbor=1)# ,nthneighbor=1 is for 1-to-1 match
sep_constraint = d2d < max_sep
# %
gns_match_hos = gns_match[idx[sep_constraint]]
hos_match_gns = hose_match[sep_constraint]
# %
RA_hos_gns = RA_match[sep_constraint]
DEC_hos_gns = DEC_match[sep_constraint]



d_pmra_hg = hos_match_gns['pmRA'] - gns_match_hos[:,0]
d_pmdec_hg = hos_match_gns['pmDE'] - gns_match_hos[:,1]

d_ra_hg = (RA_hos_gns - gns_match_hos[:,-4])*3600
d_dec_hg = (DEC_hos_gns - gns_match_hos[:,-3])*3600



fig, ax = plt.subplots(1,2,figsize = (20,10))
ax[0].set_title('Gaia-Hosek & Gaia-GNS')
ax[1].set_title('%s Common Gaia´s stars'%(len(gns_match_hos)))
ax[0].hist(d_pmra_hg, histtype = 'step',linewidth = 4, label = '$\overline{\Delta \mu_{dec}}$ = %.2f \nsig = %.2f'%(np.mean(d_pmra_hg),np.std(d_pmra_hg)))
ax[0].hist(d_pmdec_hg, histtype = 'step',linewidth = 4, label = '$\overline{\Delta \mu_{dec}}$ = %.2f \nsig = %.2f'%(np.mean(d_pmdec_hg),np.std(d_pmdec_hg)))
ax[0].legend()
ax[0].set_xlabel('$\Delta\mu$ (mas/yr)')
ax[0].set_xlim(-3,3)

# ax[1].set_title('%s Gaia stars'%(len(gns_match)))
ax[1].hist(d_ra_hg, histtype = 'step',linewidth = 4, label = '$\overline{\Delta ra}$  = %.2f \nsig = %.2f'%(np.mean(d_ra_hg),np.std(d_ra_hg)))
ax[1].hist(d_dec_hg, histtype = 'step',linewidth = 4, label = '$\overline{\Delta dec}$ = %.2f \nsig = %.2f'%(np.mean(d_dec_hg),np.std(d_dec_hg)))
ax[1].legend()
ax[1].set_xlabel('$\Delta position$ (arcsec)')
ax[1].set_xlim(-0.2,0.2)


# %%
fig, ax = plt.subplots(1,2,figsize = (20,10))
ax[0].set_title('Gaia-Hosek & Gaia-GNS')
ax[0].scatter(d_pmra_hg, d_pmdec_hg, s = 100)
ax[1].scatter(d_ra_hg, d_dec_hg,s =100)
ax[0].set_xlim(-3,3)
ax[0].set_ylim(-3,3)
ax[0].grid()
ax[0].set_xlabel('$\Delta\mu_{ra}$ (mas/yr)')
ax[0].set_ylabel('$\Delta\mu_{dec}$ (mas/yr)')
ax[0].axvline(np.mean(d_pmra_hg), color = 'r', label = 'mean = %.3f'%(np.mean(d_pmra_hg)))
ax[0].axvline(np.mean(d_pmra_hg) + np.std(d_pmra_hg), color = 'r', linestyle = 'dashed',label = '1$\sigma$ = %.3f'%(np.std(d_pmra_hg)))
ax[0].axvline(np.mean(d_pmra_hg) - np.std(d_pmra_hg), color = 'r', linestyle = 'dashed')
ax[0].axhline(np.mean(d_pmdec_hg), color = 'r', label = 'mean = %.3f'%(np.mean(d_pmra_hg)))
ax[0].axhline(np.mean(d_pmdec_hg) + np.std(d_pmdec_hg), color = 'r', linestyle = 'dashed',label = '1$\sigma$ = %.3f'%(np.std(d_pmdec_hg)))
ax[0].axhline(np.mean(d_pmdec_hg) - np.std(d_pmdec_hg), color = 'r', linestyle = 'dashed')

ax[1].set_xlim(-0.2,0.2)
ax[1].set_ylim(-0.2,0.2)
ax[1].axvline(np.mean(d_ra_hg), color = 'r', label = 'mean = %.3f'%(np.mean(d_pmra_hg)))
ax[1].axvline(np.mean(d_ra_hg) + np.std(d_ra_hg), color = 'r', linestyle = 'dashed',label = '1$\sigma$ = %.3f'%(np.std(d_ra_hg)))
ax[1].axvline(np.mean(d_ra_hg) - np.std(d_ra_hg), color = 'r', linestyle = 'dashed')
ax[1].axhline(np.mean(d_dec_hg), color = 'r', label = 'mean = %.3f'%(np.mean(d_pmra_hg)))
ax[1].axhline(np.mean(d_dec_hg) + np.std(d_dec_hg), color = 'r', linestyle = 'dashed',label = '1$\sigma$ = %.3f'%(np.std(d_dec_hg)))
ax[1].axhline(np.mean(d_dec_hg) - np.std(d_dec_hg), color = 'r', linestyle = 'dashed')

ax[1].grid()
ax[1].set_xlabel('$\Delta Ra$ (arcsec)')
ax[1].set_ylabel('$\Delta Dec$ (arcsec)')










