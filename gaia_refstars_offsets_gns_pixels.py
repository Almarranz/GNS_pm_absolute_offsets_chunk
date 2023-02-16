#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 10:30:27 2022

@author: amartinez
"""

# Generates offsets for Gaia stars and astroaligned with xy coordinates in GNS1

# Here we are going to align GNS (1 and 2) to Gaia reference frame for the 
# each of tha gns epochs
import numpy as np
from astropy import units as u
from astropy.time import Time
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import sys
from matplotlib import rcParams
from astroquery.gaia import Gaia
import astroalign as aa
import time
from astropy.coordinates import match_coordinates_sky
from compare_lists import compare_lists
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
# =============================================================================
# Gaia data:https://gea.esac.esa.int/archive/
# =============================================================================
# Load coordinates for the gns1 pointing
field_one = 7
chip_one = 4
field_two = 7
chip_two = 1
GNS_1='/Users/amartinez/Desktop/PhD/HAWK/GNS_1/lists/%s/chip%s/'%(field_one, chip_one)
GNS_2='/Users/amartinez/Desktop/PhD/HAWK/GNS_2/lists/%s/chip%s/'%(field_two, chip_two)
GNS_1off='/Users/amartinez/Desktop/PhD/HAWK/GNS_1off_chunk/lists/%s/chip%s/'%(field_one, chip_one)
GNS_2off='/Users/amartinez/Desktop/PhD/HAWK/GNS_2off_chunk/lists/%s/chip%s/'%(field_two, chip_two)
# gns1_im = GNS_1 +  'field%s_H_gns1.fits'%(field_one)
gns1_im = GNS_1 +   'calibrated_stars_%s.fits'%(chip_one)
gns2_im = GNS_2 + '%s_chip%s_holo_cal.fits'%(field_two,chip_two)
pruebas2 = '/Users/amartinez/Desktop/PhD/HAWK/GNS_2off_chunk/pruebas/'
pruebas1 = '/Users/amartinez/Desktop/PhD/HAWK/GNS_1off_chunk/pruebas/'
gaia_list1 = '/Users/amartinez/Desktop/PhD/HAWK/GNS_1off_chunk/Gaia_lists/'
gaia_list2 = '/Users/amartinez/Desktop/PhD/HAWK/GNS_2off_chunk/Gaia_lists/'

max_sig = 0.1#TODO
# max_sig = 1#TODO
# bad1 = []
# bad2 = []
bad1 =[11]
bad2 = [11]
np.savetxt(pruebas1 + 'bad1_f%sc%s.txt'%(field_one,chip_one),np.array(bad1).T, fmt='%.i')
np.savetxt(pruebas2 + 'bad2_f%sc%s.txt'%(field_two,chip_two),np.array(bad1).T, fmt='%.i')

if len(bad1) >0:
    clip_bad1 = 'yes'#TODO
else:
    clip_bad1 = 'no'#TODO
# ra1 0, dec1 1, x1 2, y1 3, f1 4, H1 5, dx1 6, dy1 7, df1 8, dH1 9 ,Ks 10, dKs 11
gns1 = np.loadtxt(GNS_1off +'stars_calibrated_HK_chip%s_on_gns2_f%sc%s_sxy%s.txt'%(chip_one,field_two,chip_two,max_sig))

# WARNING: we have to multiply the x coordinates by -1 for aa to be able to find 
# the tranformation between GAIA (ra,dec) offsets and GNS (x,y) coordinates 
gns1[:,2] = gns1[:,2]*-1
# np.savetxt(GNS_1 +'stars_calibrated_HK_chip%s_sxy%s.txt'%(chip_one,max_sig),gns1, fmt ='%.8f',header='x, dx, y, dy, raH, draH, decH, ddecH, mJ, dmJ, mH, dmH, mK, dmK')
gns1_coor =  SkyCoord(ra=gns1[:,0], dec=gns1[:,1],unit = 'degree' ,frame = 'fk5',equinox ='J2000',obstime='J2015.4')
gns1_coor =gns1_coor.transform_to('icrs')


# Here we are going get the informations in the header of GNS1
f =fits.open(gns1_im)
w = WCS(f[0].header)    
header = fits.getheader(gns1_im)
# x_cent, y_cent =header['NAXIS1']/2,header['NAXIS2']/2
x_cent, y_cent = np.mean(gns1[:,0]), np.mean(gns1[:,1])
radec_ = SkyCoord(ra = x_cent, dec = y_cent, unit = 'degree', frame = 'fk5', equinox = 'J2000', obstime ='J2015.43')
with open(pruebas1 + 'ZP_centroid_gns1_f%sc%s.reg'%(field_one, chip_one),'w') as file:
    file.write('# Region file format: DS9 version 4.1'+"\n"+'global color=blue dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1'+"\n"+'fk5'+'\n')
with open(pruebas1 + 'ZP_centroid_gns1_f%sc%s.reg'%(field_one, chip_one),'a') as file:
    file.write('circle(%s,%s,0.5")'%(x_cent, y_cent))




if field_one == 7 or field_one == 12 or field_one == 10:
    t_gns1 = Time(['2015-06-07T00:00:00','2016-01-01T00:00:00'],scale='utc')

if field_two == 7:
    t_gns2 = Time(['2022-05-27T00:00:00','2016-01-01T00:00:00'],scale='utc')
if field_two == 4:
    t_gns2 = Time(['2022-04-05T00:00:00','2016-01-01T00:00:00'],scale='utc')

Gaia.MAIN_GAIA_TABLE = "gaiaedr3.gaia_source" # Select early Data Release 3
Gaia.ROW_LIMIT = -1  # Ensure the default row limit.
coord = SkyCoord(ra=radec_.ra, dec=radec_.dec, unit='degree', frame='icrs',equinox ='J2000',obstime='J2016.0')


ancho2 = header['NAXIS1']
alto2 = header['NAXIS2']
width2 = ancho2*0.053*u.arcsec
height2 = alto2*0.053*u.arcsec

j = Gaia.cone_search_async(coord, np.sqrt((width2)**2+height2**2)/2)
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
gaia_coord =  SkyCoord(ra=gaia_good['ra'], dec=gaia_good['dec'],frame = 'icrs',obstime='J2016.0')

# %%
delta_t1 = t_gns1[0] - t_gns1[1]
delta_t2 = t_gns2[0] - t_gns2[1]


GaiaCoord = SkyCoord(ra=gaia_good['ra'],
                   dec=gaia_good['dec'],
                   frame='icrs',
                     equinox = 'J2000',
                    obstime='J2016.0')

# Asigns Offsets coordinates to Gaia stars, moves then  and return the
# corresponding ra dec coordenates using spherical_offsets_by method
radec_ = radec_.transform_to('icrs')
ragai1_off,decgai1_off = radec_.spherical_offsets_to(GaiaCoord.frame)
ragai1_off = (ragai1_off.to(u.mas)).value + (np.array(gaia_good['pmra'])*delta_t1.to(u.yr)).value
decgai1_off = (decgai1_off.to(u.mas)).value + (np.array(gaia_good['pmdec'])*delta_t1.to(u.yr)).value
GaiaGNSCoord1 = radec_.spherical_offsets_by(ragai1_off*u.mas, decgai1_off*u.mas)

# Propagation of error in gaia
dx_des = np.sqrt(gaia_good['ra_error']**2 + (delta_t1.to(u.yr).value*gaia_good['pmra_error'])**2)
dy_des = np.sqrt(gaia_good['dec_error']**2 + (delta_t1.to(u.yr).value*gaia_good['pmra_error'])**2)
dxy_des = np.sqrt(dx_des**2 + dy_des**2) 
# %%
# Here we are going to cut the Gaia stars over the area of GNS1
gaia_coord_gal = gaia_coord.galactic
gns1_coord_gal = gns1_coor.galactic

lg = gaia_coord_gal.l.wrap_at('360d')
l1 = gns1_coord_gal.l.wrap_at('360d')


# buenos1 = np.where((gaia_coord_gal.l>min(gns1_coord_gal.l)) & (gaia_coord_gal.l<max(gns1_coord_gal.l)) &
#                    (gaia_coord_gal.b>min(gns1_coord_gal.b)) & (gaia_coord_gal.b<max(gns1_coord_gal.b)))

l_cut = 0.117#TODO
buenos1 = np.where((gaia_coord_gal.l>l_cut*u.degree) & (gaia_coord_gal.l<max(gns1_coord_gal.l)) &
                   (gaia_coord_gal.b>min(gns1_coord_gal.b)) & (gaia_coord_gal.b<max(gns1_coord_gal.b)))

fig, ax = plt.subplots(1,1,figsize =(10,10))
ax.scatter(lg, gaia_coord_gal.b, label ='Gaia stars')
ax.axvline(l_cut, color = 'red')
b_size = max(gns1_coord_gal.b).value - min(gns1_coord_gal.b).value
l_zize = max(gns1_coord_gal.l).value - l_cut
chunk = np.where(l1>l_cut*u.degree)
ax.scatter(l1[chunk], gns1_coord_gal.b[chunk], label = 'GNS1',alpha = 1, color = 'orange' ) 
ax.scatter(l1, gns1_coord_gal.b,alpha = 0.05, color = 'orange')  
ax.scatter(lg[buenos1], gaia_coord_gal[buenos1].b, label ='Gaia over GNS1', color = 'red')


ax.set_xlabel('l(deg)')
ax.set_ylabel('b(deg)')
ax.legend()

# %%
# Here we are going to cut GNS1 area
gns1 = gns1[chunk]
gns1_coor = gns1_coor[chunk]
gns1_coord_gal = gns1_coord_gal[chunk]
# %%


# Each gaia stars get its own ID
ga1_id = np.arange(len(gaia_good[buenos1]))
gaia_np1 =np.array([GaiaGNSCoord1[buenos1].ra.value, GaiaGNSCoord1[buenos1].dec.value,
                     dx_des[buenos1], dy_des[buenos1],
                     ragai1_off[buenos1], decgai1_off[buenos1],
                     np.array(gaia_good['pmra'][buenos1]), np.array(gaia_good['pmdec'][buenos1]),
                     gaia_good['pmra_error'][buenos1].value, gaia_good['pmdec_error'][buenos1].value,
                     ga1_id]).T
# Saves two differnte lists: one with All gaia stars and the other after the 
# 3sigma clipping done when comparing with Gaia.
np.savetxt(GNS_1off + 'ALL_gaia_refstars_on_gns1_f%sc%s.txt'%(field_one,chip_one), gaia_np1, fmt ='%.8f', 
           header = 'ra, dec, dra(mas), ddec(mas), x, y, pmra(mas/yr), pmdec, dpmra(mas/yr), dpmdec,dradec')
with open(gaia_list1+ 'gaia_gns1_f%sc%s_on_gns2_f%sc%s_sxy%s.reg'%(field_one,chip_one,field_two,chip_two,max_sig), 'w') as f:
    f.write('# Region file format: DS9 version 4.1'+"\n"+'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1'+"\n"+'fk5'+'\n')
    f.close
for gs in range(len(ga1_id)):
    with open(gaia_list1+ 'gaia_gns1_f%sc%s_on_gns2_f%sc%s_sxy%s.reg'%(field_one,chip_one,field_two,chip_two,max_sig), 'a') as f:
        f.write('circle(%s,%s,0.5") \n point(%s,%s) # point=cross \n # text(%s,%s) font="helvetica 24 normal roman" text={%.0f} \n'%(gaia_np1[gs][0], gaia_np1[gs][1],
                                                                        gaia_np1[gs][0], gaia_np1[gs][1],
                                                                        gaia_np1[gs][0]+0.00013, gaia_np1[gs][1]+0.00013,
                                                                        gs))

if clip_bad1 == 'yes':#TODO
    gaia_np1 = np.delete(gaia_np1,bad1, axis =0)
np.savetxt(GNS_1off + 'gaia_refstars_on_gns1_f%sc%s.txt'%(field_one,chip_one), gaia_np1, fmt ='%.8f', 
           header = 'ra, dec, dra(mas), ddec(mas), x, y, pmra(mas/yr), pmdec, dpmra(mas/yr), dpmdec,dradec')


ID_gns1 = np.arange(len(gns1))

# with open(pruebas1+ 'gns1_around_gaiaf%sc%s_sxy%s.reg'%(field_one,chip_one,max_sig), 'w') as f:
#     f.write('# Region file format: DS9 version 4.1'+"\n"+'global color=blue dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1'+"\n"+'fk5'+'\n')
#     f.close

# for gst in range(len(gaia_np1)):
#     m_point = SkyCoord(ra =[gaia_np1[gst][0]], dec  =[gaia_np1[gst][1]],unit ='degree', frame ='icrs', equinox = 'J2000', obstime = 'J2015.43')
#     idx, d2d, d3d = match_coordinates_sky(m_point, gns1_coor)
#     # idxc, group_md, d2d,d3d =  ap_coord.search_around_sky( m_point ,gns1_coor, 0.6*u.arcsec)
#     with open(pruebas1+ 'gns1_around_gaiaf%sc%s_sxy%s.reg'%(field_one,chip_one,max_sig), 'a') as f:
#         f.write('point(%s,%s) # point = cross \n # text(%s,%s) text={%.0f} \n'%(float(gns1_coor.ra.value[idx]) ,float(gns1_coor.dec.value[idx]),
#                                                                                 float(gns1_coor.ra.value[idx]) ,float(gns1_coor.dec.value[idx]),
#                                                                                int(ID_gns1[idx])))
#         f.close 
        
# Asigns Offsets coordinates to Gaia stars, moves then (epoch 2) and return the
# corresponding ra dec coordenates using spherical_offsets_by method
radec_ = radec_.transform_to('icrs')
ragai2_off,decgai2_off = radec_.spherical_offsets_to(GaiaCoord.frame)
ragai2_off = (ragai2_off.to(u.mas)).value + (np.array(gaia_good['pmra'])*delta_t2.to(u.yr)).value
decgai2_off = (decgai2_off.to(u.mas)).value + (np.array(gaia_good['pmdec'])*delta_t2.to(u.yr)).value
GaiaGNSCoord2 = radec_.spherical_offsets_by(ragai2_off*u.mas, decgai2_off*u.mas)



dx_des = np.sqrt(gaia_good['ra_error']**2 + (delta_t2.to(u.yr).value*gaia_good['pmra_error'])**2)
dy_des = np.sqrt(gaia_good['dec_error']**2 + (delta_t2.to(u.yr).value*gaia_good['pmra_error'])**2)
dxy_des = np.sqrt(dx_des**2 + dy_des**2) 
ga2_id = np.arange(len(gaia_good))
# %%
# Here we are going to select GNS2 over GNS1

gns2 = np.loadtxt(GNS_2off + 'stars_calibrated_H_chip%s_on_gns1_f%sc%s_sxy%s.txt'%(chip_two,field_one,chip_one,max_sig))
gns2[:,2] = gns2[:,2]*-1#TODO
# gns2[:,3] = gns2[:,3]*-1#TODO

gns2_coor = SkyCoord(ra= gns2[:,0]*u.degree, dec=gns2[:,1]*u.degree, frame = 'fk5', equinox = 'J2000',obstime='J2022.4')
gns2_coor = gns2_coor.transform_to('icrs')
gaia_coord_gal = gaia_coord.galactic
gns2_coord_gal = gns2_coor.galactic



lg = gaia_coord_gal.l.wrap_at('360d')
l2 = gns2_coord_gal.l.wrap_at('360d')


chunk2 = np.where((gns2_coord_gal.l>min(gns1_coord_gal.l)) & (gns2_coord_gal.l<max(gns1_coord_gal.l)) &
                   (gns2_coord_gal.b>min(gns1_coord_gal.b)) & (gns2_coord_gal.b<max(gns1_coord_gal.b)))




# 

buenos2 = np.where((gaia_coord_gal.l>min(gns2_coord_gal[chunk2].l))&((gaia_coord_gal.l<max(gns2_coord_gal[chunk2].l)))
                   &(gaia_coord_gal.b>min(gns2_coord_gal[chunk2].b))&(gaia_coord_gal.b<max(gns2_coord_gal[chunk2].b)))
ga2_id = np.arange(len(gaia_good[buenos2]))

gaia_np2 = np.array([GaiaGNSCoord2[buenos2].ra.value,GaiaGNSCoord2[buenos2].dec.value,
                   dx_des[buenos2],dy_des[buenos2],
                   ragai2_off[buenos2], decgai2_off[buenos2],
                   np.array(gaia_good['pmra'][buenos2]), np.array(gaia_good['pmdec'][buenos2]),
                   gaia_good['pmra_error'][buenos2].value,gaia_good['pmdec_error'][buenos2].value,
                   ga2_id]).T

fig, ax = plt.subplots(1,1, figsize =(10,10))
ax.scatter(gns2[:,0],gns2[:,1]) 
ax.scatter(gns2[:,0][chunk2],gns2[:,1][chunk2],label ='GNS2 over GNS1 chunk') 
ax.set_xlabel('Ra(deg)')
ax.set_ylabel('Dec(deg)')
ax.scatter(GaiaGNSCoord2[buenos2].ra.value,GaiaGNSCoord2[buenos2].dec.value, label ='Gaia over chunk')
ax.legend()

# We selecte only gns2 over the chunk
gns2 = gns2[chunk2]


np.savetxt(GNS_2off + 'ALL_gaia_refstars_on_gns2_f%sc%s_gns1_f%sc%s.txt'%(field_two,chip_two,field_one,chip_one), gaia_np2, fmt ='%.8f', 
          header = 'ra, dec, dra(mas), ddec(mas), x, y, pmra(mas/yr), pmdec, dpmra(mas/yr), dpmdec,dradec')

with open(gaia_list2+ 'gaia_gns2_f%sc%s_on_gns_f%sc%s_sxy%s.reg'%(field_two,chip_two,field_one,chip_one,max_sig), 'w') as f:
    f.write('# Region file format: DS9 version 4.1'+"\n"+'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1'+"\n"+'fk5'+'\n')
    f.close
for gs in range(len(gaia_np2)):
    with open(gaia_list2+ 'gaia_gns2_f%sc%s_on_gns_f%sc%s_sxy%s.reg'%(field_two,chip_two,field_one,chip_one,max_sig), 'a') as f:
        f.write('circle(%s,%s,0.5") \n point(%s,%s) # point=cross \n # text(%s,%s) font="helvetica 24 normal roman" text={%.0f} \n'%(gaia_np2[gs][0], gaia_np2[gs][1],
                                                                        gaia_np2[gs][0], gaia_np2[gs][1],
                                                                        gaia_np2[gs][0]+0.00013, gaia_np2[gs][1]+0.00013,
                                                                        gs))

if len(bad2) >0:
    clip_bad2 = 'yes'#TODO
else:
    clip_bad2 = 'no'#TODO                 
if clip_bad2 == 'yes':#TODO
    gaia_np2 = np.delete(gaia_np2,bad2, axis =0)                    
np.savetxt(GNS_2off + 'gaia_refstars_on_gns2_f%sc%s_gns1_f%sc%s.txt'%(field_two,chip_two,field_one,chip_one), gaia_np2, fmt ='%.8f', 
          header = 'ra, dec, dra(mas), ddec(mas), x, y, pmra(mas/yr), pmdec, dpmra(mas/yr), dpmdec,dradec')


# %%
# We select GNS1 foregroud stars and astroaling their pixels coordenates with 
# the Gaia stars offsets

# ra1 0, dec1 1, x1 2, y1 3, f1 4, H1 5, dx1 6, dy1 7, df1 8, dH1 9 ,Ks 10, dKs 11

fg = np.where(((gns1[:,5]-gns1[:,10])<1.3) & (np.sqrt(gns1[:,6]**2 + gns1[:,7]**2) <np.mean([gns1[:,6],gns1[:,7]])))

# gns1_fg = gns1[fg]
gns1_fg = gns1
# We cross matchh the stars in gns1 with Gaia in order to help astroalign 
# to find the trasnformation

max_sep = 0.080*u.arcsec
gaia_coord = SkyCoord(ra = gaia_np1[:,0], dec = gaia_np1[:,1], unit = 'degree',frame = 'icrs',obstime='J2016.0' )
gns1_coor_match = SkyCoord(ra= gns1_fg[:,0]*u.degree, dec=gns1_fg[:,1]*u.degree, frame = 'fk5',equinox = 'J2000',obstime='J2015.43')
gns1_coor_match = gns1_coor_match.transform_to('icrs')
idx,d2d,d3d = gaia_coord.match_to_catalog_sky(gns1_coor_match,nthneighbor=1)# ,nthneighbor=1 is for 1-to-1 match
sep_constraint = d2d < max_sep
gaia_match = gaia_np1[sep_constraint]
gns1_match = gns1_fg[idx[sep_constraint]]


# Transform GNS xy coordinates to mas
x_mas = gns1_match[:,2]*0.5*0.106*1000
y_mas = gns1_match[:,3]*0.5*0.106*1000

xy_mas = np.array([x_mas,y_mas]).T

fig, ax = plt.subplots(1,1,figsize=(10,10))
ax.set_title('GNS1 xy and Gaia radec_offsets (%s matching stars)'%(len(x_mas)))
ax.scatter(xy_mas[:,0], xy_mas[:,1], label = 'GNS1 xy (mas)')
ax.scatter(gaia_match[:,4],gaia_match[:,5], label = 'Gaia1 offsets (mas)')
ax.legend() 


max_offset = 1
check_x,check_y=max_offset+1,max_offset+1
while abs(check_x) >max_offset or abs(check_y)>max_offset  :# only when tranformation gets better than 1 chip desplacement the coordinates are stored

    print('starting aa')
    print('Stars in transformed list %s'%(len(gns1_match)))
    tic = time.perf_counter()
    m,(_,_)= aa.find_transform(xy_mas,gaia_match[:,4:6],max_control_points=3)
    # m,(_,_)= aa.find_transform(gns1_match[:,0:2],gaia_np1[:,0:2],max_control_points=200)
    
    # m,(_,_)= aa.find_transform(dic_gns['gns_c%s'%(i)],dic_vvv['vvv_c%s'%(i)],max_control_points=400)
    print("Translation: (x, y) = (%.2f, %.2f)"%(m.translation[0],m.translation[1]))
    print("Rotation: %.1f degrees"%(m.rotation * 180.0 /np.pi)) 
    # print("Rotation: %s"%(m.rotation))
    print("Scale factor: %.4f"%(m.scale))
    toc = time.perf_counter()
    print('it took %.2f seconds'%(toc-tic))
    
    test_gns = aa.matrix_transform(xy_mas, m.params)
    
    
    
    print(20*'#'+'\n'+'CHECKING GNS1'+'\n'+20*'#')
    tic = time.perf_counter()
    t,(_,_)= aa.find_transform(test_gns,gaia_match[:,4:6],max_control_points=3)
    print("Translation: (x, y) = (%.2f, %.2f)"%(t.translation[0],t.translation[1]))
    print("Rotation: %.1f degrees"%(t.rotation * 180.0 /np.pi)) 

    # print("Rotation: %.1f arcsec"%(t.rotation * 180.0*60*60 /np.pi)) 
    print("Scale factor: %.4f"%(t.scale))
    toc = time.perf_counter()
    print('checking took %.2f seconds'%(toc-tic))
    check_x= t.translation[0]
    check_y= t.translation[1]
# %%
fig, ax = plt.subplots(1,1,figsize =(10,10))
ax.set_title('Gaia and tranformed GNS1')
ax.scatter(gaia_match[:,4],gaia_match[:,5],s=100, label ='Gaia1 offsets (mas)')
ax.scatter(test_gns[:,0],test_gns[:,1],s=20, label = 'GNS1 aa (mas)\nRot = %.1fº \n$Tras_{(x, y)}$ = (%.0f, %.0f)'%(m.rotation * 180.0 /np.pi,m.translation[0],m.translation[1]))
ax.legend()
gns1[:,2:4] = aa.matrix_transform(gns1[:,2:4]*0.5*0.106*1000, m.params)
np.savetxt(GNS_1off + 'aa_stars_calibrated_HK_chip%s_on_gns2_f%sc%s_sxy%s.txt'%(chip_one,field_two,chip_two,max_sig),gns1,
           fmt = '%.8f', header = 'ra1 0, dec1 1, x1 2, y1 3, f1 4, H1 5, dx1 6, dy1 7, df1 8, dH1 9 ,Ks 10, dKs 11')

fig, ax = plt.subplots(1,1, figsize=(10,10))
ax.set_title('GNS1 in Gaia ref frame (icrs)')
ax.scatter(gns1[:,2],gns1[:,3])

# %%
# Makes the alignment with GNS2
with open(gaia_list2+ 'gaia_gns2_f%sc%s_on_gns_f%sc%s_sxy%s.reg'%(field_two,chip_two,field_one,chip_one,max_sig), 'w') as f:
    f.write('# Region file format: DS9 version 4.1'+"\n"+'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1'+"\n"+'fk5'+'\n')
    f.close
for gs in range(len(gaia_np2)):
    with open(gaia_list2+ 'gaia_gns2_f%sc%s_on_gns_f%sc%s_sxy%s.reg'%(field_two,chip_two,field_one,chip_one,max_sig), 'a') as f:
        f.write('circle(%s,%s,0.5") \n point(%s,%s) # point=cross \n # text(%s,%s) font="helvetica 24 normal roman" text={%.0f} \n'%(gaia_np2[gs][0], gaia_np2[gs][1],
                                                                        gaia_np2[gs][0], gaia_np2[gs][1],
                                                                        gaia_np2[gs][0]+0.00013, gaia_np2[gs][1]+0.00013,
                                                                        gs))


ID_gns2 = np.arange(len(gns2))
# with open(pruebas2+ 'gns2_around_gaiaf%sc%s_sxy%s.reg'%(field_two,chip_two,max_sig), 'w') as f:
#     f.write('# Region file format: DS9 version 4.1'+"\n"+'global color=blue dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1'+"\n"+'fk5'+'\n')
#     f.close

# for gst in range(len(gaia_np2)):
#     m_point = SkyCoord(ra =[gaia_np2[gst][0]], dec  =[gaia_np2[gst][1]],unit ='degree', frame ='icrs', equinox = 'J2000', obstime = 'J2025.4')
#     idx, d2d, d3d = match_coordinates_sky(m_point, gns2_coor)
#     # idxc, group_md, d2d,d3d =  ap_coord.search_around_sky( m_point ,gns1_coor, 0.6*u.arcsec)
#     with open(pruebas2+ 'gns2_around_gaiaf%sc%s_sxy%s.reg'%(field_two,chip_two,max_sig), 'a') as f:
#         f.write('point(%s,%s) # point = cross \n # text(%s,%s) text={%.0f} \n'%(float(gns2_coor.ra.value[idx]) ,float(gns2_coor.dec.value[idx]),
#                                                                                 float(gns2_coor.ra.value[idx]) ,float(gns2_coor.dec.value[idx]),
#                                                                                 int(ID_gns2[idx])))
#         f.close 

gns1_ra_dec = SkyCoord(ra = gns1_fg[:,0], dec = gns1_fg[:,1], unit ='deg', frame = 'fk5',equinox ='J2000',obstime='J2015.4')
gns2_ra_dec = SkyCoord(ra= gns2[:,0]*u.degree, dec=gns2[:,1]*u.degree, frame = 'fk5', equinox = 'J2000',obstime='J2022.4')

gns1_ra_dec =gns1_ra_dec.transform_to('icrs')
gns2_ra_dec =gns2_ra_dec.transform_to('icrs')


# Then we macth with GAIA stars for helping aa to find the transformation
gaia_coord = SkyCoord(ra = gaia_np2[:,0], dec = gaia_np2[:,1], unit = 'degree',frame = 'icrs',obstime='J2016.0' )
gns2_coord = SkyCoord(ra= gns2[:,0]*u.degree, dec=gns2[:,1]*u.degree, frame = 'icrs',obstime='J2022.4')

max_sep = 0.080*u.arcsec
idx,d2d,d3d = gaia_coord.match_to_catalog_sky(gns2_coord,nthneighbor=1)# ,nthneighbor=1 is for 1-to-1 match
sep_constraint = d2d < max_sep
gaia2_match = gaia_np2[sep_constraint]
gns2_match = gns2[idx[sep_constraint]]


x2_mas = gns2_match[:,2]*0.5*0.106*1000
y2_mas = gns2_match[:,3]*0.5*0.106*1000

xy2_mas = np.array([x2_mas,y2_mas]).T

fig, ax = plt.subplots(1,1,figsize=(10,10))
ax.set_title('GNS2 xy and Gaia radec_offsets (%s matching stars)'%(len(x2_mas)))
ax.scatter(xy2_mas[:,0], xy2_mas[:,1], label = 'GNS2 xy (mas)')
ax.scatter(gaia2_match[:,4],gaia2_match[:,5], label = 'Gaia offsets (mas)')
ax.legend()



max_offset = 1
check_x,check_y=max_offset+1,max_offset+1
while abs(check_x) >max_offset or abs(check_y)>max_offset  :# only when tranformation gets better than 1 chip desplacement the coordinates are stored

    print('starting aa')
    print('Stars in transformed list %s'%(len(gns2_match )))
    tic = time.perf_counter()
    m2,(_,_)= aa.find_transform(xy2_mas,gaia2_match[:,4:6],max_control_points=3)
    # m,(_,_)= aa.find_transform(gns1_match[:,0:2],gaia_np1[:,0:2],max_control_points=200)
    
    # m,(_,_)= aa.find_transform(dic_gns['gns_c%s'%(i)],dic_vvv['vvv_c%s'%(i)],max_control_points=400)
    print("Translation: (x, y) = (%.2f, %.2f)"%(m2.translation[0],m2.translation[1]))
    print("Rotation: %.1f degrees"%(m2.rotation * 180.0 /np.pi)) 
    # print("Rotation: %s"%(m.rotation))
    print("Scale factor: %.4f"%(m2.scale))
    toc = time.perf_counter()
    print('it took %.2f seconds'%(toc-tic))
    
    test_gns2 = aa.matrix_transform(xy2_mas, m2.params)
    
    
    
    print(20*'#'+'\n'+'CHECKING GNS2'+'\n'+20*'#')
    tic = time.perf_counter()
    t2,(_,_)= aa.find_transform(test_gns2,gaia2_match[:,4:6],max_control_points=3)
    print("Translation: (x, y) = (%.2f, %.2f)"%(t2.translation[0],t2.translation[1]))
    print("Rotation: %.1f degrees"%(t2.rotation * 180.0 /np.pi)) 

    # print("Rotation: %.1f arcsec"%(t.rotation * 180.0*60*60 /np.pi)) 
    print("Scale factor: %.4f"%(t2.scale))
    toc = time.perf_counter()
    print('checking took %.2f seconds'%(toc-tic))
    check_x= t2.translation[0]
    check_y= t2.translation[1]

fig, ax = plt.subplots(1,1,figsize =(10,10))
ax.set_title('Gaia and tranformed GNS2')
ax.scatter(gaia2_match[:,4],gaia2_match[:,5],s=100, label = 'Gaia') 
ax.scatter(test_gns2[:,0],test_gns2[:,1],s=20, label = 'GNS2 aa (mas)\nRot = %.1fº \n$Tras_{(x, y)}$ = (%.0f, %.0f)'%(m2.rotation * 180.0 /np.pi,m2.translation[0],m2.translation[1]))
ax.legend(loc =2)




gns2[:,2:4] = aa.matrix_transform(gns2[:,2:4]*0.5*0.106*1000, m2.params)

# np.savetxt(GNS_2off + 'aa_stars_calibrated_H_chip%s_sxy%s.txt'%(chip_two,max_sig),gns2,
#            fmt = '%.8f', header = 'ra2, dec2, xi, yi, f2, H2, dx2, dy2, df2, dH2')
np.savetxt(GNS_2off + 'aa_stars_calibrated_H_chip%s_on_gns1_f%sc%s_sxy%s.txt'%(chip_two,field_one,chip_one,max_sig),gns2,
           fmt = '%.8f', header = 'ra2, dec2, xi, yi, f2, H2, dx2, dy2, df2, dH2')
# sys.exit('502')
# %%

# %%
fig, ax = plt.subplots(1,1,figsize =(10,10))
# ax.set_title('Gaia and tranformed GNS2')
ax.scatter(gaia_match[:,4],gaia_match[:,5],s=400, label = 'Gaia 1')
ax.scatter(test_gns[:,0],test_gns[:,1],s=250, label = 'GNS1 aa') 
ax.scatter(gaia2_match[:,4],gaia2_match[:,5],s=150,label = 'Gaia 2') 
ax.scatter(test_gns2[:,0],test_gns2[:,1],s=50, label = 'GNS2 aa')
ax.legend()

# %%
# bg = np.where(((gns1[:,5]-gns1[:,10])>1.3) & (np.sqrt(gns1[:,6]**2 + gns1[:,7]**2) <np.mean([gns1[:,6],gns1[:,7]])))
# bg = np.where(((gns1[:,5]-gns1[:,10])>1.3) & (np.sqrt(gns1[:,6]**2 + gns1[:,7]**2) <0.5))

bg = np.where((gns1[:,5]-gns1[:,10])>1.3)

gns1_bg = gns1[bg]
com1, com2 =compare_lists(gns1_bg,gns2,2,3,2,3,100)
# %
bins = 20

xdes = (com2[:,2]-com1[:,2])/6.975
ydes = (com2[:,3]-com1[:,3])/6.975
fig, ax = plt.subplots(1,2,figsize =(20,10))
ax[0].set_title('Checking distrutions')
ax[1].set_title('No Gaia clipping')
if clip_bad1 == 'yes':
    ax[1].set_title('Clipped %s Gaia1 and %s Gaia2 stars'%(len(bad1),len(bad2)))
ax[0].set_title('Checking distrutions')
ax[0].hist(xdes,bins=bins,label = '$\overline{\mu_{ra}}$ =%.2f\n$\sigma_{ra}$ =%.2f'%(np.mean(xdes),np.std(xdes)))
ax[1].hist(ydes,bins=bins,label = '$\overline{\mu_{dec}}$ =%.2f\n$\sigma_{dec}$ =%.2f'%(np.mean(ydes),np.std(ydes)))
ax[0].legend()
ax[1].legend()
ax[0].set_xlabel('$\mu_{ra}$ (mas/yr)')
ax[1].set_xlabel('$\mu_{dec}$ (mas/yr)')









