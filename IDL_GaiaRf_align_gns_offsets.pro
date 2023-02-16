pro IDL_GaiaRf_align_gns_offsets,dmax;,grf_degree,dmax

; dmax =200
for grf_degree =1, 2 do begin
; It uses lists previously aligned with Gaia referece frame. Then it aligns
; GNS1 to GNS2 
 ; this is the degree of the fitting polynomial for gaia reference frame.
field_one = '7' 
chip_one = '4'
field_two = '7'
chip_two = '1'


GNS_1off='/Users/amartinez/Desktop/PhD/HAWK/GNS_1off_chunk/lists/'+strn(field_one)+'/chip'+strn(chip_one)+'/'
GNS_2off='/Users/amartinez/Desktop/PhD/HAWK/GNS_2off_chunk/lists/'+strn(field_two)+'/chip'+strn(chip_two)+'/'
pruebas2 = '/Users/amartinez/Desktop/PhD/HAWK/GNS_2off_chunk/pruebas/'
pruebas1 = '/Users/amartinez/Desktop/PhD/HAWK/GNS_1off_chunk/pruebas/'
pm_folder_off ='/Users/amartinez/Desktop/PhD/HAWK/pm_gns1_gns2_off_chunk/'
uncer_folder_1= '/Users/amartinez/Desktop/PhD/HAWK/GNS_1off_chunk/lists/'+strn(field_one)+'/uncert_lits/'
uncer_folder_2= '/Users/amartinez/Desktop/PhD/HAWK/GNS_2off_chunk/lists/'+strn(field_one)+'/uncert_lits/'

max_sig = 0.1
if max_sig lt 1 then begin
    max_sig=  string(max_sig)
    max_sig = max_sig.Remove(-5)
endif
imagenes = 0

readcol, GNS_1off + 'gns1_in_gaiarf_f'+strn(field_one)+'c'+strn(chip_one)+'_deg'+strn(grf_degree)+'_sxy'+strn(max_sig)+'.txt',RaH1, DecH1, x1, y1, f1, H1, dx1, dy1, df1, dH1, Ks1, dKs1,Format ='A,	A,	A,	A,	A,	A,	A,	A,	A,	A,	A,	A',SKIPLINE = 1
; readcol, GNS_1off + 'gns1_in_gaiarf_f'+strn(field_one)+'c'+strn(chip_one)+'_deg'+strn(grf_degree)+'_sxy'+strn(max_sig)+'.txt',RaH1, DecH1, x1, y1, f1, H1, dx1, dy1, df1, dH1, Ks1, dKs1,Format ='D,	D,	D,	D,	D,	D,	D,	D,	D,	D,	D,	D'
; readcol, GNS_1off + 'gns1_in_gaiarf_f'+strn(field_one)+'c'+strn(chip_one)+'_deg'+strn(grf_degree)+'_sxy'+strn(max_sig)+'.txt',RaH1, DecH1, x1, y1, f1, H1, dx1, dy1, df1, dH1, Ks1, dKs1,SKIPLINE =1

x_off = 0
y_off = 0


x1=double(x1)
dx1=double(dx1)
y1=double(y1)
dy1 = double(dy1)
RaH1=double(RaH1)
DecH1=double(DecH1)
H1=double(H1)
dH1 =double(dH1)
Ks1=double(Ks1)
dKs1=double(dKs1)






gns1_ID = findgen(n_elements(x1),star=1); ID number to identify each star
print,'++++++++++++++++++++++++++++++++++++++'
print, 'Stars in GNS1 common area:',n_elements(x1)
print,'++++++++++++++++++++++++++++++++++++++'

;-----------------------------------------------------------------------------
; WARNINg: This was giving me wrong array, where a particular value for KS
; repeat itself 700 times or so. Did not find where this thing comes from. 
; I will make the color cut in the next scrip
; 
; colorin = 1.3
; HKs= H1-Ks1
; print,'HKs',HKs[0:10]
; color_cut = where(HKs gt colorin)
; 
; x1	=x1[color_cut]
; dx1	=dx1[color_cut]
; y1	=y1[color_cut]
; dy1	=dy1[color_cut]
; RaH1	=RaH1[color_cut]
; dRaH1	=dRaH1[color_cut]
; DecH1	=DecH1[color_cut]
; dDecH1	=dDecH1[color_cut]
; H1	=H1[color_cut]
; dH1	=dH1[color_cut]
; Ks1	=Ks1[color_cut]
; dKs1	=dKs1[color_cut]
; Ks1	=Ks1[color_cut]
; dKs1	=dKs1[color_cut]
;-----------------------------------------------------------------------------
; print,'++++++++++++++++++++++++++++++++++++++'
; print, 'Stars in GNS1 common area:',n_elements(x1)
; print,'++++++++++++++++++++++++++++++++++++++'

readcol, GNS_2off+  'gns2_in_gaiarf_f'+strn(field_two)+'c'+strn(chip_two)+'_on_gns1f'+strn(field_one)+'c'+strn(chip_one)+'_deg'+strn(grf_degree)+'_sxy'+strn(max_sig)+'.txt',raH2, decH2, x2, y2, f2, H2, dx2, dy2, df2, dH2,Format ='A,A,A,A,A,A,A,A,A,A',SKIPLINE = 1 

raH2=double(raH2)
decH2=double(decH2)
x2=double(x2)
y2=double(y2)
f2=double(f2)
H2=double(H2)
dx2=double(dx2)
dy2=double(dy2)
df2=double(df2)
dH2=double(dH2)


print,'++++++++++++++++++++++++++++++++++++++'
print, 'Stasr in GNS2 common area:',n_elements(x2)
print,'++++++++++++++++++++++++++++++++++++++'


; When in GRF one have to compare the list without alignmen (other wise you
; would do an relative alignment all over again)
compare_lists, x2, y2, x1, y1, x2c, y2c, x1c, y1c, MAX_DISTANCE=dmax, SUBSCRIPTS_1=subc1, SUBSCRIPTS_2 = subc2, SUB1 = sub1, SUB2 = SUB2
nc = n_elements(subc1)
print,'++++++++++++++++++++++++++++++++++++++'
print, 'Found ' + strn(nc) + ' common stars. With dmax = ',dmax
print,'++++++++++++++++++++++++++++++++++++++'

; dmax_alig = 50
; readcol, uncer_folder_1 + 'uncer_alig_jk_f'+strn(field_one)+'c'+strn(chip_one)+'_deg'+strn(grf_degree)+'_dmax'+strn(dmax_alig)+'_sxy'+strn(max_sig)+'.txt',x_mean, y_mean, x_std, y_std, Ra_mean, Dec_mean, Format ='A,	A,	A,	A,	A,	A',SKIPLINE = 1                   
; y1_mean	=double(y_mean)
; x1_std	=double(x_std)
; y1_std	=double(y_std)
; Ra1_mean	=double(Ra_mean)
; Dec1_mean	=double(Dec_mean)
; print,'n_elemts(x_std)',n_elements(x1_std)
; ; 
; readcol, uncer_folder_2 + 'uncer_alig_jk_f'+strn(field_two)+'c'+strn(chip_two)+'_on_gns1_f'+strn(field_one)+'c'+strn(chip_one)+'_deg'+strn(grf_degree)+'_dmax'+strn(dmax_alig)+'_sxy'+strn(max_sig)+'.txt',x2_mean, y2_mean, x2_std, y2_std, Ra2_mean, Dec2_mean, Format ='A,	A,	A,	A,	A,	A',SKIPLINE = 1                   
; x2_mean	=double(x2_mean)
; y2_mean	=double(y2_mean)
; x2_std	=double(x2_std)
; y2_std	=double(y2_std)
; Ra2_mean	=double(Ra2_mean)
; Dec2_mean	=double(Dec2_mean)
; print,'n_elemts(x_std)',n_elements(x2_std)


x_dis=(x2c-x1c)/6.975
y_dis=(y2c-y1c)/6.975
print,'n_elements(x_dis)',n_elements(x_dis)
print,'n_elements(subc1),n_elements(subc2)',n_elements(subc1),n_elements(subc2)
;~ Adding velocities uncertanties for x and y directions

dx1=dx1[subc2]
dy1=dy1[subc2]
		
dx2=dx2[subc1]
dy2=dy2[subc1]

; uncertainties in positions are in pixels, we haver to transform the into mas

dx1 = dx1*0.053*1000
dx2 = dx2*0.053*1000	

dy1 = dy1*0.053*1000
dy2 = dy2*0.053*1000		

; dvx=sqrt((dx1+x1_std)^2+(dx2 + x2_std)^2)/6.975
; dvy=sqrt((dy1+y2_std)^2+(dy2+y2_std)^2)/6.975

dvx=sqrt((dx1)^2+(dx2)^2)/6.975
dvy=sqrt((dy1)^2+(dy2)^2)/6.975
x1 = x1[subc2]
y1 = y1[subc2]

x2 = x2[subc1]
y2 = y2[subc1]

H1 = H1[subc2]
dH1 = dH1[subc2]
Ks1 = Ks1[subc2]
dKs1 = dKs1[subc2]

H2 = H2[subc1]
dH2 = dH2[subc1]
RaH1 = RaH1[subc2]
DecH1 = DecH1[subc2]

raH2 = raH2[subc1] 
decH2 = decH2[subc1]

gns1_ID = gns1_ID[subc2]
; dx_al =sqrt((x1_std+x2_std)^2)
; dy_al =sqrt((y1_std+y2_std)^2)
forprint, TEXTOUT= pm_folder_off + 'pm_GaiaRF_ep1_f'+strn(field_one)+'c'+strn(chip_one)+'_ep2_f'+strn(field_two)+'c'+strn(chip_two)+'deg'+strn(grf_degree)+'_dmax'+strn(dmax)+'_sxy'+strn(max_sig)+'.txt',x_dis,y_dis,dvx,dvy,x1,y1,x2,y2,H1,dH1,Ks1,dKs1,H2,dH2,RaH1,DecH1,raH2,decH2,format='(18(f, 4X))', COMMENT ='# x_dis,y_dis,dvx,dvy,x1,y1,x2,y2,H1,dH1,Ks1,dKs1,H2,dH2,RaH1,DecH1,raH2,decH2 '
forprint, TEXTOUT= pm_folder_off + 'ID_pm_GaiaRF_ep1_f'+strn(field_one)+'c'+strn(chip_one)+'_ep2_f'+strn(field_two)+'c'+strn(chip_two)+'deg'+strn(grf_degree)+'_dmax'+strn(dmax)+'_sxy'+strn(max_sig)+'.txt', gns1_ID, format='(1(f, 4X))', COMMENT ='# IDs'
; forprint, TEXTOUT= pruebas1 + 'GRF_sig_astro_f'+strn(field_one)+'c'+strn(chip_one)+'_ep2_f'+strn(field_two)+'c'+strn(chip_two)+'deg'+strn(grf_degree)+'_dmax'+strn(dmax)+'_sxy'+strn(max_sig)+'.txt',dx1,dy1,dx_al,dy_al,H1,dH1, format='(6(f, 4X))', COMMENT ='# dx,dy,dx_ali,dy_ali,H,dH'
endfor 

END



