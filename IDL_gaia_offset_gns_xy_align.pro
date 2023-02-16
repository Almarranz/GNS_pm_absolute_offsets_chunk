pro IDL_gaia_offset_gns_xy_align;,survey,align_degree,dmax


; Aligenment oF GNS offsets(from xy coordenates previously aligened with
; Gaia offsste), with Gaia offsets

; survey = 1; gns1 is '1' and gns2 is '2'
field_one = '7' 
chip_one = '4'
field_two = '7'
chip_two = '1'
GNS_1off='/Users/amartinez/Desktop/PhD/HAWK/GNS_1off_chunk/lists/'+strn(field_one)+'/chip'+strn(chip_one)+'/'
GNS_2off='/Users/amartinez/Desktop/PhD/HAWK/GNS_2off_chunk/lists/'+strn(field_two)+'/chip'+strn(chip_two)+'/'
pruebas2 = '/Users/amartinez/Desktop/PhD/HAWK/GNS_2off_chunk/pruebas/'
pruebas1 = '/Users/amartinez/Desktop/PhD/HAWK/GNS_1off_chunk/pruebas/'

dmax =50
max_sig =0.1
; max_sig =1  
if max_sig lt 1 then begin
    max_sig=  string(max_sig)
    max_sig = max_sig.Remove(-5)
endif
for survey =1, 2 do begin
;    dmax = dmax*survey
  for align_degree =1, 2 do begin
if survey eq 1 then begin
    print,'++++++++++++++++++++++++++++++++++++++'
    print, 'Working in GNS1 alignment'
    print,'++++++++++++++++++++++++++++++++++++++'
    readcol, GNS_1off + 'aa_stars_calibrated_HK_chip'+strn(chip_one)+'_on_gns2_f'+strn(field_two)+'c'+strn(chip_two)+'_sxy'+strn(max_sig)+'.txt',$
    ra, dec, x, y, f, H, dx, dy, df, dH,Ks, dKs,Format ='A,	A,	A,	A,	A,	A,	A,	A,	A,	A,	A,	A,',SKIPLINE =1  
    ra	=float(ra)
    dec	=float(dec)
    x	=float(x)
    y	=float(y)
    f	=float(f)
    H	=float(H)
    dx	=float(dx)
    dy	=float(dy)
    df	=float(df)
    dH	=float(dH)
    Ks	=float(Ks)
    dKs	=float(dKs)   
    
    readcol, GNS_1off + 'gaia_refstars_on_gns1_f'+strn(field_one)+'c'+strn(chip_one)+'.txt',$
    ra_ref, dec_ref, dx_ref, dy_ref, x_ref, y_ref, pmra_ref, pmdec_ref, dpmra_ref, dpmdec_ref,id_ref,$
    FORMAT = 'A,	A,	A,	A,	A,	A,	A,	A,	A,	A,	A,',SKIPLINE = 1 
    
endif

if survey eq 2 then begin
    print,'++++++++++++++++++++++++++++++++++++++'
    print, 'Working in GNS2 alignment'
    print,'++++++++++++++++++++++++++++++++++++++'
    readcol, GNS_2off + 'aa_stars_calibrated_H_chip'+strn(chip_two)+'_on_gns1_f'+strn(field_one)+'c'+strn(chip_one)+'_sxy'+strn(max_sig)+'.txt',$
    ra, dec, x, y, f, H, dx, dy, df, dH,Format ='A,	A,	A,	A,	A,	A,	A,	A,	A,	A',SKIPLINE =1  
    ra	=float(ra)
    dec	=float(dec)
    x	=float(x)
    y	=float(y)
    f	=float(f)
    H	=float(H)
    dx	=float(dx)
    dy	=float(dy)
    df	=float(df)
    dH	=float(dH)
     
    
    readcol, GNS_2off + 'gaia_refstars_on_gns2_f'+strn(field_two)+'c'+strn(chip_two)+'_gns1_f'+strn(field_one)+'c'+strn(chip_one)+'.txt',$
    ra_ref, dec_ref, dx_ref, dy_ref, x_ref, y_ref, pmra_ref, pmdec_ref, dpmra_ref, dpmdec_ref,id_ref,$
    FORMAT = 'A,	A,	A,	A,	A,	A,	A,	A,	A,	A,	A,',SKIPLINE = 1  
    
endif

 
ra_ref	=float(ra_ref)
dec_ref	=float(dec_ref)
dx_ref	=float(dx_ref)
dy_ref	=float(dy_ref)
dx	=float(dx)
dy	=float(dy)
pmra_ref	=float(pmra_ref)
pmdec_ref	=float(pmdec_ref)
dpmra_ref	=float(dpmra_ref)
dpmdec_ref	=float(dpmdec_ref)
id_ref	=float(id_ref)          

print, n_elements(ra_ref)

; dmax =0.7

compare_lists, x_ref, y_ref, x, y, xrefc, yrefc, xc, yc, MAX_DISTANCE=dmax, SUBSCRIPTS_1=subc1, SUBSCRIPTS_2 = subc2, SUB1 = sub1, SUB2 = SUB2

print,'++++++++++++++++++++++++++++++++++++++'
print,'Common stars with Gaia:',n_elements(subc1),'dmax =',dmax
print,'++++++++++++++++++++++++++++++++++++++'

    
count=0
comm=[]
it=0
lim_it=1
if (align_degree eq 1) || (align_degree eq 2) || (align_degree eq 3)  then begin
while count lt lim_it do begin
	  it=it+1
	 
	  polywarp, x_ref[subc1], y_ref[subc1], x[subc2], y[subc2], 1, Kx, Ky
	  print, Kx
	  print, Ky		
	  xi = Kx[0,0] + Kx[0,1]*x + Kx[1,0]*y + Kx[1,1]*x*y
	  yi = Ky[0,0] + Ky[0,1]*x + Ky[1,0]*y + Ky[1,1]*x*y
	  compare_lists, x_ref, y_ref, xi, yi, x2c, y2c, x1c, y1c, MAX_DISTANCE=dmax, SUBSCRIPTS_1=subc1, SUBSCRIPTS_2 = subc2, SUB1 = sub1, SUB2 = sub2
	  nc = n_elements(subc1)
	  print, 'Iteration ' + strn(it)
	  print, 'Found ' + strn(nc) + ' common stars.'
	  comm=[comm,nc]
	  if (n_elements(comm) gt 2) then begin
	   if comm[-2] ge comm[-1] then begin
	   count=count+1
	  endif else begin
	   count=0
	  endelse
	  endif
	 endwhile
endif    

count=0
comm=[]
it=0
lim_it=1

; Is all the gaia stars are find in the degree 1 loop, Do I need to run the
; second degree fit? Will it improve things up? Maybe it will worth to check
; it later, when calculating the alignment uncertainies ...
if (align_degree eq 2) || (align_degree eq 3)  then begin
print, '#######################'
print, 'Now Degree 2 alignment.'
print, '#######################'

while count lt lim_it do begin
	  it=it+1
	  
	  polywarp, x_ref[subc1], y_ref[subc1], x[subc2], y[subc2], 2, Kx, Ky
	  print, Kx
	  print, Ky		
	   xi = Kx[0,0] + Kx[0,1]*x + Kx[1,0]*y + Kx[1,1]*x*y + Kx[0,2]*x^2 + Kx[1,2]*x^2*y + Kx[2,2]*x^2*y^2 + Kx[2,0]*y^2 + Kx[2,1]*y^2*x
       yi = Ky[0,0] + Ky[0,1]*x + Ky[1,0]*y + Ky[1,1]*x*y + Ky[0,2]*x^2 + Ky[1,2]*x^2*y + Ky[2,2]*x^2*y^2 + Ky[2,0]*y^2 + Ky[2,1]*y^2*x

	  compare_lists, x_ref, y_ref, xi, yi, x2c, y2c, x1c, y1c, MAX_DISTANCE=dmax, SUBSCRIPTS_1=subc1, SUBSCRIPTS_2 = subc2, SUB1 = sub1, SUB2 = sub2
	  nc = n_elements(subc1)
	  print, 'Iteration ' + strn(it)
	  print, 'Found ' + strn(nc) + ' common stars.'
	  comm=[comm,nc]
	  if (n_elements(comm) gt 2) then begin
	   if comm[-2] ge comm[-1] then begin
	   count=count+1
	  endif else begin
	   count=0
	  endelse
	  endif
	 endwhile
endif    
count=0
comm=[]
it=0
lim_it=1
if align_degree eq 3  then begin
print, '#######################'
print, 'Now Degree 3 alignment.'
print, '#######################'

while count lt lim_it do begin
	  it=it+1
	  
	  polywarp, x_ref[subc1], y_ref[subc1], x[subc2], y[subc2], 3, Kx, Ky
	  print, Kx
	  print, Ky		
	   xi = Kx[0,0] + Kx[0,1]*x + Kx[1,0]*y + Kx[1,1]*x*y + Kx[0,2]*x^2 + Kx[1,2]*x^2*y + Kx[2,2]*x^2*y^2 + Kx[2,0]*y^2 + Kx[2,1]*y^2*x +$
 		Kx[0,3]*x^3 + Kx[1,3]*x^3*y + Kx[2,3]*x^3*y^2 + Kx[3,0]*y^3 + Kx[3,1]*x*y^3 + Kx[3,2]*x^2*y^3 + Kx[3,3]*x^3*y^3
       yi = Ky[0,0] + Ky[0,1]*x + Ky[1,0]*y + Ky[1,1]*x*y + Ky[0,2]*x^2 + Ky[1,2]*x^2*y + Ky[2,2]*x^2*y^2 + Ky[2,0]*y^2 + Ky[2,1]*y^2*x +$
 	    Ky[0,3]*x^3 + Ky[1,3]*x^3*y + Ky[2,3]*x^3*y^2 + Ky[3,0]*y^3 + Ky[3,1]*x*y^3 + Ky[3,2]*x^2*y^3 + Ky[3,3]*x^3*y^3


	  compare_lists, x_ref, y_ref, xi, yi, x2c, y2c, x1c, y1c, MAX_DISTANCE=dmax, SUBSCRIPTS_1=subc1, SUBSCRIPTS_2 = subc2, SUB1 = sub1, SUB2 = sub2
	  nc = n_elements(subc1)
	  print, 'Iteration ' + strn(it)
	  print, 'Found ' + strn(nc) + ' common stars.'
	  comm=[comm,nc]
	  if (n_elements(comm) gt 2) then begin
	   if comm[-2] ge comm[-1] then begin
	   count=count+1
	  endif else begin
	   count=0
	  endelse
	  endif
	 endwhile
endif    



if survey eq 1 then begin
forprint, TEXTOUT= GNS_1off + 'gns1_in_gaiarf_f'+strn(field_one)+'c'+strn(chip_one)+'_deg'+strn(align_degree)+'_sxy'+strn(max_sig)+'.txt',ra, dec, xi, yi, f, H, dx, dy,df, dH, Ks, dKs,format='(12(f, 4X))', COMMENT ='# raH, decH, xi, yi, f1, mH, dx, dy,df2, dmH, mK, dmK '
                                                                                                                                           
endif
                                                                                                                                          
if survey eq 2 then begin
forprint, TEXTOUT= GNS_2off + 'gns2_in_gaiarf_f'+strn(field_two)+'c'+strn(chip_two)+'_on_gns1f'+strn(field_one)+'c'+strn(chip_one)+'_deg'+strn(align_degree)+'_sxy'+strn(max_sig)+'.txt',ra, dec, xi, yi, f, H, dx, dy,df, dH,format='(10(f, 4X))', COMMENT ='# ra2, dec2, xi, yi, f2, H2, dx2, dy2, df2, dH2 '
endif
endfor
endfor

end    