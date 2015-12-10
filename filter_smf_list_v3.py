'''
Author: Matthew J. Holman

filter_smf_list_v3.py smfFileList

Reads the file names in smfFileList
One by one opens those files, figures out the time and the pointing, and then
checks for stars and minor planets that might be very close to each of the detections.
Given those matches, the smf files are updated to record that information.  That
information is used in success machine learning classification.

This routine is set up to run on Harvard's Odyssey cluster.  It relies upon a
directory structure that contains are large set of catalogs of the positions of
stationary sources.  In additon, it relies upon access to a code (also written by
Holman) that predicts the positions of known minor plnets at the time of each smf
exposure and matches those positions with the sky plane positions of detections in 
the smf file.
'''
import sys, math, commands, os
from math import degrees
import ephem
from astropy.io import fits
from scipy import spatial
import healpy as hp
import numpy as np
import warnings
import subprocess

nside = 32
nested_scheme = True
sr = (1.0/3600.)*math.pi/180.
mp_sr = (2.0/3600.)*math.pi/180.

def parse_catalog_line(line):
    l = line.split()
    ra  = l[0]
    dec = l[1]
    nNights      = l[-2]
    pairTimeSpan = l[-5]
    return ra, dec, nNights, pairTimeSpan

def ra_dec2pix(ra, dec, nside, nested=nested_scheme):
    phi   = ra*math.pi/180.
    theta = math.pi/2.0 - dec*math.pi/180.
    pix = hp.ang2pix(nside, theta, phi, nested)
    return pix
	
def read_stationary_catalog(px):
    main_dir = "/n/panlfs2/OSS/data/boresight_healpix_redo/"
    theta_r, phi_r = hp.pixelfunc.pix2ang(nside, px, nest=nested_scheme)
    ra_r = phi_r
    dec_r = math.pi/2.0 - theta_r
    dr = main_dir + "%05.1lf_%+06.1lf_%05d" % (ra_r*180./math.pi, dec_r*180./math.pi, px)
    filename =   dr + "/stationary_catalog10"
    if os.path.isfile(filename):
        ras  = []
        decs = []
        with open(filename) as f:
            f.readline()
            lines = f.readlines()
            for l in lines:
                ra, dec, pairTimeSpan, nNights = parse_catalog_line(l)
                if(nNights >= 3 or pairTimeSpan >= 0.5):
                    ras.append(float(ra))
                    decs.append(float(dec))
                else:
                    # It appears that all of the catalog points satisfy the criteria,
                    # so this check is not needed.
                    print "point rejected %lf %lf %lf %d" % (ra, dec, pairTimeSpan, nNights)
        return ras, decs
    else:
        return None, None

def sunradec(jd):
    day = jd -2415020
    sun = ephem.Sun(day)
    return degrees(sun.a_ra), degrees(sun.a_dec)

# =============================
# JD Convertor (use pyephem)
# This piece came from one of Ed's codes.
# =============================
def date2JD(Cinput):
    #2010-07-22T10:17:00.187239
    Cinput=str(Cinput)
    Cdate = Cinput.split('T')[0]
    Ctime = Cinput.split('T')[1]
    Cyear = Cdate.split('-')[0]
    Cmonth= Cdate.split('-')[1]
    Cday  = Cdate.split('-')[2]
    d = ephem.Date('%s/%s/%s %s' %(Cyear,Cmonth,Cday,Ctime))
    #print '%s/%s/%s %s' %(Cyear,Cmonth,Cday,Ctime)
    #print float(d)+2415020
    return float(d)+2415020


def get_deteff(hdu, zpt_obs, exptime):
    #print hdu.columns
    #print hdu.data
    mag_ref = hdu.header["DETEFF.MAGREF"]

    mag_limit = mag_ref + zpt_obs + 2.5*math.log10(exptime)
    #print mag_limit
    count_max = hdu.data[0][1]
    for offset, count, diff_mean, diff_stdev, err_mean in hdu.data:
        frac = count/float(count_max)
        val = 1.0/(1.0+math.exp((offset-0.12)/0.2))
        #print "%8.4lf %8.4lf %8.4lf " % (offset, frac, val)

def adjustnan(x):
    if(np.isnan(x) or np.isinf(x)):
        return 99.0
    else:
        return x

def pick_mag(x, y):
    if(not(np.isnan(x)) and not(np.isinf(x))):
        return x
    else:
        return y

def foldx(x, xgap=8, ygap=10):
    x= float(x)
    nx = x%(600+xgap)
    return nx

def foldy(y, xgap=8, ygap=10):
    y = float(y)
    ny = y%(600+ygap)
    return ny

def mag_err2snr(mag_err):
    nsr = 1.0 - math.pow(10.0, -0.4*math.fabs(mag_err))
    if(nsr != 0.0):
        snr = 1.0/nsr
    else:
        small_err = 0.0001
        nsr = 1.0 - math.pow(10.0, -0.4*small_err)
        snr = 1.0/nsr
    return snr

def inBox(x, y):
    x_fold = foldx(x)
    y_fold = foldy(y)
    inbox = (10 < x_fold < 582 and 5 < y_fold < 589)
    return inbox

def magOK(mag, mag_err):
    mag_bad = np.isnan(mag) or np.isinf(mag) or np.isnan(mag_err) or np.isinf(mag_err)
    return not(mag_bad)

def convert_points(ra, dec):
    ra = np.array(ra)*np.pi/180.
    dec = np.array(dec)*np.pi/180.
    x = np.cos(ra)*np.cos(dec)
    y = np.sin(ra)*np.cos(dec)
    z = np.sin(dec)
    points = np.array([x, y, z]).T
    return points


def get_catalog_tree(ra_c, dec_c, search_radius):
    #
    # Get stationary catalogs and convert the
    # RA/Dec values to x,y,z unit vectors
    #
    # This code should be caching the catalogs
    # for speed.
    #
    # And probably there should be a smaller 
    # number of larger catalogs that have already
    # been converted to x,yz unit vectors and
    # stored in binary files
    #
    phi_c   = ra_c*math.pi/180.
    theta_c = math.pi/2.0 - dec_c*math.pi/180.
    search_radius = 1.6*math.pi/180.

    vec = hp.ang2vec(theta_c, phi_c)
    res = hp.query_disc(nside, vec, search_radius, nest=nested_scheme, inclusive=True)

    ras  = []
    decs = []
    print ra_c, dec_c
    for pix in res:
        ras_tmp, decs_tmp = read_stationary_catalog(pix)
        if(ras_tmp != None and decs_tmp != None):
            ras = ras + ras_tmp
            decs = decs + decs_tmp

    # Conver the points to x,y,z unit vectors
    if(ras != [] and decs != []):
        points = convert_points(ras, decs)
        # Build a KD-tree of the stationary catalog
        # points
        tree = spatial.cKDTree(points)
    else:
        tree = None

    # Return the tree
    return tree


def get_minor_planets(jd, ra_c, dec_c, search_radius):
    #
    # Get the minor planets that have positions within
    # the angular search radius (in degrees) of the central
    # RA/Dec values at the specified JD.
    #
    #
    prog = "/n/home01/mholman/OSS/propagate/observe_minor_planet_simple"
    mpcat = "/n/home01/mholman/smf/mpcorb_K11BF.txt"
    run_args = [prog, str(jd), str(ra_c), str(dec_c), mpcat, str(search_radius)]
    output = subprocess.check_output(run_args).split('\n')
    #print len(output)
    ras  = []
    decs = []
    others = []
    for line in output:
        if line != '':
            ls = line.rstrip('\n').split()
            ra  = float(ls[0])
            ras.append(ra)
            dec = float(ls[1])
            decs.append(dec)

            mag = float(ls[2])
            desig = ls[5]
            rrate = float(ls[-2])
            drate = float(ls[-1])
            others.append((desig, mag, ra, dec, rrate, drate))

    return convert_points(ras, decs), others

def process_file(infile, outfile, headerfile, txtfile):
    tfile = open(txtfile, "w")
    with fits.open(infile) as hdulist:

        warnings.resetwarnings()
        warnings.filterwarnings('ignore', category=UserWarning, append=True)

        # Collect some general header information
        zpt_obs  = hdulist[0].header["ZPT_OBS"]
        filterID = hdulist[0].header["FILTERID"].rstrip('.00000')
        ra_c     = hdulist[0].header["RA"]
        dec_c    = hdulist[0].header["DEC"]
        exptime  = hdulist[0].header["EXPTIME"]
        mjdObs   = hdulist[0].header["MJD-OBS"]
        airmass  = hdulist[0].header["AIRMASS"]
    
        # Do a few calculations
        jd = mjdObs + 2400000.5
        sun_ra = sunradec(jd)[0]
        sun_dec = sunradec(jd)[1]

        search_radius = 1.6
        tree = get_catalog_tree(ra_c, dec_c, search_radius*math.pi/180)
        if tree == None:
            return None

        mp_xyz, mp_other_fields = get_minor_planets(jd, ra_c, dec_c, search_radius)
        #print mp_other_fields

        #print filterID

        mp_count = 0
        j = 0
        nrows_total = 0
        newhdus = []
        shutoutc = None
        for hdu in hdulist:
            #print hdu.name
            ext_type = hdu.name.split('.')[-1]
            if ext_type == 'deteff':
                # One of the detection efficiency extensions.
                # We'll want to do something with this information.
                get_deteff(hdu, zpt_obs, exptime)

            elif ext_type == 'hdr':
                # One of the psf postage stamp extensions
                # We can get shutoutc from here, but we
                # only need to do that once.
                if shutoutc == None:
                    try:
                        shutoutc = hdu.header['SHUTOUTC']
                        shutcutc = hdu.header['SHUTCUTC']
                    except Exception,e:
                        pass
            
            elif ext_type == 'psf' and "EXTTYPE" in hdu.header and (hdu.header["EXTTYPE"] == "PS1_V3"):
            # One of the extensions with tables of detections.
            # This holds our primary data.
                chip=hdu.header['EXTHEAD'].rstrip('.hdr')
                hdu_name = hdu.name.split('.')[0]
                nrows = hdu.header['NAXIS2']
                nrows_total = nrows_total + nrows

                col_dict = {}
                    #print hdu.columns.names
                for col in hdu.columns.names:
                    col_dict[col] = hdu.data.field(col)
                
                ras  = col_dict['RA_PSF']
                decs = col_dict['DEC_PSF']
                xs   = col_dict['X_PSF']
                ys   = col_dict['Y_PSF']
                x_sigs   = col_dict['X_PSF_SIG']
                y_sigs   = col_dict['Y_PSF_SIG']
                plts = col_dict['PLTSCALE']
                mag0s= col_dict['CAL_PSF_MAG']
                mag1s= col_dict['PSF_INST_MAG'] + zpt_obs + 2.5*math.log10(exptime)
                mags = map(pick_mag, mag0s, mag1s)
                mag_errs = col_dict['PSF_INST_MAG_SIG']
                psf_minors= col_dict['PSF_MINOR']*plts
                psf_majors= col_dict['PSF_MAJOR']*plts
                flags    = col_dict['FLAGS']
                flags2   = col_dict['FLAGS2']
                fwhms = 2.3548 * np.sqrt(psf_majors*psf_minors)
                moments_R1s = col_dict['MOMENTS_R1']*plts
                psf_qf_perfects = col_dict['PSF_QF_PERFECT']
                psf_extents = (moments_R1s*moments_R1s)/(psf_majors*psf_minors)

                # Should check if ras and decs are empty
                qpoints = convert_points(ras, decs)

                # Make a tree of the detections, in order to search it for minor planets
                # This is not the most efficient approach because I am building a separate
                # tree for each chip, rather than one big tree.
                # But I can time the code with and without this section.
                qtree = spatial.cKDTree(qpoints)
                mp_idx = qtree.query_ball_point(mp_xyz, mp_sr)
                mp_desig_val = [None]*len(ras)
                mp_prox_val  = np.zeros(len(ras), dtype=np.float64)
                mp_prox_val[:] = np.NAN
                mp_mag_val   = np.zeros(len(ras), dtype=np.float64)
                mp_mag_val[:] = np.NAN
                mp_rrate_val = np.zeros(len(ras), dtype=np.float64)
                mp_rrate_val[:] = np.NAN
                mp_drate_val = np.zeros(len(ras), dtype=np.float64)
                mp_drate_val[:] = np.NAN
                for i, idx in enumerate(mp_idx):
                    if len(idx) >= 1:
                        for j in idx[0:1]: # Taking just the closest one here.
                            mp_count +=1
                            mp_desig, mp_mag, mp_ra, mp_dec, mp_rrate, mp_drate = mp_other_fields[i]
                            mp_prox = math.acos(np.dot(mp_xyz[i], qpoints[j]))*206265.
                            #print mp_count, chip, j, mp_other_fields[i], ras[j], decs[j], mags[j], mp_prox
                            mp_desig_val[j] = mp_desig
                            mp_prox_val[j]  = mp_prox
                            mp_mag_val[j]   = mp_mag
                            mp_rrate_val[j] = mp_rrate
                            mp_drate_val[j] = mp_drate

                idx_fr = tree.query_ball_point(qpoints, sr)
                offStar_mask = [matches == [] for matches in idx_fr]
                onStar_mask = [matches != [] for matches in idx_fr]

                inBox_mask = [inBox(x, y) for x, y in zip(xs, ys)]
                magOK_mask = [magOK(mag, mag_err) for mag, mag_err in zip(mags, mag_errs)]

                mask = map(lambda x, y, z: (x and y and  z), inBox_mask, magOK_mask, offStar_mask)
                mask = np.array(mask)

                cols = [] 
                cols.append(
                    fits.Column(name='on_star', format='int16', array=np.array(onStar_mask)*1)
                    )
                cols.append(
                    fits.Column(name='mp_desig', format='A12', array=mp_desig_val)
                    )
                cols.append(
                    fits.Column(name='mp_prox', format='D8', array=np.array(mp_prox_val))
                    )
                cols.append(
                    fits.Column(name='mp_mag', format='D8', array=np.array(mp_mag_val))
                    )
                cols.append(
                    fits.Column(name='mp_ra_rate', format='D8', array=np.array(mp_rrate_val))
                    )
                cols.append(
                    fits.Column(name='mp_dec_rate', format='D8', array=np.array(mp_drate_val))
                    )

                orig_cols = hdu.columns
                new_cols = fits.ColDefs(cols)
                hdu = fits.BinTableHDU.from_columns(orig_cols + new_cols, header=hdu.header)

                # Here we can check which detections are angularly close to known sources
                # by comparing to a catalog with a KD-tree.
                # 

                #hdu.data = hdu.data[mask]

                for i, (ra, dec, x, y, x_sig, y_sig, mag, mag_err, fwhm, flag, flag2, psf_major, psf_minor, moments_R1, psf_qf_perfect, psf_extent) in enumerate(zip(ras, decs, xs, ys, x_sigs, y_sigs, mags, mag_errs, fwhms, flags, flags2, psf_majors, psf_minors, moments_R1s, psf_qf_perfects, psf_extents)):
                    if inBox_mask[i] and magOK_mask[i] and offStar_mask[i]:
                        snr = mag_err2snr(mag_err)
                        pix = ra_dec2pix(ra, dec, nside, nested=nested_scheme)
                        wstring = "%d %lf %lf %lf %lf %x %s %.3lf %.3lf %.3lf %.3lf %.3lf %.3lf %.3lf %.3lf %.3lf %.3lf %.3lf %x %d\n" % (j, ra, dec, mag, mag_err, flag, chip, fwhm, snr, x, y, x_sig, y_sig, psf_major, psf_minor, moments_R1, psf_qf_perfect, psf_extent, flag2, pix)
                        tfile.write(wstring)

                    j = j+1

            elif hdu.name == 'PRIMARY' or hdu.name == 'MATCHED_REFS':
                # One of the first two extensions.
                # There is nothing to do, at this point.
                pass

            else:
                # Unknown type of extension.
                    #print "Some other kind of extension " + hdu.name
                pass

            newhdus.append(hdu)

        thdulist = fits.HDUList(newhdus)
        thdulist.writeto(outfile, clobber=True, output_verify="silentfix")
        #print shutoutc
        shutoutc_jd = date2JD(shutoutc)
        output_txt = '%20.10f %6.3f %s %11.6f %12.6f %6.3f %8.4f %s %11.6f %12.6f' % (shutoutc_jd, exptime, str(filterID), ra_c, dec_c, airmass, zpt_obs, shutoutc, sun_ra, sun_dec)
        h = open(headerfile, 'w')
        h.write(output_txt+'\n')
        h.close()
        return True


if __name__ == '__main__':

  # Get arguments from command line and do checks
  try:
      inlist = sys.argv[1]
  except:
    print 'filter_smf_list_v3.py infile'
    sys.exit()

file = open(inlist, "r")
for infile in file.readlines():
    infile = infile.rstrip("\n")
    outfile = infile.rstrip('.smf') + '.n.smf'
    headerfile = infile.rstrip('.smf') + '.n.pointing'
    txtfile = infile.rstrip('.smf') + '.n.txt'
    #txtfile = open(txtfilename, "r")
    print infile, 
    if process_file(infile, outfile, headerfile, txtfile) == None:
        print "Skipped"
    else:
        print "Finished"
