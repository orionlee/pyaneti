import math
import pandas as pd
import numpy as np

from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.mast import Observations
from astroquery.mast import Catalogs
from astroquery.gaia import Gaia

import matplotlib
import matplotlib.pyplot as plt

# Get stellar data from Exofop using TIC

def get_exofop_data (TIC_no):
    
    TIC = 'TIC ' + TIC_no
    min_magnitude = 22
    radius = "10 arcsec"   # How far round the target star to look.  Each TESS pixel is 21 arcsecs square

    catalog_data = Catalogs.query_criteria(catalog="Tic",objectname=TIC,radius=radius,objType="*")
    dat = np.array(catalog_data)  #extract table data into an array
    df = pd.DataFrame(dat, columns=catalog_data.columns)  #create a pandas data frame from the array and the headers

    # Extract the ids, radius, Visual Magnitude, ras and decs into arrays
    ra=np.array(catalog_data['ra'])
    dec=np.array(catalog_data['dec'])
    id=np.array(catalog_data['ID'])
    rad=np.array(catalog_data['rad'])    # Star radius
    rad[np.isnan(rad)]=1  # Where there is no radius data, assume size of sol
    vmag=np.array(catalog_data['Vmag'])    # Star visual magnitude
    vmag[np.isnan(vmag)]=min_magnitude-0.1  # Where there is no magnitude data assume it is dim
    teff=np.array(catalog_data['Teff'])   # Temperature
    teff[np.isnan(teff)]=5700  # Where there is no temp data, assume temp of sol
    lum=np.array(catalog_data['lum'])   # Luminosity
    lum[np.isnan(lum)]=1  # Where there is no luminosity data, assume lum = 1
    mass=np.array(catalog_data['mass'])   # Mass
    mass[np.isnan(mass)]=1  # Where there is no mass data, assume same as sol
    GAIA=np.array(catalog_data['GAIA'])

    # Error data
    e_rad=np.array(catalog_data['e_rad'])    # Star radius
    e_rad[np.isnan(e_rad)]=0  # Where there is no radius data, assume +-0
    e_mass=np.array(catalog_data['e_mass'])    # Star radius
    e_mass[np.isnan(e_mass)]=0  # Where there is no mass data, assume +-0
    e_teff=np.array(catalog_data['e_Teff'])    # Star temp
    e_teff[np.isnan(e_teff)]=50  # Where there is no mass data, assume +-0

    dist=np.array(catalog_data['d'])   # Distance
    dist[np.isnan(dist)]=1000  # Where there is no distance, assume 1000
    max_dist=np.array(catalog_data['epos_dist'])   # Max Distance
    max_dist[np.isnan(max_dist)]=1000  # Where there is no distance, assume 1000
    min_dist=np.array(catalog_data['eneg_dist'])   # Min Distance
    min_dist[np.isnan(min_dist)]=1000  # Where there is no distance, assume 1000

    tic_index = np.where(id==str(TIC_no))    # Look for the target in the id array and return its index
    TIC_Teff = teff[tic_index]
    TIC_lum = lum[tic_index]
    TIC_rad = rad[tic_index]
    TIC_mass = mass[tic_index]
    TIC_Vmag = vmag[tic_index]
    GAIA_ID = GAIA[tic_index]
    TIC_dist = dist[tic_index]
    TIC_mind = min_dist[tic_index]
    TIC_maxd = max_dist[tic_index]
    TIC_erad = e_rad[tic_index]
    TIC_emass = e_mass[tic_index]
    TIC_eTeff = e_teff[tic_index]
    TIC_ra = ra[tic_index]
    TIC_dec = dec[tic_index]

    print ('\033[1mExofop {}\033[0m'.format(TIC))
    print ('R* = {:7.4f} (+/-{:7.4f})Rsol  M* = {:7.4f} (+/-{:7.4f}Msol)'.format(*TIC_rad, *TIC_erad, *TIC_mass, *TIC_emass))
    print ('Teff = {:5.0f}K (+/-{:3.0f})K Lum = {:6.4f}'.format(*TIC_Teff, *TIC_eTeff, *TIC_lum))
    print ('Vmag = {:4.2f}  GAIA = {}'.format(*TIC_Vmag, *GAIA_ID))
    print ('Distance = {:7.4f} (-{:7.4f} +{:7.4f})pc'.format(*TIC_dist, *TIC_mind, *TIC_maxd))
    print ('ra = {:9.6f}   dec = {:9.6f}'.format(*TIC_ra, *TIC_dec))
    
    return GAIA_ID, TIC_rad, TIC_erad, TIC_mass, TIC_emass, TIC_Teff, TIC_eTeff, TIC_lum, TIC_Vmag, TIC_dist, TIC_mind, TIC_maxd, TIC_ra, TIC_dec

# Gaia data

def get_gaia_data (gaia_id):
    #coord = SkyCoord(ra=280, dec=-60, unit=(u.degree, u.degree), frame='icrs')
    #width = u.Quantity(0.1, u.deg)
    #height = u.Quantity(0.1, u.deg)
    #r = Gaia.query_object_async(coordinate=coord, width=width, height=height)
    
    query1 = """SELECT 
    TOP 3000
    source_id, 
    ra,
    dec,
    teff_val,
    teff_percentile_lower,
    teff_percentile_upper,
    radius_val,
    radius_percentile_lower,
    radius_percentile_upper,
    lum_val,
    lum_percentile_lower,
    lum_percentile_upper
    FROM gaiadr2.gaia_source
    WHERE source_id=%d""" %gaia_id

    job = Gaia.launch_job(query1)
    results = job.get_results()
    gaia_data = np.array(results)
    gaia_ra = np.array(results['ra'])
    gaia_dec = np.array(results['dec'])
    gaia_rad_dr2 = np.array(results['radius_val'])
    gaia_rad_upper_dr2 = np.array(results['radius_percentile_upper'])
    gaia_rad_lower_dr2 = np.array(results['radius_percentile_lower'])
    gaia_rad_sigma_dr2 = (gaia_rad_upper_dr2-gaia_rad_lower_dr2)/2
    gaia_teff_dr2 = np.array(results['teff_val'])
    gaia_teff_upper_dr2 = np.array(results['teff_percentile_upper'])
    gaia_teff_lower_dr2 = np.array(results['teff_percentile_lower'])
    gaia_teff_sigma_dr2 = (gaia_teff_upper_dr2-gaia_teff_lower_dr2)/2    
    gaia_lum_dr2 = np.array(results['lum_val'])
    gaia_lum_upper_dr2 = np.array(results['lum_percentile_upper'])
    gaia_lum_lower_dr2 = np.array(results['lum_percentile_lower'])
    print('')    
    print('\033[1mGaia DR2\033[0m')
    print('Gaia ra = {:9.6f}  dec = {:9.6f}'.format(*gaia_ra, *gaia_dec))
    print('Radius = {:7.4f} (-{:7.4f} +{:7.4f}) Sigma={:7.4f}'.format(*gaia_rad_dr2, *gaia_rad_dr2-gaia_rad_lower_dr2, *gaia_rad_upper_dr2-gaia_rad_dr2, *gaia_rad_sigma_dr2))
    print('Teff = {:5.0f} (-{:3.0f} +{:3.0f}) Sigma={:3.0f}'.format(*gaia_teff_dr2, *gaia_teff_dr2-gaia_teff_lower_dr2, *gaia_teff_upper_dr2-gaia_teff_dr2, *gaia_teff_sigma_dr2))
    print('Lum = {:7.4f} (-{:7.4f} +{:7.4f})'.format(*gaia_lum_dr2, *gaia_lum_dr2-gaia_lum_lower_dr2, *gaia_lum_upper_dr2-gaia_lum_dr2))

    query2 = """SELECT 
    source_id, 
    ruwe
    FROM gaiadr2.ruwe
    WHERE source_id=%d""" %gaia_id
    job = Gaia.launch_job(query2)
    results = job.get_results()
    #ruwe_data = np.array(results)
    ruwe = np.array(results['ruwe'])
    print('RUWE DR2 (>~1.4 suggests EB) = {:6.4f}'.format(*ruwe))

    query3 = """SELECT 
    source_id, 
    ra,
    dec,
    ruwe,
    astrometric_excess_noise,
    astrometric_excess_noise_sig,
    radial_velocity,
    radial_velocity_error,
    pmra,
    pmdec
    FROM gaiadr3.gaia_source
    WHERE source_id=%d""" %gaia_id
    job1 = Gaia.launch_job(query3)
    results3 = job1.get_results()
    #ruwe_data = np.array(results)
    ra3 = np.array(results3['ra'])
    dec3 = np.array(results3['dec'])
    ruwe3 = np.array(results3['ruwe'])
    astrometric_excess_noise = np.array(results3['astrometric_excess_noise'])
    astrometric_excess_noise_sig = np.array(results3['astrometric_excess_noise_sig'])
    RV=np.array(results3['radial_velocity'])
    RVerr=np.array(results3['radial_velocity_error'])
    pmra=np.array(results3['pmra'])
    pmdec=np.array(results3['pmdec'])
    print('')
    print('\033[1mGaia DR3\033[0m')
    print('Gaia DR3 ra = {:9.6f}  dec = {:9.6f}'.format(*ra3, *dec3))
    print('RUWE DR3 (>~1.4 suggests EB) = {:6.4f}'.format(*ruwe3))
    print('astrometric_excess_noise (higher value means poss EB if aen is significant) = {:6.4f}'.format(*astrometric_excess_noise))
    print('astrometric_excess_noise_sig (>2 aen value is significant) = {:6.4f}'.format(*astrometric_excess_noise_sig))
    print('Radial velocity = {:7.4f} Sigma={:7.4f} km/s'.format(*RV,*RVerr))
    print('Proper Motion ra = {:7.4f} dec = {:7.4f}'.format(*pmra,*pmdec))
    
    query4 = """SELECT 
    classprob_dsc_combmod_star,
    classprob_dsc_combmod_binarystar,
    teff_gspphot,
    teff_gspphot_lower,
    teff_gspphot_upper,
    distance_gspphot,
    distance_gspphot_lower,
    distance_gspphot_upper,
    radius_gspphot,
    radius_gspphot_lower,
    radius_gspphot_upper,
    spectraltype_esphs,
    radius_flame,
    radius_flame_lower,
    radius_flame_upper,
    lum_flame,
    lum_flame_lower,
    lum_flame_upper,
    mass_flame,
    mass_flame_lower,
    mass_flame_upper,
    age_flame,
    age_flame_lower,
    age_flame_upper,
    evolstage_flame,
    classprob_dsc_specmod_star,
    classprob_dsc_specmod_binarystar
    FROM gaiadr3.astrophysical_parameters
    WHERE source_id=%d""" %gaia_id
    
    job2 = Gaia.launch_job(query4)
    results4 = job2.get_results()

    gaia_rad = np.array(results4['radius_flame'])
    gaia_rad_upper = np.array(results4['radius_flame_upper'])
    gaia_rad_lower = np.array(results4['radius_flame_lower'])
    gaia_rad_sigma = (gaia_rad_upper-gaia_rad_lower)/2
    gaia_mass = np.array(results4['mass_flame'])
    gaia_mass_upper = np.array(results4['mass_flame_upper'])
    gaia_mass_lower = np.array(results4['mass_flame_lower'])
    gaia_mass_sigma = (gaia_mass_upper-gaia_mass_lower)/2
    gaia_teff = np.array(results4['teff_gspphot'])
    gaia_teff_upper = np.array(results4['teff_gspphot_upper'])
    gaia_teff_lower = np.array(results4['teff_gspphot_lower'])
    gaia_teff_sigma = (gaia_teff_upper-gaia_teff_lower)/2    
    gaia_lum = np.array(results4['lum_flame'])
    gaia_lum_upper = np.array(results4['lum_flame_upper'])
    gaia_lum_lower = np.array(results4['lum_flame_lower'])
    gaia_distance = np.array(results4['distance_gspphot'])
    gaia_distance_upper = np.array(results4['distance_gspphot_upper'])
    gaia_distance_lower = np.array(results4['distance_gspphot_lower'])    
    gaia_age = np.array(results4['age_flame'])
    gaia_age_upper = np.array(results4['age_flame_upper'])
    gaia_age_lower = np.array(results4['age_flame_lower'])
    prob_star = np.array(results4['classprob_dsc_combmod_star'])
    prob_binary = np.array(results4['classprob_dsc_combmod_binarystar'])
    spectral_type = np.array(results4['spectraltype_esphs'])
    evolstage = np.array(results4['evolstage_flame'])
    rad_gspphot = np.array(results4['radius_gspphot'])
    prob_star_spec = np.array(results4['classprob_dsc_specmod_star'])
    prob_binary_spec = np.array(results4['classprob_dsc_specmod_binarystar'])

    print('')
    print('Radius (flame) = {:7.4f} (-{:7.4f} +{:7.4f}) Sigma={:7.4f} Rsol'.format(*gaia_rad, *gaia_rad-gaia_rad_lower, *gaia_rad_upper-gaia_rad, *gaia_rad_sigma))
    print('Mass = {:7.4f} (-{:7.4f} +{:7.4f}) Sigma={:7.4f} Msol'.format(*gaia_mass, *gaia_mass-gaia_mass_lower, *gaia_mass_upper-gaia_mass, *gaia_mass_sigma))
    print('Teff (flame) = {:5.0f} (-{:3.0f} +{:3.0f}) Sigma={:3.0f} K'.format(*gaia_teff, *gaia_teff-gaia_teff_lower, *gaia_teff_upper-gaia_teff, *gaia_teff_sigma))
    print('Lum = {:7.4f} (-{:7.4f} +{:7.4f})'.format(*gaia_lum, *gaia_lum-gaia_lum_lower, *gaia_lum_upper-gaia_lum))
    print('Distance (flame) = {:7.4f} (-{:7.4f} +{:7.4f}) pc'.format(*gaia_distance, *gaia_distance-gaia_distance_lower, *gaia_distance_upper-gaia_distance))
    print('Age = {:4.2f} (-{:4.2f} +{:4.2f}) GYr'.format(*gaia_age, *gaia_age-gaia_age_lower, *gaia_age_upper-gaia_age))
    print('Radius (gspphot) = {:7.4f} Rsol'.format(*rad_gspphot))
    
    print('')
    print('Probability a star (combmod) = {:4.2f}'.format(*prob_star))
    print('Probability a binary (combmod) = {:4.2f}'.format(*prob_binary))
    print('Probability a star (specmod) = {:4.2f}'.format(*prob_star_spec))
    print('Probability a binary (specmod) = {:4.2f}'.format(*prob_binary_spec))
    print('Spectral type = {}'.format(*spectral_type))
    print('Evolutionary stage = {:3.0f}'.format(*evolstage))
    print('100: zero age main sequence (ZAMS)\n300: first minimum of Teff for massive stars or central hydrogen mass fraction = 0.30 for low-mass stars\n360: main sequence turn-off\n420: central hydrogen mass fraction = 0.00\n490: base of the red giant branch (RGB)\n1290: Tip of the RGB')

    if np.isnan(gaia_rad):
        gaia_rad = gaia_rad_dr2
    if np.isnan(gaia_rad_upper):
        gaia_rad_upper = gaia_rad_upper_dr2
    if np.isnan(gaia_rad_lower):
        gaia_rad_lower = gaia_rad_lower_dr2
    if np.isnan(gaia_rad_sigma):
        gaia_rad_sigma = gaia_rad_sigma_dr2        
    if np.isnan(gaia_teff):
        gaia_teff = gaia_teff_dr2
    if np.isnan(gaia_teff_upper):
        gaia_teff_upper = gaia_teff_upper_dr2
    if np.isnan(gaia_teff_lower):
        gaia_teff_lower = gaia_teff_lower_dr2
    if np.isnan(gaia_teff_sigma):
        gaia_teff_sigma = gaia_teff_sigma_dr2        
    if np.isnan(gaia_lum):
        gaia_lum = gaia_lum_dr2
    if np.isnan(gaia_lum_upper):
        gaia_lum_upper = gaia_lum_upper_dr2
    if np.isnan(gaia_lum_lower):
        gaia_lum_lower = gaia_lum_lower_dr2            

    return gaia_rad, gaia_rad_lower, gaia_rad_upper, gaia_rad_sigma, gaia_teff, gaia_teff_lower, gaia_teff_upper, gaia_teff_sigma, gaia_lum, gaia_lum_lower, gaia_lum_upper, gaia_mass, gaia_mass_lower, gaia_mass_upper, gaia_mass_sigma

def get_simbad_and_EB_data (TIC_no):
        
    # -----------------------
    # Simbad and TESS EB data
    # -----------------------
    TIC = 'TIC ' + TIC_no
    
    from astroquery.simbad import Simbad
    Simbad.add_votable_fields('sptype','otype')
    try:
        result_table = Simbad.query_object(TIC)
        print('')
        print('\033[1mSimbad\033[0m')
        print('Main ID = ' + result_table['MAIN_ID'][0])
        print('Object Type = ' + result_table['OTYPE'][0])
        print('Spectral Type = ' + result_table['SP_TYPE'][0])
        result_table = Simbad.query_objectids(TIC)
        print('Other IDs')
        print(result_table)
    except:
        print('No entry in Simbad')

    #! python -m pip install mastcasjobs
    import mastcasjobs

    query = """SELECT count(ebs.tess_id) as found
    FROM TESS_EBs AS ebs WHERE ebs.tess_id=%s
    """ %TIC_no
    #print(query)
    # user is your MAST Casjobs username
    # pwd is your Casjobs password
    user = "mase22"
    pwd = "Taurus63"
    jobs = mastcasjobs.MastCasJobs(username=user, password=pwd,context='HLSP_TESS_EBs')
    results = jobs.quick(query, task_name="TESS EB search")
    #print(results)
    print('In TESS EB Catalog: ', end="")
    if (results['found'] > 0): 
        print('Yes')
    else:
        print('No')
    return

def plot_star_chart (TIC_no, radius = "115 arcsec", min_magnitude = 22, plot_size = 16):
    # Radius - How far round the target star to look.  Each TESS pixel is 21 arcsecs square
    TIC = 'TIC ' + TIC_no
    catalog_data = Catalogs.query_criteria(catalog="Tic",objectname=TIC,radius=radius,objType="STAR")
    dat = np.array(catalog_data)  #extract table data into an array
    df = pd.DataFrame(dat, columns=catalog_data.columns)  #create a pandas data frame from the array and the headers
    
    plt.style.use('dark_background')

    # Define some factors for the plot
    scale_stars = 25   # This scales the stars up so they are more visible on the chart
    sigma = 5.670374419e-8  # W⋅m−2⋅K−4  

    # Extract the ids, radius, Visual Magnitude, ras and decs into arrays
    ra=np.array(catalog_data['ra'])
    dec=np.array(catalog_data['dec'])
    id=np.array(catalog_data['ID'])
    rad=np.array(catalog_data['rad'])    # Star radius
    rad[np.isnan(rad)]=0  # Where there is no radius data, set to 0
    vmag=np.array(catalog_data['Vmag'])    # Star visual magnitude
    vmag[np.isnan(vmag)]=min_magnitude-0.1  # Where there is no magnitude data assume it is dim
    teff=np.array(catalog_data['Teff'])   # Temperature
    teff[np.isnan(teff)]=5700  # Where there is no temp data, assume temp of sol
    lum=np.array(catalog_data['lum'])   # Luminosity
    lum[np.isnan(lum)]=1  # Where there is no luminosity data, assume lum = 1
    mass=np.array(catalog_data['mass'])   # Mass
    mass[np.isnan(mass)]=0  # Where there is no mass data, set to 0
    GAIA=np.array(catalog_data['GAIA'])
    dist=np.array(catalog_data['d'])
    dist[np.isnan(dist)]=0  # Where there is no distance data, set to 0

    tic_index = np.where(id==str(TIC_no))    # Look for the target in the id array and return its index
    TIC_Teff = teff[tic_index]
    TIC_lum = lum[tic_index]
    TIC_rad = rad[tic_index]
    if TIC_lum == 0:
        TIC_lum = (4 * np.pi * (TIC_rad*696.34e6)**2) * sigma * (TIC_Teff**4)/3.827e+26

    # Plot them on a scatter chart
    fig, axs = plt.subplots(figsize=(plot_size+6, plot_size))
    plt.gca().invert_xaxis()   # Plot dec in reverse if needed
    #plt.gca().invert_yaxis()   # Plot dec in reverse if needed

    sc = plt.scatter(ra, dec, s=(min_magnitude-vmag)*scale_stars, c = np.log(teff), cmap = 'RdYlBu', alpha = 0.9, zorder = -1, edgecolors = 'white', linewidths = 0.1) #Plot by magnitude
    #plt.scatter(ra, dec, s=(rad)*scale_stars, c = teff, cmap = 'RdYlBu', alpha = 0.9, zorder = -1, edgecolors = 'white', linewidths = 0.1) #Plot by radius

    # Add a highlight on the TIC of interest
    plt.scatter(ra[tic_index],dec[tic_index],s=(min_magnitude-vmag[tic_index])*scale_stars, c='white', alpha = 0.8, marker='*', zorder = -1, edgecolors = 'black', linewidths = 0.5)
    #plt.scatter(ra[tic_index],dec[tic_index],s=(rad[tic_index])*scale_stars, c='white', alpha = 0.5, marker='*', zorder = -1, edgecolors = 'black', linewidths = 0.5)

    #Temperature range
    #sc.set_clim(2500, 10000)
    sc.set_clim(np.log(2500), np.log(20000))
    #plt.colorbar(sc)
    #Axes
    #axs.set_xlim(ra[tic_index]-0.1,ra[tic_index]+0.1)
    #axs.set_ylim(dec[tic_index]-0.032,dec[tic_index]+0.032)

    # Add labels and gridlines
    plt.xlabel('ra')
    plt.ylabel('dec')
    plt.grid(color='gray')

    # Annotate each point with its TIC
    for i, txt in enumerate(id):
        axs.annotate(txt, (ra[i], dec[i]),color='grey',fontsize=10,textcoords="offset points",xytext=(7,3))  # Test is offest from the point by the xytext value
    for i, txt in enumerate(rad):
        distance = dist[i]
        axs.annotate("R:{:0.2f} d:{:0.0f}".format(txt, distance), (ra[i], dec[i]),color='Orange',fontsize=10,textcoords="offset points",xytext=(7,-7))  # Test is offest from the point by the xytext value
        
    return TIC_Teff, TIC_lum, TIC_rad    

def plot_HR(TIC_no, TIC_Teff, TIC_lum, TIC_rad, plot_size = 16, TOI = False):
    
    # import the data using a module called Pandas (I love Pandas as it's an easy tool to import and work with data using python)
    data = pd.read_csv("./Gaia_kepler_stellar_properties_tab2.csv")
    TOI_data = pd.read_csv("TOIs.txt")    
 
    plt.style.use('dark_background')
    
    # set up the plotting region
    fig, ax = plt.subplots(figsize=(plot_size,plot_size))
    
    # plot all of the stars, where the size of the symbol scales with the radius of the star and the colour with the temperature
    sc = plt.scatter(data.Teff, 10**(data.logLsun), s = 5*data.Rsun, c = np.log(data.Teff), cmap = 'RdYlBu', alpha = 0.9, zorder = -1, edgecolors = 'black', linewidths = 0.1)
    sc.set_clim(np.log(2500), np.log(20000))
    sun_marker_color = 'k'
    
    if TOI == True:
        sigma = 5.670374419e-8  # W⋅m−2⋅K−4
        TOI_data['Luminosity'] = (4 * np.pi * ((TOI_data['Stellar Radius (R_Sun)']*696.34e6)**2) * sigma * (TOI_data['Stellar Eff Temp (K)']**4))/3.827e+26
        plt.scatter(TOI_data['Stellar Eff Temp (K)'], TOI_data['Luminosity'], s = 5*TOI_data['Stellar Radius (R_Sun)'], c = 'green', cmap = 'RdYlBu', alpha = 0.9, zorder = -1, edgecolors = 'black', linewidths = 0.1)
        sun_marker_color = 'white'

    print ('TIC: ' + str(TIC_no), 'T: ' + str(TIC_Teff), 'L: ' + str(TIC_lum), 'R: ' +  str(TIC_rad))
    sc_target = plt.scatter(TIC_Teff, TIC_lum, s = 250, c = np.log(TIC_Teff), cmap = 'RdYlBu', marker='*', alpha = 0.9, zorder = -1, edgecolors = 'black', linewidths = 0.4) #Add the TIC of interest
    sc_target.set_clim(np.log(2500), np.log(20000))
    ax.annotate(TIC_no, (TIC_Teff, TIC_lum), size=15, color = 'Green', textcoords="offset points",xytext=(7,-7))
    
    # plot the properties of the Sun using a symbol in the shape of a star
    plt.scatter(5778,1, color = sun_marker_color, marker = '*', s = 200, zorder = 1)
        
    # invert the x axis (this is standard for HR diagrams) and make the y axis log scale
    ax.invert_xaxis()
    ax.set_yscale('log')
    
    # label the axes
    ax.set_xlabel("Temperature (Kelvin)", fontsize = 20)
    ax.set_ylabel("Luminosity (Solar unit)", fontsize = 20)
    ax.tick_params(axis="both", direction="inout", labelsize = 18)
    
    # -- -- -- -- Add labels -- -- -- -- --
    # label the Sun
    ax.text(6200,1.5, 'Sun', size=20, color = sun_marker_color)
    
    #label the main sequence
    ax.text(9500,0.2, 'main sequence', size=20, color = 'white', rotation=-47)
    
    # label red giants
    ax.text(7000,1e4, 'red giants', size=20, color = 'white')
    
    ax.text(4200, 1e-3, 'cooler', size=25, color = 'red')
    ax.text(19550, 1e-3, 'hotter', size=25, color = 'dodgerblue')
    
    
    ax.text(18200, 10**-1.46, '1 Solar radii', size=14, color = 'white')
    plt.scatter(18900, 10**-1.4, s = 5*1, c = 'white', edgecolors = 'black', linewidths = 0.1)
    
    ax.text(18200, 10**-1.76, '10 Solar radii', size=14, color = 'white')
    plt.scatter(18900, 10**-1.7, s = 5*10, c = 'white', edgecolors = 'black', linewidths = 0.1)
    
    ax.text(18200, 10**-2.17, '100 Solar radii', size=14, color = 'white')
    plt.scatter(18900, 10**-2.1, s = 5*100, c = 'white', edgecolors = 'black', linewidths = 0.1)
    
    plt.ylim(10**(-3.1), 10**4.5)
    plt.xlim(20000,1200)
    plt.tight_layout()
    plt.show()
    
def add_gaia_figure_elements(TIC_no, tpf, ax, magnitude_limit=18, zero_origin=False):
    from astropy.coordinates import SkyCoord, Angle
    import astropy.units as u
    import numpy as np
    from astroquery.vizier import Vizier
    from astroquery.mast import Catalogs
    
    """Make the Gaia Figure Elements"""
    # Get the positions of the Gaia sources
    c1 = SkyCoord(tpf.ra, tpf.dec, frame='icrs', unit='deg')
    # Use pixel scale for query size
    pix_scale = 4.0  # arcseconds / pixel for Kepler, default
    if tpf.mission == 'TESS':
        pix_scale = 21.0
    # We are querying with a diameter as the radius, overfilling by 2x.
    
    Vizier.ROW_LIMIT = -1
    result = Vizier.query_region(c1, catalog=["I/345/gaia2"],
                                 radius=Angle(np.max(tpf.shape[1:]) * pix_scale, "arcsec"))
    no_targets_found_message = ValueError('Either no sources were found in the query region '
                                          'or Vizier is unavailable')
    too_few_found_message = ValueError('No sources found brighter than {:0.1f}'.format(magnitude_limit))
    if result is None:
        raise no_targets_found_message
    elif len(result) == 0:
        raise too_few_found_message
    result = result["I/345/gaia2"].to_pandas()
    result = result[result.Gmag < magnitude_limit]
    if len(result) == 0:
        raise no_targets_found_message
    radecs = np.vstack([result['RA_ICRS'], result['DE_ICRS']]).T
    coords = tpf.wcs.all_world2pix(radecs, 0) ## TODO, is origin supposed to be zero or one?

    # Gently size the points by their Gaia magnitude
    sizes = np.array(64.0 / 2**(result['Gmag']/5.0))
    i=0
    if zero_origin == True:  # For plots that do not have the tpf grid applied - eg in/out plot
        x0=0
        y0=0
    else:
        x0=tpf.column
        y0=tpf.row

    for x in coords:
        size=sizes[i]/100
        circle1 = plt.Circle((x0+x[0],y0+x[1]),size,color="white")
        ax.add_patch(circle1)
        i+=1
        
    TIC = 'TIC ' + TIC_no
    radius = "1 arcsec"
    catalog_data = Catalogs.query_criteria(catalog="Tic",objectname=TIC,radius=radius,objType="STAR")

    # Extract the ids, radius, Visual Magnitude, ras and decs into arrays
    ra=np.array(catalog_data['ra'])
    dec=np.array(catalog_data['dec'])        
    radect= np.vstack([ra, dec]).T
    coordt = tpf.wcs.all_world2pix(radect, 0)
 
    magt = catalog_data['Vmag']
    sizet = 64.0 / 2**(magt/5.0)
    circle1 = plt.Circle((x0+coordt[0][0],y0+coordt[0][1]),0.5,color="orange",fill=False)
    ax.add_patch(circle1)    
    return coords,sizes

def add_gaia_stars(tpf, ax, magnitude_limit=18):
    from astropy.coordinates import SkyCoord, Angle
    import astropy.units as u
    import numpy as np
    from astroquery.vizier import Vizier
    from astroquery.mast import Catalogs
    
    """Make the Gaia Figure Elements"""
    # Get the positions of the Gaia sources
    c1 = SkyCoord(tpf.ra, tpf.dec, frame='icrs', unit='deg')
    # Use pixel scale for query size
    pix_scale = 4.0  # arcseconds / pixel for Kepler, default
    if tpf.mission == 'TESS':
        pix_scale = 21.0
    # We are querying with a diameter as the radius, overfilling by 2x.
    
    Vizier.ROW_LIMIT = -1
    result = Vizier.query_region(c1, catalog=["I/345/gaia2"],
                                 radius=Angle(np.max(tpf.shape[1:]) * pix_scale, "arcsec"))
    no_targets_found_message = ValueError('Either no sources were found in the query region '
                                          'or Vizier is unavailable')
    too_few_found_message = ValueError('No sources found brighter than {:0.1f}'.format(magnitude_limit))
    if result is None:
        raise no_targets_found_message
    elif len(result) == 0:
        raise too_few_found_message
    result = result["I/345/gaia2"].to_pandas()
    result = result[result.Gmag < magnitude_limit]
    if len(result) == 0:
        raise no_targets_found_message
    radecs = np.vstack([result['RA_ICRS'], result['DE_ICRS']]).T
    coords = tpf.wcs.all_world2pix(radecs, 0) ## TODO, is origin supposed to be zero or one?

    # Gently size the points by their Gaia magnitude
    sizes = np.array(64.0 / 2**(result['Gmag']/5.0))
    i=0
    for x in coords:
        size=sizes[i]/100
        circle1 = plt.Circle((tpf.column+x[0],tpf.row+x[1]),size,color="white")
        ax.add_patch(circle1)
        i+=1

    return coords,sizes
