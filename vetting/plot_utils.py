import numpy as np
import os
import astropy.io.fits as pf
import csv
import pandas as pd
def momentum_dumps_info(datafileroot):
    '''
    function to the create a list of all of the momentum dump times for each sector
    '''
    print (" - - - - - - - -")
    print ("We need to store the times of the momentum dumps for each sector. \
        this will quickly open one file from each sector (no files are stored). \
        This will take a minute.")
    print()
        
    md_filename = datafileroot / "tess_mom_dumps.txt"
    tempfile = datafileroot / "temp.txt"
    sectorfile = datafileroot / "sector_download_codes.txt"        

    if not md_filename.exists():
        with open(md_filename,'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['sec', 'time'])
        first_sec = 0 # start with sector 1 but this has to be 0 because the next step of the code adds one (needs to be like this otherwise it will dowload the last sector multiple times when re run)
        print ("Will determine the times of the momentum dumps for all sectors and store them... ")

    else:
        if os.name =='nt':
            os.system(r'powershell.exe "Get-Content -Path \\wsl.localhost\Ubuntu\home\ian\LATTE_output\data\tess_mom_dumps.txt -Tail 1 | Set-Content \\wsl.localhost\Ubuntu\home\ian\LATTE_output\data\temp.txt"')
        else:
            os.system('tail -n 1 /home/ian/LATTE_output/data/tess_mom_dumps.txt > /home/ian/LATTE_output/data/temp.txt')

        with open(tempfile, 'r') as f:
            string = f.readlines()#[-1]
            if string == ['sec\ttime\n']:
                first_sec = 0 # start with sector 1 but this has to be 0 because the next step of the code adds one (needs to be like this otherwise it will dowload the last sector multiple times when re run)
            else:
                first_sec = int(string[0][0:2])
    print("First {}".format(first_sec))
    infile = pd.read_csv(sectorfile, delimiter = ' ', usecols=[0,1,2],names = ['sec', 'first', 'second'], comment = '#')

    for sec in range(first_sec+1,500): # a large number - how many TESS sectors will there be?

        try:
            if sec < 10:
                download_sector = "00{}".format(sec)
            else:
                download_sector = "0{}".format(sec)

            # load the data file of the sector that the first marked transit appears in
            # sort the list to be ordered by camera, RA and then Dec
            alltargets = datafileroot / "all_targets_S{}_v1.txt".format(download_sector)
            tic_list = pd.read_csv(alltargets)

            # select the first tic listed for that sector - the momentum dumps are the same for all tic ids - they are sector dependent.
            tic =  list(tic_list['TICID'])[0]
            
            this_sector_code = infile[infile.sec == int(sec)]

            sector = str(sec)
            print(sector)
            download_url = (
                "https://mast.stsci.edu/api/v0.1/Download/file/?uri=mast:TESS/product/tess"
                + str(this_sector_code['first'].values[0]).rjust(13, "0")
                + "-s"
                + sector.rjust(4, "0")
                + "-"
                + str(tic).rjust(16, "0")
                + "-"
                + str(this_sector_code['second'].values[0]).rjust(4, "0")
                + "-s_lc.fits")

            response = requests.get(download_url)
            lchdu  = pf.open(response.url) # This needs to be a URL - not a file

            # Open and view columns in lightcurve extension
            lcdata = lchdu[1].data
            quality = lcdata['QUALITY']
            time    = lcdata['TIME']

            lchdu.close()

            mom_dump_mask = np.bitwise_and(quality, 2**5) >= 1

            momdump = (list(time[mom_dump_mask]))
            sector = list([sec] * len(momdump))

            with open(md_filename, 'a', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerows(zip(sector,momdump))

        except:
            print ("Done")
            break
            

def get_mds_from_file(datafileroot):
    # Get all momentum dump times from the LATTE data files - don't forget to refresh using LATTE when a new sector becomes available
    import csv
    md = []
    try:
        with open(datafileroot / "tess_mom_dumps.txt",'r',newline='') as f:
            lines = csv.DictReader(f, delimiter='\t')  #  Use the dictionary reader to create pairs of values from the file
            for line in lines:
                md.append(dict(line))  #  Add dictionary entry to the array
        f.close()
    except IOError as e:
        print(e)
    return md

def get_md(sector,md):
    '''
    Function to retrieve the dump times for this sector
    '''
    md_sec = list(filter(lambda m: m['sec'] == str(sector), md)) #  Create a list of just the dump times for this sector
    md_times = []
    for mdi in md_sec:
        md_times.append(float(mdi.get('time')))  # Extract the times only from the list and convert them to real numbers for plotting
    return md_times

def get_mds(ax,md):
    '''
    Function to retrieve the dump times between x limits
    '''
    xlimits = ax.get_xlim()
    xmin = xlimits[0]
    xmax = xlimits[1]
    #md_sec = list(filter(lambda m: m['sec'] == str(sector), md)) #  Create a list of just the dump times for this sector
    md_times = []
    for mdi in md:
        mdtime = float(mdi.get('time'))
        if mdtime > xmin and mdtime < xmax:
            md_times.append(mdtime)  # Extract the times only from the list and convert them to real numbers for plotting
    return md_times    

def mark_events (ax, lce, md, sector_no=0, marker_len = 0.1):
    '''
    Mark quality events
    Bit 1. Attitude Tweak
    Bit 2. Safe Mode
    Bit 3. Coarse Point
    Bit 4. Earth Point
    Bit 5. Argabrightening Event (Sudden brightening across the CCD.)
    Bit 6. Reaction Wheel Desaturation
    Bit 8. Manual Exclude
    Bit 10. Impulsive outlier
    Bit 12. Straylight detected
    '''
    
    ylimits = ax.get_ylim()
    ymin = ylimits[0]
    ymax = ylimits[1]
    
    ydiff = ymax-ymin
    yhigh = ymin+(ydiff*marker_len) # Marker lines just at the bottom of the graph so not as intrusive

    quality = lce['quality']
    ttime = lce['time'].jd - 2457000

    stray = np.bitwise_and(quality, 2**11) >= 1 
    desat = np.bitwise_and(quality, 2**5) >= 1 
    bright = np.bitwise_and(quality, 2**4) >= 1 
    att = np.bitwise_and(quality, 1) >= 1 

    ax.vlines(ttime[stray],ymin,yhigh, colors = 'gold', label = "Stray light")
    ax.vlines(ttime[desat],ymin,yhigh, colors = 'lightgreen', label = "Reac Wl Desaturation")
    ax.vlines(ttime[bright],ymin,yhigh, colors = 'lightblue', label = "Brightening")
    ax.vlines(ttime[att],ymin,yhigh, colors = 'pink', label = "Attitude Tweak")
    
    mom_dumps = get_mds(ax,md)
    ax.vlines(mom_dumps,ymin,yhigh, colors = 'lightgrey', label = "Momentum Dump",alpha=0.3)    
    #ax.legend(loc = 'upper left')
    
def add_sector_labels (ax):
    # Add sector labels
    secname, sectime = np.loadtxt("Sectors.txt", delimiter=',', unpack=True, dtype=[('myint','i8'),('myfloat','f8')])

    #ylimits = ax.get_ylim()
    #i=0
    #for txt in secname:
    #    x=sectime[i]
    #    label="Sec " + str(txt)
    #    ax.annotate(label,(x,ylimits[0]),c="darkgray")
    #    i+=1
        
    ax2 = ax.twiny()
    xlimits = ax.get_xlim()
    i=0
    x=[]
    y=[]
    xmin=xlimits[0]
    xmax=xlimits[1]
    for txt in secname:
        if sectime[i] >= xmin and sectime[i] <= xmax:
            x.append(sectime[i])
            y.append("S" + str(txt))
        i+=1
    ax2.set_xlim(xmin,xmax)
    ax.set_xticks(x)
    ax.set_xticklabels(y)
    
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('left')
   