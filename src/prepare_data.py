#-----------------------------------------------------------
#                       prepare_data.py
#  This file contains all the variable initializations,
#  both for RV and Transit fittings.
#                   O. Barragan, March 2016
#-----------------------------------------------------------

nconv    = niter
nwalkers = nchains
#What transit data are we fitting
if ( lc_data == 'kepler_lc' ):
  n_cad = 10
  t_cad = 29.425 / 60. / 24.0 #days
elif ( lc_data == 'kepler_sc' ):
  n_cad = 1
  t_cad = 1.5 / 60. / 24.0 #days
elif ( lc_data == 'tess_sc' ):
  n_cad = 10
  t_cad = 2.0 / 60. / 24.0 #days
elif ( lc_data == 'free' ):
  #values given by the user
  n_cad = n_cad
  t_cad = t_cad

#-----------------------------------------------------------
#                         RV DATA
#-----------------------------------------------------------

#Let us check the kind of variable
nplanets_rv = 0
for o in range(0,len(fit_rv)):
  nplanets_rv = nplanets_rv + int(fit_rv[o])

#Let us ensure that we want to fit rv data
if ( nplanets_rv > 0 ):

  #Read the data file
  #time, RV, errors, and Telescope label
  time,rv,err,tspe = np.loadtxt('inpy/'+star+'/'+fname_rv[0],usecols=(0,1,2,3), \
  dtype={'names': ('time', 'rv', 'err','telescope'), \
    'formats': ('float', 'float', 'float', 'S10')}, \
    comments='#',unpack=True)
  if ( is_special_jitter ):
    sjitter = np.loadtxt('inpy/'+star+'/'+fname_rv[0],usecols=(4),dtype=str,unpack=True)



  #These lists have lists with data for
  #the different telescopes
  time_all = []
  rv_all   = []
  errs_all = []

  #Number of telescopes
  nt = len(telescopes)

  if ( nt < 1 ):
    print 'Please, indicate the telescope labels!'
    sys.exit('')

  #Separate the data for each telescope and create labels
  for i in range(0,nt):
    time_dum = []
    rv_dum   = []
    errs_dum = []
    for j in range(0,len(tspe)):
      if (tspe[j] == telescopes[i]):
        time_dum.append(time[j])
        rv_dum.append(rv[j])
        errs_dum.append(err[j])
  #The *all variables are lists of lists, each list constains
  # a list with the data of each telescope
    time_all.append(time_dum)
    rv_all.append(rv_dum)
    errs_all.append(errs_dum)

  #The mega* variables contains all the data
  #All this is neccesary because you do not have
  #the same number of data for each telescope
  mega_rv   = []
  mega_time = []
  mega_err  = []
  tlab      = []
  jrvlab    = []
  #create mega with data of all telescopes
  for i in range(0,nt):
    #fill the mega variable with all the data of the
    #telescope i
    for j in range(0,len(rv_all[i])):
      #tlab has the label of the telescope (an integer)
      #this is useful because matches with the index of
      #the mega variables
      tlab.append(i)
      mega_rv.append(rv_all[i][j])
      mega_time.append(time_all[i][j])
      mega_err.append(errs_all[i][j])

  if ( is_special_jitter ):
    n_jrv = len(jrvvec)
    for o in range(0,len(tlab)):
      for ndum in range(0,len(jrvvec)):
        if (sjitter[o][0] == jrvvec[ndum][0] ):
          jrvlab.append(ndum)
          break
  else:
    jrvlab = list(tlab)
    n_jrv = nt



  total_rv_fit = True

else:
  n_jrv = 1
  nt = 1
  tlab = [0]
  jrvlab = list(tlab)
  mega_rv = [None]
  mega_time = [None]
  mega_err  = [None]
  total_rv_fit = False
  is_jitter_rv = False

#RV DATA READY

#-----------------------------------------------------------
#                     TRANSIT DATA
#-----------------------------------------------------------

#Let us check the kind of variable
nplanets_tr = 0
for o in range(0,len(fit_tr)):
  nplanets_tr = nplanets_tr + int(fit_tr[o])

if ( nplanets_tr > 0 ):

  #Each transit planet hasa different file
  myn = len(fname_tr)
  xt= [None]*myn
  yt= [None]*myn
  et= [None]*myn


  for o in range(0,myn):

    filename = 'inpy/'+star+'/'+fname_tr[o]
    if ( my_tr_err == 0 ): #The error bars come from the input file
        dummyd,dummyf,dummye = np.loadtxt(filename,usecols=columns_tr, \
        comments='#',unpack=True)
    else: #The error bars are given in the input
        dummyd,dummyf = np.loadtxt(filename,usecols=[0,1], \
        comments='#',unpack=True)
        dummye = [my_tr_err]*len(dummyd)

    dummyd = dummyd + textra

    hdate = dummyd
    wflux = dummyf
    errs  = dummye

    #Each element of these lists will have the information
    #of a given transit
    xt[o]= hdate
    yt[o]= wflux
    et[o]= errs


  #Let us put together the information of all the arrays
  #the mega* lists have the data of all the transits
  #in 1D array
  megax = np.concatenate(xt)
  megay = np.concatenate(yt)
  megae = np.concatenate(et)
  megap = [0]*len(megax)

  total_tr_fit = True

else:
  megax = [1.]
  megay = [1.]
  megae = [1.]
  megap = [0]
  total_tr_fit = False
  is_jitter_tr = False
  fit_q1 = 'f'
  fit_q2 = 'f'

#TRANSIT DATA READY
#Take care with span_tr
if ( len(span_tr) == 1 and nplanets > 1): #The user did not change this option in input_file.py
  span_tr = [0.0]*nplanets

if ( len(a_from_kepler) == 1 and nplanets > 1):
  a_from_kepler = [False]*nplanets

#CHECK WHAT WE HAVE TO FIT
#If we are not going to fit RV or TR data, let us turn off the variables
#for the given case
for o in range(0,nplanets):
    if (fit_tr[o] == False ):
      fit_rp[o] = 'f'
      min_rp[o] = 0.0
      if ( not is_b_factor ):
        fit_i[o]  = 'f'
        min_i[o]  = np.pi/2.0
      else:
        fit_b[o]  = 'f'
        min_b[o]  = 0.0
      fit_a[o]  = 'f'
    if (fit_rv[o] == False ):
      fit_k[o]  = 'f'

if is_single_transit:
    for o in range(0, nplanets):
        fit_P[o] = 'f'
        min_P[o] = 10*(max(megax)-min(megax))
        fit_a[o] = 'u'
        min_a[o] = 1.1
        max_a[o] = 1e3
        plot_binned_data = False

#Let us turn off velocity offset for a pure TR fit
if ( not total_rv_fit ):
  fit_v0 = 'f'
  nt = 1
  min_rv0 = [-1.]
  max_rv0 = [1.]
  min_phys_rv0 = [-1.]
  max_phys_rv0 = [1.]
  rvs = [0.0]
  telescopes = ['O']
  telescopes_labels = ['']

if ( is_den_a ): #For a multiplanet system the density has to be the same
  if ( nplanets > 1 ):
    for o in range(1,nplanets):
      fit_a[o] = 'f'
