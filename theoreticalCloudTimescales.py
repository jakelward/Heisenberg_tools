import numpy as np
#theoreticalCloudTimescales v0.3
#A collection of functions to implement the general theory of molecular cloud lifetimes of Jeffreson & Kruijssen (2018)
#The aim here is to provide a set of simple python functions for predicting mean molecular (and atomic) cloud lifetimes
#For maximum accessability all functions only require numpy as a pre-requisite.
#Note that the current script has been tested on python 2.7.14. and python 3.6.6
#If you have any problems, please raise an issue at https://github.com/JLWard-90/Heisenberg_tools and I'll see what I can do.
#For usage see the example script "predict_timescales_example.py"

def G():
    return 4.30091E-3 #In pc / M_o (km/s)^2 ~ pc / M_o (pc/Myr)^2

def combineTimescales(timescales,printTimes=False):
    t_out_inv = 0.
    if printTimes == True:
        print(timescales)
    for t in timescales:
        t_out_inv = t_out_inv + (1./t)
    tout = np.abs(1. / t_out_inv)
    return tout

def shearParameter(R_array, V_array): #R_array: Array of radius values, V_array: Array of velocity values. 
    #shearParameter returns an array of size n-1 where input arrays are of size n. The input arrays should therefore be considered as bin edges
    #Shear parameter is defined as the differential of the rotation curve of a galaxy
    Beta = np.diff(np.log(V_array)) / np.diff(np.log(R_array))
    return Beta

def ToomreQ(veldisp, ep_frequency, surface_density):
    Q = (veldisp * ep_frequency) / (3.36*G()*surface_density)
    return Q

def getOmega(vel, radius):
    Omega = vel / (radius)
    return Omega

def EpFreq(R_array,V_array):
    omega_array = []
    rsqrd_omega_array = []
    for i in range(len(R_array)):
        omega_array.append(getOmega(V_array[i],R_array[i]))
        rsqrd_omega_array.append(R_array[i]**2.*omega_array[i])
    mean_diff = np.mean(np.diff(rsqrd_omega_array) / np.diff(R_array))
    ksqr = ((2*np.mean(omega_array)) / np.mean(R_array)) * mean_diff
    k = np.sqrt(ksqr)
    return k

#Q = Toomre Q, Omega = orbital speed (from rotation curve), Beta = shear parameter,Sigma_star = stellar surface density, Sigma_gas = gas surface density, veldisp_star = stellar velocity dispersion, veldisp_gas = gas velocity dispersion
def freeFallTime(Q,Omega,Beta,Sigma_star,Sigma_gas,veldisp_star,veldisp_gas): 
    phi_p = 1 + ((Sigma_star/Sigma_gas)*(veldisp_gas/veldisp_star)) #phi_p is from Elmegreen (1989)
    tff = np.sqrt((3.*np.pi**2.)/(32.*phi_p*(1+Beta))) * (Q / Omega) #Jeffreson 2018 eqn. 5
    return tff

def cloudCloudCollisionsTime(Q, fg, Omega,Beta): #Q: Toomre Q, fg: probability of collision in any given encounter
    if Beta > 0.12: #See Jeffreson 2018, Section 2.2
        Beta_adjusted = 0.12
    else:
        Beta_adjusted = Beta
    tcc = (2*np.pi*Q) / (9.4*fg*Omega*(1+0.3*Beta_adjusted)*(1-Beta_adjusted)) #This eqn is from Tan 2000
    return tcc

def spiralArmTimescale(Omega,Omega_P,m): #m: number of spiral arms, Omega: Orbital speed, Omega_P: Pattern speed
    if m == 0:
        tps = 1000000 #[Myr] If there are no spiral arms, set timescale to be longer than a Hubble time
    else:
        tps = (2*np.pi)/(m * Omega * np.abs(1-(Omega_P/Omega)))
    return tps

def epicyclicPerturbationTimescale(Omega, Beta): #Omega: orbital speed, Beta: shear parameter
    tk = ((4*np.pi) / (Omega * np.sqrt(2*(1+Beta)))) * (1 / np.sqrt(3+Beta)) #See Jeffreson 2018 Section 2.5
    return tk

def shearTimescale(Beta, Omega): #Beta: Shear parameter, Omega: orbital speed
    tB = 2. / (Omega * (1-Beta))
    return tB * (-1.0)

def calculateGalaxyWideTimescale(fg, V_array, R_array, surface_density_star, surface_density_gas, veldisp_star, veldisp_gas, Omega_p, m):
    shearArray = shearParameter(R_array,V_array)
    avgBeta = np.mean(shearArray)
    Omega = getOmega(np.median(V_array), np.median(R_array))
    freqep = EpFreq(R_array,V_array)
    tQ = ToomreQ(veldisp_gas,freqep,surface_density_gas)
    tff = freeFallTime(tQ,Omega,avgBeta,surface_density_star,surface_density_gas,veldisp_star,veldisp_gas)
    tcc = cloudCloudCollisionsTime(tQ,fg,Omega,avgBeta)
    tep = epicyclicPerturbationTimescale(Omega,avgBeta)
    tB = shearTimescale(avgBeta,Omega)
    tsp = spiralArmTimescale(Omega,Omega_p,m)
    t = combineTimescales([tff,tcc,tep,tsp,tB])
    #t = combineTimescales([tff,tcc])
    return t

#When calculating a single galactic average, set include_edges = False, when plotting timescales as a function of radius, set to True.
def SetupBins(R_array, V_array, binwidth=1000.0, maximum=4000., minimum=0., include_edges=True): #Bin width, maximum, and minimum in pc
    if(np.max(R_array) < 100):
        print ("Guessing that radius values are in kpc. Converting to pc...")
        R_array *= 1000.
    maxR = np.max(R_array) - (np.max(R_array) % binwidth)
    if maxR > maximum:
        maxR = maximum
    N_bins = int(maxR / binwidth)
    binArray = np.zeros(N_bins)
    for i in range(N_bins):
        binArray[i] = minimum + (i*binwidth) + (binwidth / 2.)
    Vbinned = np.zeros(N_bins)
    countArray = np.zeros(N_bins)
    if(include_edges==True):
        for i in range(len(R_array)):
            for j in range(N_bins):
                if (R_array[i] >= j * binwidth) and (R_array[i] <= (j+1) * binwidth):
                    Vbinned[j] = Vbinned[j] + V_array[i]
                    countArray[j] += 1
                    break
    else:
        for i in range(len(R_array)):
            for j in range(N_bins):
                if (R_array[i] > minimum +(j * binwidth)) and (R_array[i] < minimum +((j+1) * binwidth)):
                    Vbinned[j] = Vbinned[j] + V_array[i]
                    countArray[j] += 1
                    break
    Vbinned / countArray
    return binArray,Vbinned

def readInData(input_file): #Reads in data in the form R,V,Verr with delimiter = ',' and three lines of header
    data = np.genfromtxt(input_file,delimiter=',',skip_header=3)
    R_array = data[:,0]
    V_array = data[:,1]
    Verr_array = data[2,:]
    return R_array,V_array,Verr_array

def freeFallTimePerBin(V_array, R_array,sigma_star,sigma_gas,veldisp_star,veldisp_gas):
    #Need to allow sigma gas to be input as an array of length R_array or length shearArray (R_array-1)
    try:
        length_sigma = len(sigma_gas) #Try to measure the length of sigma gas
        sigma_is_array = True
    except:
        sigma_is_array = False #If it doesn't have a length then it is probably (hopefully) a float
    shearArray = shearParameter(R_array,V_array)
    try:
        length_veldisp_gas = len(veldisp_gas)
        veldispgas_is_array = True
    except:
        veldispgas_is_array = False

    try:
        length_veldisp_star = len(veldisp_star)
        veldispstar_is_array = True
    except:
        veldispstar_is_array = False
    #Also need to do the same for sigma star
    try:
        length_sigma_star = len(sigma_star)
        sigma_star_is_array = True
    except:
        sigma_star_is_array = False
    
    tff_array = []
    for i in range(len(shearArray)):
        R = (R_array[i]+R_array[i+1])/2.
        V = (V_array[i]+V_array[i+1])/2.
        Omega = getOmega(V,R)
        freqep = EpFreq([R_array[i],R_array[i+1]],[V_array[i],V_array[i+1]])
        if sigma_is_array == True: #If there is a surface density array then we need to check if, and how, we can use it
            if length_sigma == len(shearArray):
                sgas = sigma_gas[i] #If it is the same length as shear array, then we assume that it is input in the correct format and we can just use the corresponding value
            elif length_sigma == len(shearArray)+1:
                sgas = (sigma_gas[i]+sigma_gas[i+1]) / 2. #if it has a length of that of the shear array then we assume that it is in the same format as R_array and V_array. If so we take an average of the values surrounding the one that we actually want.
            else:
                print ("Error encountered in freeFallTimePerBin: Gas surface density array (sigma_gas) is wrong length. length must equal len(shearArray) or len(shearArray)+1")
                print ("len(sigma_gas) =  ", length_sigma)
        else:
            sgas = sigma_gas #If there is only a single surface density measurement then we will use that

        if sigma_star_is_array == True:
            if length_sigma_star == len(shearArray):
                sstar = sigma_star[i]
            elif length_sigma_star == len(shearArray) +1:
                sstar = (sigma_star[i] + sigma_star[i+1]) / 2.
            else:
                print ("Error encountered in freeFallTimePerBin: Stellar surface density array (sigma_star) is wrong length. length must equal len(shearArray) or len(shearArray)+1")
                print ("len(sigma_gas) =  ", length_sigma_star)
        else:
            sstar = sigma_star     
        if veldispstar_is_array == True:
            if length_veldisp_star == len(shearArray):
                vd_star = veldisp_star[i]
            elif length_veldisp_star == len(shearArray)+1:
                vd_star = (veldisp_star[i] + veldisp_star[i+1]) / 2.
            else:
                print ("Error encountered in freeFallTimePerBin: Stellar velocity dispersion array (veldisp_star) is wrong length. length must equal len(shearArray) or len(shearArray)+1")
                print ("len(sigma_gas) =  ", length_veldisp_star)
        else:
            vd_star = veldisp_star
        
        if veldispgas_is_array == True:
            if length_veldisp_gas == len(shearArray):
                vd_gas = veldisp_gas[i]
            elif length_veldisp_gas == len(shearArray)+1:
                vd_gas = (veldisp_gas[i] + veldisp_gas[i+1]) / 2.
            else:
                print ("Error encountered in freeFallTimePerBin: Gas velocity dispersion array (veldisp_gas) is wrong length. length must equal len(shearArray) or len(shearArray)+1")
                print ("len(sigma_gas) =  ", length_veldisp_gas)
        else:
            vd_gas = veldisp_gas
        #Calculate the free-fall timescale:
        tQ = ToomreQ(vd_gas,freqep,sgas)
        tff_array.append(freeFallTime(tQ,Omega,shearArray[i],sstar,sgas,vd_star,vd_gas))
    return tff_array

def cloudcloudCollisionTimePerBin(veldisp_gas,sigma_gas,R_array,V_array,fg):
    #Need to allow sigma gas to be input as an array of length R_array or length shearArray (R_array-1)
    try:
        length_sigma = len(sigma_gas) #Try to measure the length of sigma gas
        sigma_is_array = True
    except:
        sigma_is_array = False #If it doesn't have a length then it is probably (hopefully) a float
    shearArray = shearParameter(R_array,V_array)
    try:
        length_veldisp_gas = len(veldisp_gas)
        veldispgas_is_array = True
    except:
        veldispgas_is_array = False
    tcc_array = []
    for i in range(len(shearArray)):
        R = (R_array[i]+R_array[i+1])/2.
        V = (V_array[i]+V_array[i+1])/2.
        Omega = getOmega(V,R)
        freqep = EpFreq([R_array[i],R_array[i+1]],[V_array[i],V_array[i+1]])
        if sigma_is_array == True: #If there is a surface density array then we need to check if, and how, we can use it
            if length_sigma == len(shearArray):
                sgas = sigma_gas[i] #If it is the same length as shear array, then we assume that it is input in the correct format and we can just use the corresponding value
            elif length_sigma == len(shearArray)+1:
                sgas = (sigma_gas[i]+sigma_gas[i+1]) / 2. #if it has a length of that of the shear array then we assume that it is in the same format as R_array and V_array. If so we take an average of the values surrounding the one that we actually want.
            else:
                print ("Error encountered in cloudcloudCollisionTimePerBin: Gas surface density array (sigma_gas) is wrong length. length must equal len(shearArray) or len(shearArray)+1")
                print ("len(sigma_gas) =  ", length_sigma)
        else:
            sgas = sigma_gas #If there is only a single surface density measurement then we will use that
        if veldispgas_is_array == True:
            if length_veldisp_gas == len(shearArray):
                vd_gas = veldisp_gas[i]
            elif length_veldisp_gas == len(shearArray)+1:
                vd_gas = (veldisp_gas[i] + veldisp_gas[i+1]) / 2.
            else:
                print ("Error encountered in freeFallTimePerBin: Gas velocity dispersion array (veldisp_gas) is wrong length. length must equal len(shearArray) or len(shearArray)+1")
                print ("len(sigma_gas) =  ", length_veldisp_gas)
        else:
            vd_gas = veldisp_gas
        tQ = ToomreQ(vd_gas,freqep,sgas) #calculate Toomre Q
        tcc_array.append(cloudCloudCollisionsTime(tQ,fg,Omega,shearArray[i])) #append the cloud-cloud collision timescale at the radius that we are interested in to the output array
    return tcc_array #return an array of cloud-cloud collision timescales

def epicyclicPerturbationTimescalePerBin(R_array,V_array):
    tep_array = []
    shearArray = shearParameter(R_array,V_array)
    for i in range(len(shearArray)):
        R = (R_array[i]+R_array[i+1])/2.
        V = (V_array[i]+V_array[i+1])/2.
        Omega = getOmega(V,R)
        tep_array.append(epicyclicPerturbationTimescale(Omega,shearArray[i]))
    return tep_array

def shearTimescalePerBin(R_array,V_array):
    tB_array = []
    shearArray = shearParameter(R_array,V_array)
    for i in range(len(shearArray)):
        R = (R_array[i]+R_array[i+1])/2.
        V = (V_array[i]+V_array[i+1])/2.
        Omega = getOmega(V,R)
        tB_array.append(shearTimescale(shearArray[i],Omega))
    return tB_array

def spiralArmsTimescalePerBin(R_array, V_array, Omega_P, m):
    #spiralArmTimescale(Omega,Omega_P,m)
    tsa = []
    try:
        length_Omega_P = len(Omega_P)
        Omega_P_is_array = True
    except:
        Omega_P_is_array = False
    for i in range(len(R_array)-1): #need to output something that will match the arrays produced by calculating the shear
        R = (R_array[i]+R_array[i+1])/2.
        V = (V_array[i]+V_array[i+1])/2.
        Omega = getOmega(V,R)
        if Omega_P_is_array == True:
            if length_Omega_P == len(R_array):
                O_p = (Omega[i] + Omega_P[i+1]) / 2.
            elif length_Omega_P == len(R_array) - 1:
                O_p = Omega_P[i]
            else:
                print ("Omega_P (pattern speed) must either be a floating point value or an array with length len(shearArray) or len(shearArray)+1!")
        else:
            O_p = Omega_P
        tsa.append(spiralArmTimescale(Omega, O_p, m))
    return tsa

def combineTimescalesPerBin(tff_array,tcc_array,tep_array,tsa_array,tB_array):
    t_array = []
    for i in range(len(tff_array)):
        t_array.append(combineTimescales([tff_array[i],tcc_array[i],tep_array[i],tsa_array[i],tB_array[i]]))
    return t_array
    
def CalculateRbinsForShear(R_array):
    newR_array = []
    for i in range(len(R_array)-1):
        newR_array.append((R_array[i]+R_array[i+1])/2.)
    return newR_array

def GetRadialProfileFromImage(Image,binEdgearray,pixelScale=50.,distance=50000.,setCentre=True,Normalised=True,centre=[0,0]):
        imshape = np.shape(Image)
        r_output_array = []
        density_array = []
        pixelTopc = distance * np.radians(pixelScale / 3600.)
        sum_array = np.zeros(len(binEdgearray)-1)
        if setCentre == True:
            centre = [(imshape[0] / 2.),(imshape[1] / 2.)]
        for i in range(imshape[0]):
            for j in range(imshape[1]):
                r = pixelTopc*np.sqrt((i-centre[0])**2.+(j-centre[1])**2.)
                for k in range(len(binEdgearray)-1):
                    if (r >= binEdgearray[k]) and (r < binEdgearray[k+1]):
                        sum_array[k] = sum_array[k] + Image[i,j]
                        break
        for i in range(len(sum_array)):
            density_array.append(sum_array[i] / ((2*np.pi*binEdgearray[i+1]**2.)-(2*np.pi*binEdgearray[i]**2.)))
            r_output_array.append((binEdgearray[i]+binEdgearray[i+1])/2.)
        if Normalised == True:
            density_array = np.array(density_array / (np.mean(density_array)))
        return r_output_array, density_array

def patternSpeedOverOmega(pspeed,V_array):
    p_array = []
    for i in range(len(V_array)-1):
        V = (V_array[i]+V_array[i+1])/2.
        p_array.append(pspeed / V)
    return p_array

def printHelp():
    print("TheoreticalCloudTimescales v0.4")
    print("Note that there is no main function. The functions must be called from another script.")
    print("For usage example see predict_timescales_example.py")
    print("######################################################################")
    print("######################################################################")
    print("######################## Basic functions: ############################")
    print("######################################################################")
    print("######################################################################")
    print("theoreticalCloudTimescales.SetupBins(R_array,V_array,binwidth=1000.0, maximum=4000., minimum=0.,include_edges=False)")
    print("Splits the radius and velocity arrays into bins of width binwidth between the maximum and minimum values.")
    print("Returns two arrays from the rotation curve data file: radius_array, velocity_array")
    print("######################################################################")
    print("theoreticalCloudTimescales.calculateGalaxyWideTimescale(cloudcloudcollisionProbablity, v_rebinned, r_rebinned, stellarSurfaceDensity, gasSurfaceDensity, veldispStar, veldispGas, Omega_p, n_spiralarms)")
    print("Calculates the galaxy-wide average timescales using the cloudcloudCollision probability (fudicial value=0.5), the rebinned arrays from SetupBins, the mean stellar and gas surface densities, the meann stellar and gas velocity dispersions, the pattern speed, and the number of spiral arms")
    print("returns a single floating point timescale")
    print("######################################################################")
    print("######################################################################")
    print("########### Galactic average individual timescales: ##################")
    print("######################################################################")
    print("######################################################################")
    print("theoreticalCloudTimescales.freeFallTime(toomreQ,Omega,avgBeta,surface_density_star,surface_density_gas,veldisp_star,veldisp_gas)")
    print("theoreticalCloudTimescales.cloudCloudCollisionsTime(tQ,fg,Omega,avgBeta)")
    print("theoreticalCloudTimescales.epicyclicPerturbationTimescale(Omega,avgBeta)")
    print("theoreticalCloudTimescales.shearTimescale(avgBeta,Omega)")
    print("theoreticalCloudTimescales.spiralArmTimescale(Omega,Omega_p,m)")
    print("theoreticalCloudTimescales.combineTimescales([t0,t1,t2,t3,...,tn],printTimes=False)")
    print("######################################################################")
    print("######################################################################")
    print("################# Timescales per radial bin: #########################")
    print("######################################################################")
    print("######################################################################")
    print("theoreticalCloudTimescales.freeFallTimePerBin(v_rebinned,r_rebinned,stellar_surface_density_profile,gas_surface_densities,veldispStar,veldispGas)")
    print("theoreticalCloudTimescales.cloudcloudCollisionTimePerBin(veldispGas,gas_surface_densities,r_rebinned,v_rebinned,prob_cc)")
    print("theoreticalCloudTimescales.epicyclicPerturbationTimescalePerBin(r_rebinned,v_rebinned)")
    print("theoreticalCloudTimescales.shearTimescalePerBin(r_rebinned,v_rebinned)")
    print("theoreticalCloudTimescales.spiralArmsTimescalePerBin(r_rebinned,v_rebinned,Omega_p,n_spiralarms)")
    print("theoreticalCloudtimescales.combineTimescalesPerBin(tff_array,tcc_array,tep_array,tspa_array,tB_array)")
    print("Note that unlike combineTimescales, combineTimescalesPerBin requires arrays for all 5 timescales")
    print("######################################################################")
    print("######################################################################")
    print("###################### Utility functions: ############################")
    print("######################################################################")
    print("######################################################################")
    print("theoreticalCloudTimescales.G()")
    print("Returns gravitational constant in units of pc / M_o (km/s)^2")
    print("######################################################################")
    print("theoreticalCloudTimescales.GetRadialProfileFromImage(image,binEdges,pixelScale=50.,distance=50000.,setCentre=True,Normalised=True)")
    print("Creates a radial profile from an image (e.g. to obtain a density profile). image = input fits image,  binEdges = radius_array from SetupBins, pixel scale must be set to pixel scale of the input image in arcseconds, distance is distance to galaxy in pc")
    print("returns two arrays: radius_array ,surface_densities")
    print("######################################################################")
    print("theoreticalCloudTimescales.shearParameter(R_array, V_array):")
    print("returns an array of the shear parameter with dimensions len(R_array)-1")
    print("######################################################################")
    print("theoreticalCloudtimescales.ToomreQ(veldisp, ep_frequency, surface_density):")
    print("returns ToomreQ for a given velocity dispersion, epicyclic frequency and surface density")
    print("######################################################################")
    print("theoreticalCloudTimescales.getOmega(vel, radius):")
    print("returns Omega for a given velocity and radius")
    print("######################################################################")
    print("theoreticalCloudTimescales.EpFreq(R_array,V_array):")
    print("returns epicyclic frequency")
    print("######################################################################")
    print("theoreticalCloudTimescales.readInData(input_file):")
    print("limited usage for a very specific rotation curve data file format")
    print("Reads in data in the form R,V,Verr with delimiter = ',' and three lines of header")
    print("returns three arrays: radius, velocity, velocity_error")
    print("######################################################################")
    print("theoreticalCloudTimescales.CalculateRbinsForShear(R_array):")
    print("calculates new radius array with dimensions (len(R_array)-1) so that shear parameter can be plotted")
    print("######################################################################")
    print("theoreticalCloudTimescales.printHelp()")
    print("Displays this information in terminal")
    print("######################################################################")


if __name__ == "__main__":
    printHelp()

