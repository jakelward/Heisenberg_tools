import theoreticalCloudTimescales as tt
import matplotlib.pyplot as plt
import pyfits
import numpy as np
#This is an example script for making use of the functions contained in theoreticalCloudTimescales

#Read in rotation curve data:
inputRotationCurve = 'LMC_stellar_RC.dat'
R_array, V_array, Verr_array = tt.readInData(inputRotationCurve)
R_array = R_array *1000. #Convert radial array into pc from kpc
#Rebin the data:
#When calculating a single galactic average, set include_edges = False, when plotting timescales as a function of radius, set to True. This is necessary to avoid double counting in the average case and to get proper averages for bins in the function of radius case
r_rebinned, v_rebinned = tt.SetupBins(R_array,V_array,binwidth=1000.0, maximum=4000., minimum=0.,include_edges=False) #If binwidth and maximum are ommited, then it reverts to defaults of binwidth = 1000. and maximum = 4000 pc

#Inputs for constant surface density and constant velocity dispersions
gasSurfaceDensity = 12.8
stellarSurfaceDensity = 78.9 #Total stellar mass = 2.7*10**9 (within 8.9 kpc) --> density = 10.85. However, the vast majority of this is within the 3.3 kpc that we look at here
#If we place all 2.7e9 Mo within 3.3 kpc we get a stellar density of 78.9 Mo /pc**2.
veldispStar = 21.
veldispGas = 15.8
#Fudicial value for probability of cloud-cloud collisions (prob_cc) = 0.5
prob_cc = 0.5

#Calculate the average timescale for cloud-cloud collisions across a galaxy with a single spiral arm:
tavg = tt.calculateGalaxyWideTimescale(prob_cc, v_rebinned, r_rebinned, stellarSurfaceDensity, gasSurfaceDensity, veldispStar, veldispGas, 40./1000., 1)
print ("mean timescale over galaxy: ",tavg)
r_rebinned, v_rebinned = tt.SetupBins(R_array,V_array,binwidth=500.0, maximum=4000., minimum=0., include_edges=True) 


#Now to do the same but we will calculate gas surface density as a function of radius from an image:
binEdges = r_rebinned
input_image_file = '/Volumes/Seagate_1/LMC_commonscale/lmc_hi_Karl_fcalJy_rmpSHASSA.fits'
image = pyfits.getdata(input_image_file)
r_surfdense,gas_surface_densities = tt.GetRadialProfileFromImage(image,binEdges,pixelScale=50.,distance=50000.,setCentre=True,Normalised=True) #pixel scale = 50 arcseconds and distance = 50 kpc
#Multiply the normalised surface density profile by the mean surface density:
gas_surface_densities = gas_surface_densities * gasSurfaceDensity

#Calculate the free-fall timescale for len(v_rebinned) -1 bins with the new surface density profile:
tff_array = tt.freeFallTimePerBin(v_rebinned,r_rebinned,stellarSurfaceDensity,gas_surface_densities,veldispStar,veldispGas)

#The same can be done to get and use radial profiles for stellar surface density, stellar velocity dispersion, and gas velocity dispersion as a function of radius.
image2 = pyfits.getdata('/Volumes/Seagate_1/LMC_commonscale/lmc.ha.fcalJy_rmpSHASSA.fits')
r_surfdense, stellar_surface_density_profile = tt.GetRadialProfileFromImage(image2,binEdges,pixelScale=50.,distance=50000.,setCentre=True,Normalised=True)
stellar_surface_density_profile = stellar_surface_density_profile * stellarSurfaceDensity

#We can then calculate a new freefall timescale array using surface density profiles of both stars and gas:
tff_array = tt.freeFallTimePerBin(v_rebinned,r_rebinned,stellar_surface_density_profile,gas_surface_densities,veldispStar,veldispGas)
#We then can then calculate the other timescales per bin:
tcc_array = tt.couclcloudCollisionTimePerBin(veldispGas,gas_surface_densities,r_rebinned,v_rebinned,prob_cc)
tep_array = tt.epicyclicPerturbationTimescalePerBin(r_rebinned,v_rebinned)
tB_array = tt.shearTimescalePerBin(r_rebinned,v_rebinned)
tspa_array = tt.spiralArmsTimescalePerBin(r_rebinned,v_rebinned,(40./1000),1) #40 / 1000 comes from pattern speed of 40km/s / kpc (Dottori1996). Also possible is 21 +/- 3 km/s/kpc from Shimizu 2012
#Finally we combine the timescales per bin to get a cloud timescale profile for the galaxy:
t_array = tt.combineTimescalesPerBin(tff_array,tcc_array,tep_array,tspa_array,tB_array)
print(t_array)
#Then if we want to make a plot, we need to multiply the shear timescale by minus 1 to convert it to a positive timescale
tB_array = -1.0 * np.array(tB_array)
#We also need to get the bins for which the timescales are actually calculated. This has a dimensionality of len(r_rebinned)-1:
r_array = tt.CalculateRbinsForShear(r_rebinned)
#Now we can plot the results:
plt.plot(r_array,t_array,label='t',alpha=0.8)
plt.plot(r_array,tff_array,ls='--',label=r'$t_{ff}$',alpha=0.8)
plt.plot(r_array,tcc_array,ls='--',label=r'$t_{cc}$',alpha=0.8)
plt.plot(r_array,tep_array,ls='--',label=r'$t_{ep}$',alpha=0.8)
plt.plot(r_array,tspa_array,ls='--',label=r'$t_{sp}$',alpha=0.8)
plt.plot(r_array,tB_array,ls='--',label=r'$t_{\beta}$',alpha=0.8)
plt.legend()
plt.xlabel("R [pc]")
plt.ylabel("t [Myr]")
plt.yscale("log")
plt.show()