import theoreticalCloudTimescales as tt
import matplotlib.pyplot as plt
import pyfits

#This is a script demonstrating the use of the theoreticalCloudTimescales as a module
#In this case the script will provide an analysis of the LMC

#Read in rotation curve data:
inputRotationCurve = 'LMC_stellar_RC.dat'
R_array, V_array, Verr_array = tt.readInData(inputRotationCurve)
R_array = R_array *1000. #Convert radial array into pc from kpc
#Rebin the data:
r_rebinned, v_rebinned = tt.SetupBins(R_array,V_array,binwidth=500.0, maximum=4000., minimum=0.) #If binwidth and maximum are ommited, then it reverts to defaults of binwidth = 1000. and maximum = 4000 pc

#Inputs for constant surface density and constant velocity dispersions
gasSurfaceDensity = 12.8
stellarSurfaceDensity = 78.9 #Total stellar mass = 2.7*10**9 (within 8.9 kpc) --> density = 10.85. However, the vast majority of this is within the 3.3 kpc that we look at here
#If we place all 2.7e9 Mo within 3.3 kpc we get a stellar density of 78.9 Mo /pc**2.
veldispStar = 21.
veldispGas = 15.8
#Fudicial value for probability of cloud-cloud collisions (prob_cc) = 0.5
prob_cc = 0.5

#Calculate the average timescale for cloud-cloud collisions across a galaxy with no spiral arms:
tavg = tt.calculateGalaxyWideTimescale(prob_cc, v_rebinned, r_rebinned, stellarSurfaceDensity, gasSurfaceDensity, veldispStar, veldispGas)
print("mean tiemscale over galaxy: ",tavg)

#Calculate the free-fall timescale for len(v_rebinned) -1 bins
tff_array = tt.freeFallTimePerBin(v_rebinned,r_rebinned,stellarSurfaceDensity,gasSurfaceDensity,veldispStar,veldispGas)
print(tff_array)

#Calculate the cloud-cloud collision timescale across the same rane of bins:
tcc_array = tt.couclcloudCollisionTimePerBin(veldispGas,gasSurfaceDensity,r_rebinned,v_rebinned,prob_cc)
print(tcc_array)

#And for epicyclic perturbations
tep_array = tt.epicyclicPerturbationTimescalePerBin(r_rebinned,v_rebinned)
print(tep_array)

#spiral arms timescale (in this case we have no spiral arms):
#spiralArmTimescale(Omega,Omega_P,m) 
# #m=no. spiral arms, if set to 0  then timescale is set to 1000000 Myr. This is just a number longer than a Hubble time because the spiral arms will never destroy clouds if they do not exist
#Omega_P is the pattern speed
Omega_P = 0.05
m = 0.
tspa_array = tt.spiralArmsTimescalePerBin(r_rebinned,v_rebinned,Omega_P,m)

#Shear:
tB_array = tt.shearTimescalePerBin(r_rebinned,v_rebinned)
print(tB_array)

tspa_array = tt.spiralArmsTimescalePerBin(r_rebinned,v_rebinned,0.05,1)

#Combine these timescales:
t_array = tt.combineTimescalesPerBin(tff_array,tcc_array,tep_array,tspa_array,tB_array)
print(t_array)

#Convert shear timescale into something positive for plotting:
for i in range(len(tB_array)):
    tB_array[i] = -1.0 * tB_array[i]

#rebin radii to match the bins that the timescales are calculated over:
r_array = tt.CalculateRbinsForShear(r_rebinned)
plot the various timescales as a function of radius:
plt.plot(r_array,t_array)
plt.plot(r_array,tff_array)
plt.plot(r_array,tcc_array)
plt.plot(r_array,tep_array)
plt.plot(r_array,(tB_array))
plt.show()

#Now to do the same but we will calculate gas surface density as a function of radius from an image:
binEdges = r_rebinned
input_image_file = 'lmc_hi_Karl_fcalJy_rmpSHASSA.fits' #
image = pyfits.getdata(input_image_file)
r_surfdense,gas_surface_densities = tt.GetRadialProfileFromImage(image,binEdges,pixelScale=50.,distance=50000.,setCentre=True,Normalised=True) #pixel scale = 50 arcseconds and distance = 50 kpc
#Multiply the normalised surface density profile by the mean surface density:
gas_surface_densities = gas_surface_densities * gasSurfaceDensity

#Calculate the free-fall timescale for len(v_rebinned) -1 bins with the new surface density profile:
tff_array = tt.freeFallTimePerBin(v_rebinned,r_rebinned,stellarSurfaceDensity,gas_surface_densities,veldispStar,veldispGas)

#The same can be done to get and use radial profiles for stellar surface density, stellar velocity dispersion, and gas velocity dispersion as a function of radius.
image2 = pyfits.getdata('lmc.ha.fcalJy_rmpSHASSA.fits')
r_surfdense, stellar_surface_density_profile = tt.GetRadialProfileFromImage(image2,binEdges,pixelScale=50.,distance=50000.,setCentre=True,Normalised=True)
stellar_surface_density_profile = stellar_surface_density_profile * stellarSurfaceDensity
plt.close
plt.plot(r_surfdense,gas_surface_densities)
plt.plot(r_surfdense,stellar_surface_density_profile)
plt.show()

tff_array = tt.freeFallTimePerBin(v_rebinned,r_rebinned,stellar_surface_density_profile,gas_surface_densities,veldispStar,veldispGas)
tcc_array = tt.couclcloudCollisionTimePerBin(veldispGas,gas_surface_densities,r_rebinned,v_rebinned,prob_cc)
tep_array = tt.epicyclicPerturbationTimescalePerBin(r_rebinned,v_rebinned)
tB_array = tt.shearTimescalePerBin(r_rebinned,v_rebinned)
tspa_array = tt.spiralArmsTimescalePerBin(r_rebinned,v_rebinned,(50./1000),1) #50 / 1000 comes from pattern speed of 40km/s / kpc

t_array = tt.combineTimescalesPerBin(tff_array,tcc_array,tep_array,tspa_array,tB_array)
print(t_array)
for i in range(len(tB_array)):
    tB_array[i] = -1.0 * tB_array[i]
# plt.close()
plt.plot(r_array,t_array,label='t')
plt.plot(r_array,tff_array,ls='--',label='tff')
plt.plot(r_array,tcc_array,ls='--',label='tcc')
plt.plot(r_array,tep_array,ls='--',label='tep')
plt.plot(r_array,tspa_array,ls='--',label='tp')
plt.plot(r_array,tB_array,ls='--',label='tB')
plt.legend()
plt.xlabel("R [pc]")
plt.ylabel("t [Myr]")
plt.yscale("log")
plt.show()