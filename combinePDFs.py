import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.stats as ss
import numpy.polynomial.polynomial as poly
from os import listdir
from scipy.optimize import curve_fit
#######################################################################################################################
#combinePDFs v.0.3: A script for combining probability distribution functions associated with multiple measurements
#of the same quantity using different methods. This is largely based on the work of Lyons et al. 1988 and details
#on how the process works can be found in Appendix E of Hygate et al. 2019.
#######################################################################################################################
#Currently this script assumes that whichever input PDF file comes first in the list spans an appropriate 
#range to contain everything of interest. Arguably this should not be an issue unless something has gone
#wrong at an earlier stage but I will hopefully fix this at some point.
#######################################################################################################################
###This is the only section that should be changed in normal use###
###################################################################
input_dir = '/some/directory/with/input/PDF.dat/files'# <-- Put all of the input PDFs into a common directory with no other files present
output_file = '/path/to/the/output/file.dat'# <-- The path to your output file goes here
plot = 1#[0,defaut] do not plot the output, [1] plot the output PDF. Also note that this is not actually implemented so does absolutely nothing
plot_file = 'a_properly_combined_test.eps' #The name of the output plot if plot == 1
EstimateUncertaintiesFromPDFs = True#If set to true uncertainties are estimated from the width of the input PDFs, otherwise the uncertainty lists specified below are used
uncertaintiesHigh = [1.7,2.6,2.0] #These values are used if EstimateUncertaintiesFromPDFs == False
uncertaintiesLow = [1.8,2.5,2.0] #These values are used if EstimateUncertaintiesFromPDFs == False
ForceEqualWeighting = False#If set to True, all PDFs are forced to have equal weighting and therefore the average PDF is returned
ForceCustomWeighting = False #If set to True, use the weighting below
CusomWeighting = [0.,0.5,0.5] #These weights are used if ForceCustomWeighting is set to True
mode = 'gradient_avg' #Don't worry about this. This is just the method for calculating the correlation coefficients. gradient_avg is the best so far but I may update this with more interesting methods in the future. Some modes do not exist, others may cause crashes or weird stuff.
Nstep = 1000000 #Number of steps used to calculate the best weights.
nComments = 7 #Number of comment lines in the input PDF files (must be common across all files!)
overlap = 10 #Don't worry about this for now
###################################################################
###End of User Input###############################################
###################################################################
def findleftright(x,a):
    left = np.where(x[:-1]>a)[0][-1]
    right = 1 + np.where(x[1:]<a)[0][0]
    diff= (x[right] - x[left])
    fracfromleft = (a-x[left]) / diff
    return left, right, fracfromleft
def findrightleft(x,a):
    x = np.flip(x)
    left = np.where(x[:-1]<a)[0][-1]
    right = 1 + np.where(x[1:]>a)[0][0]
    diff= (x[right] - x[left])
    fracfromleft = (a-x[left]) / diff
    return right, left, fracfromleft

def InterpOverGaps(out_array, count_array, newdims, Interp1s):
    for i in range(int(newdims[1])):
        if count_array[i] == 0:
            if Interp1s == False:
                if i != 0 and i != int(newdims[1])-1:
                    if count_array[i-1] != 0 and count_array[i+1] != 0:
                        out_array[1,i] = (out_array[1,i-1]+out_array[1,i+1]) / 2.
            else:
                if i != 0 and i != int(newdims[1])-1:
                    if count_array[i-1] != 0 and count_array[i+1] != 0 and count_array[i-1] != 1 and count_array[i+1] != 1:
                        out_array[1,i] = (out_array[1,i-1]+out_array[1,i+1]) / 2.
                    else:
                        if i > 1 and i < int(newdims[1])-2:
                            out_array[1,i] = (out_array[1,i-2]+out_array[1,i+2]) / 2.
        if Interp1s == True:
            if count_array[i] == 1:
                if Interp1s == False:
                    if i != 0 and i != int(newdims[1])-1:
                        if count_array[i-1] != 0 and count_array[i+1] != 0:
                            out_array[1,i] = (out_array[1,i-1]+out_array[1,i+1]) / 2.
                else:
                    if i != 0 and i != int(newdims[1])-1:
                        if count_array[i-1] != 0 and count_array[i+1] != 0 and count_array[i-1] != 1 and count_array[i+1] != 1:
                            out_array[1,i] = (out_array[1,i-1]+out_array[1,i+1]) / 2.
                        else:
                            if i > 1 and i < int(newdims[1])-2:
                                out_array[1,i] = (out_array[1,i-2]+out_array[1,i+2]) / 2.
    return out_array

#This just makes my life a bit simpler for now
def setupNewMethod(a,newdims):
    P1 = a[0,:]
    minP1 = np.min(P1)
    maxP1 = np.max(P1)
    P2 = a[1,:]
    stepsize = (maxP1-minP1)/(newdims[1]+3)
    out_array = np.zeros((int(newdims[0]),int(newdims[1])))
    binmin = minP1
    binmax = minP1 + stepsize
    binEdgeArray = np.zeros((int(newdims[1])+1))
    count_array = np.zeros(int(newdims[1]))
    for j in range(len(binEdgeArray)):
        binEdgeArray[j] = binmin + (stepsize*j)
    return P1, P2, out_array,binEdgeArray,count_array,binmin,binmax,stepsize


#Rebin PDF1 to the axis of PDF2
def rebinPDF1toAxis2(PDF1, PDF2,method ='average',InterpGaps=True,Interp1s=False):
    inputdim = np.shape(PDF1)
    targetdim = np.shape(PDF2)
    out_array = np.zeros(targetdim)
    minB = np.min(PDF2[:,0])
    maxB = np.max(PDF2[:,0])
    stepsize = (maxB - minB) / targetdim[0]
    binEdgeArray = np.zeros(targetdim[0]+1)
    binEdgeArray[0] = minB
    binEdgeArray[len(binEdgeArray)-1] = maxB
    for i in range(len(binEdgeArray)):
        if (i < len(binEdgeArray)-2):
            binEdgeArray[i+1] = PDF2[i,0]+(PDF2[i+1,0]-PDF2[i,0])/2.
    if method in ['average']:
        count_array = np.zeros(targetdim[0])
        for i in range(targetdim[0]):
            out_array[i,0] = PDF2[i,0]
        for i in range(targetdim[0]):
            for j in range(inputdim[0]):
                if (PDF1[j,0] >= binEdgeArray[i]) and (PDF1[j,0] <= binEdgeArray[i+1]):
                    out_array[i,1] = out_array[i,1] + PDF1[j,1]
                    out_array[i,2] = out_array[i,2] + PDF1[j,2]
                    count_array[i] = count_array[i] + 1
            out_array[i,1] = out_array[i,1] / count_array[i]
            out_array[i,2] = out_array[i,2] / count_array[i]
        if InterpGaps ==True:
            out_array = InterpOverGaps(out_array, count_array, newdims, Interp1s)
        return out_array
    elif method in ['quadratureAdd']:
        print ('quadratureAdd not yet implemented')
        return None
    else:
        print ('rebin error: method not recognised')
        return None
#Puts PDFs in same axis and multiplys them
def multiplyPDFs(PDF1, PDF2,method ='average',InterpGaps=True,Interp1s=True):
    PDF1_ax2 = rebinPDF1toAxis2(inputPDF1, inputPDF2, method=method, InterpGaps=InterpGaps, Interp1s=Interp1s)
    PDF1_Y = PDF1_ax2[1,:]
    PDF2_Y = PDF2[1,:]
    outY = PDF1_Y * PDF2_Y
    outY = np.nan_to_num(outY)
    outY = outY / np.sum(outY)
    outArray = np.array([PDF1_ax2[0,:],outY])
    return outArray
def getFWHM(FWHM_array):
    FWHM_array = np.nan_to_num(FWHM_array)
    half_maxmult = np.max(FWHM_array[:,1]) / 2.
    go = 0
    i = 0
    while (go < 1):
        if FWHM_array[i,1] == half_maxmult:
            halfmax_lo = FWHM_array[i,0]
            go = 1
        elif FWHM_array[1,i] > half_maxmult:
            halfmax_lo = (FWHM_array[i-1,0] + FWHM_array[i,0]) / 2.
            go = 1
        else:
            i = i+1
    i = np.argmax(FWHM_array[:,1])
    go = 0
    while (go < 1):
        if FWHM_array[i,1] == half_maxmult:
            halfmax_hi = FWHM_array[i,0]
            go = 1
        elif FWHM_array[i,1] < half_maxmult:
            halfmax_hi = (FWHM_array[i-1,0] + FWHM_array[i,0]) / 2.
            go = 1
        else:
            i = i+1

    return (halfmax_hi-halfmax_lo), halfmax_lo, halfmax_hi
def calcCorrelationCoef(PDF1,PDF2,mode='simple'):
    if mode in ['simple']:
        tau, p_value = ss.kendalltau(PDF1[1,:],PDF2[1,:])
        wtaur, wtp = ss.weightedtau(PDF2[1,:],PDF1[1,:],rank=None)
        pr, prp = ss.pearsonr(PDF1[1,:],PDF2[1,:])
        print (tau)
        print (wtaur)
        print (pr)
        return wtaur
    elif mode in ['complex','complex1']: #In complex mode, we will define the correlation coefficient as the degree to which the observations conform to the expectation
        rcorr = np.zeros(len(PDF1[1,:]))
        def expectation(x):
            return x #i.e. as we are measuring the same quantity, we expect that the PDFs should follow a 1:1 relationship in the absence of bias or poor sampling
        for i in range(len(PDF1[1,:])):
            if PDF2[1,i] < expectation(PDF1[1,i]):
                rcorr[i] = PDF2[1,i] / expectation(PDF1[1,i])
            else:
                rcorr[i] =  expectation(PDF1[1,i]) / PDF2[1,i]
        print (rcorr)
        return rcorr
    elif mode in ['gradient','gradient_avg']:
        rcorr = np.zeros(len(PDF1[1,:]))
        stretchlength = 2
        expectedGradient = 1.0
        for i in range(len(PDF1[1,:])):
            if (i > stretchlength) and (i < len(PDF1[1,:]) - stretchlength):
                measgrad = (PDF2[1,i+stretchlength] - PDF2[1,i-stretchlength]) / (PDF1[1,i+stretchlength]-PDF1[1,i-stretchlength])
            elif (i <= stretchlength):
                measgrad = (PDF2[1,i+1] - PDF2[1,i]) / (PDF1[1,i+1]-PDF1[1,i])
            elif (i >=(len(PDF1[1,:]) - stretchlength)):
                measgrad = (PDF2[1,i] - PDF2[1,i-1]) / (PDF1[1,i]-PDF1[1,i-1])
            else:
                print ("something unexpected has occured")
            if measgrad > expectedGradient:
                rcorr[i] = expectedGradient / measgrad
            else:
                rcorr[i] = measgrad / expectedGradient
        if mode in ['gradient']:
            return rcorr
        elif mode in ['gradient_avg']:
            return np.median(np.nan_to_num(rcorr))
        else:
            print ('something has gone horribly wrong')
            return None
    else:
        print ('mode does not exist')
def getybest(alpha_array,y_array):
    y = 0.
    for i in range(len(alpha_array)):
        y = y+(alpha_array[i]*y_array[i])
    return y

def getybestalog(alpha_array,y_array):
    y = 0.
    for i in range(len(alpha_array)):
        y = y+(alpha_array[i]*(10**y_array[i]))
    y = np.log10(y)
    return y

def MinimiseSigma(E,Nvals,Ntry):
    minsig = 10000. #some Arbitrary high number
    print('calculating best weights...')
    for i in range(Ntry):
        alphatry = np.zeros(Nvals)
        for j in range(Nvals):
            alphatry = np.random.uniform(low=0.,high=1.0,size=len(alphatry))
        alphatry = 1.0 / np.sum(alphatry) * alphatry
        sigma2 = CalcSigma2(E,Nvals,Nstep,alphatry)
        if sigma2 < minsig:
            minsig = sigma2
            best_alphas = alphatry
    return best_alphas, minsig

def CalcSigma2(E,Nvals,Nstep,alpha_array):
    sigma2 = 0.
    for i in range(Nvals):
        for j in range(Nvals):
            sigma2 = sigma2 + E[i,j]*alpha_array[i]*alpha_array[j]
    return sigma2

def dealWithANaN(i,ab,j,data_array_regrid):
    l = i
    k = i
    gohigh = True
    foundhi = False
    golow = True
    foundlo = False
    while gohigh:
        if l >= (len(data_array_regrid[:,ab,j])-1):
            gohigh = False
        else:
            l+=1
            hival = data_array_regrid[l,ab,j]
            if not np.isnan(hival):
                gohigh = False
                foundhi = True
    while golow:
        if k <=0:
            golow = False
        else:
            k -= 1
            loval = data_array_regrid[k,ab,j]
            if not np.isnan(loval):
                golow = False
                foundlo = True
    if foundlo and foundhi:
        diffl = l-i
        diffk = i-k
        yval = np.log10((10**hival + 10**loval) / 2.)
        if diffl == diffk:
            yval = (hival + loval) / 2.
        elif diffl > diffk:
            rat = float(diffk) / float(diffl)

            yval = (rat*highval + (2.-rat*loval)) / 2.
        else:
            rat = float(diffl) / float(diffk)
            yval = (rat*loval + (2.-rat*highval)) / 2.
    else:
        yval = np.nan
    return yval


def main():
    input_files_list = listdir(input_dir)
    NPDF = len(input_files_list)
    print (input_files_list[0])
    datatemp = np.genfromtxt((str(input_dir+input_files_list[0])),dtype=float,delimiter='       ',skip_header=nComments)
    datashape = np.shape(datatemp)
    datatemp = None
    data_array = np.empty((datashape[0],datashape[1],NPDF))
    for i in range(NPDF):
        fullpath = str(input_dir + input_files_list[i])
        print (input_files_list[i])
        data = np.genfromtxt(fullpath,dtype=float,delimiter='       ',skip_header=nComments)
        data_array[:,:,i] = data
    #Step 2: Regrid all PDFs to same axis
    PDFbase = np.array(data_array[:,:,0])
    data_array_regrid = np.copy(data_array)
    for i in range(NPDF):
        PDFinp = np.array(data_array[:,:,i])
        data_array_regrid[:,:,i] = rebinPDF1toAxis2(PDFinp,PDFbase,method ='average',InterpGaps=False,Interp1s=False)

    for i in range(NPDF):
        for j in [1,2]:
            for k in range(len(data_array_regrid[:,0,0])):
                if np.isnan(data_array_regrid[k,j,i]):
                    data_array_regrid[k,j,i] = dealWithANaN(k,j,i,data_array_regrid)
    #Step 3: Determine correlation coeficient matrix
    rmatrix = np.empty((NPDF,NPDF))
    for i in range(NPDF):
        for j in range(NPDF):
            PDF1 = np.array([data_array_regrid[:,0,i],(10**data_array_regrid[:,1,i]) * (10**data_array_regrid[:,2,i])])
            PDF2 = np.array([data_array_regrid[:,0,j],(10**data_array_regrid[:,1,j]) * (10**data_array_regrid[:,2,j])])
            rmatrix[i,j] = calcCorrelationCoef(PDF1,PDF2,mode=mode)

    #print rmatrix
    #Step 4: Contstruct error matrix
    #Step 4a: Estimate uncertainties in measurements
    inpErrorArray = np.zeros(NPDF)
    for i in range(NPDF):
        #Proper uncertainty estimation:
        if EstimateUncertaintiesFromPDFs:
            PDF1 = np.array([data_array_regrid[:,0,i],(10**data_array_regrid[:,1,i]) * (10**data_array_regrid[:,2,i])])
            modeValue = data_array_regrid[np.argmax(np.nan_to_num(PDF1[1,:])),0,0]
            sizea = np.shape(data_array_regrid[:,:,0])
            inpFWHM = np.zeros((sizea[0],2))
            for k in range(sizea[0]):
                for j in range(2):
                    if j == 0:
                        inpFWHM[k,j] = PDF1[0,k]
                    else:
                        inpFWHM[k,j] = PDF1[1,k]
            FWHM_mult, FWHM_lo,FWHM_hi = getFWHM(inpFWHM) #Need proper input
            ErrHI = 0.741*(FWHM_hi-modeValue)
            ErrLO = 0.741*(modeValue-FWHM_lo)
            print (10**modeValue, '+/-', 10**(ErrHI+modeValue)-10**modeValue,'/', 10**modeValue-10**(modeValue-ErrLO))
        else:
            ErrHI = np.log10(uncertaintiesHigh[i])
            ErrLO = np.log10(uncertaintiesLow[i])
            modeValue = data_array_regrid[np.argmax(np.nan_to_num(PDF1[1,:])),0,1]
        inpErrorArray[i] = np.sqrt(10**ErrHI+10**ErrLO) / 10**modeValue#geometric average of ErrHI and ErrLO divided by modeValue

    print ('realtive error array:')
    print (inpErrorArray)

    #Step 4b: Construct the matrix:
    E = np.zeros((NPDF,NPDF))
    for i in range(NPDF):
        for j in range(NPDF):
            if i == j:
                E[i,j] = inpErrorArray[i]**2.
            else:
                E[i,j] = rmatrix[i,j]*inpErrorArray[i]*inpErrorArray[j]
    print (E)
    #Step 5: Determine 'best' values for alpha array by minising sigma
    bestalphavalues, bestSigma = MinimiseSigma(E,NPDF,Nstep)
    if ForceEqualWeighting:
        for i in range(len(bestalphavalues)):
            bestalphavalues[i] = 1. / len(bestalphavalues)
    if ForceCustomWeighting:
        bestalphavalues = CusomWeighting
    print (bestalphavalues)
    print (bestSigma)

    #Step 6: Calculate best timescale PDF
    size = np.shape(data_array[:,:,0])
    print (size[0])
    outPDF = np.zeros(size[0])
    finalPDF = np.zeros((3,size[0]))
    for i in range(size[0]):
        y_array = np.zeros(len(bestalphavalues))
        ya_array = np.zeros(len(bestalphavalues))
        yb_array = np.zeros(len(bestalphavalues))
        for j in range(NPDF):
            if np.isnan(data_array_regrid[i,1,j]): #This if statement will deal with NaNs by offering a very simplistic interpolation
                ya_array[j] = dealWithANaN(i,1,j,data_array_regrid)
            else:
                ya_array[j] = data_array_regrid[i,1,j]
            if np.isnan(data_array_regrid[i,2,j]):
                yb_array[j] = dealWithANaN(i,2,j,data_array_regrid)
            else:
                yb_array[j] = data_array_regrid[i,2,j]
            y_array[j] = 10**data_array_regrid[i,1,j]*10**data_array_regrid[i,2,j]
        outPDF[i] = getybest(bestalphavalues,y_array)
        finalPDF[0,i] = data_array_regrid[i,0,0]
        finalPDF[1,i] = getybestalog(bestalphavalues,ya_array)
        finalPDF[2,i] = getybestalog(bestalphavalues,yb_array)


    #Step 7 obtain best timescale and uncertainties from PDF

    #Step 7a Fit PDF with 2-sided Gaussian:
    #First split the Gaussian in half:
    maxindex = np.argmax(np.nan_to_num(outPDF))
    PDFlow = np.nan_to_num(10**finalPDF[1,0:maxindex+overlap]) * np.nan_to_num(10**finalPDF[2,0:maxindex+overlap])
    PDFhigh = np.nan_to_num(10**finalPDF[1,maxindex-overlap:len(outPDF)-1]) * np.nan_to_num(10**finalPDF[2,maxindex-overlap:len(outPDF)-1])
    PDFlowX = finalPDF[0,0:maxindex+overlap]
    PDFhighX = finalPDF[0,maxindex-overlap:len(outPDF)-1]
    #Then fit a Gaussian to each half of the PDF:
    def gauss(x, *p):
        A, mu, sigma = p
        return A*np.exp(-(x-mu)**2/(2.*sigma**2))

    p0 = [0.03, np.max(np.nan_to_num(outPDF)), 1.]

    coefflow, var_matrixlow = curve_fit(gauss, PDFlowX,PDFlow, p0=p0)
    coeffhigh, var_matrixhigh = curve_fit(gauss, PDFhighX,PDFhigh, p0=p0)

    print ('t = ', 10**np.mean([coefflow[1],coeffhigh[1]]), ' +/- ', (10**(coeffhigh[1]+coeffhigh[2])-10**(coeffhigh[1])), '/', (10**(coefflow[1]-coefflow[2])-10**(coefflow[1])), ' Myr')

    # Get the fitted curve
    hist_fitlow = gauss(data_array_regrid[:,0,0], *coefflow)
    hist_fithigh = gauss(data_array_regrid[:,0,0], *coeffhigh)
    combined_fit = np.zeros(len(finalPDF[0,:]))
    for i in range(len(finalPDF[1,:])):
        if i < maxindex-overlap:
            combined_fit[i] = hist_fitlow[i]
        elif i > maxindex + overlap:
            combined_fit[i] = hist_fithigh[i]
        else:
            combined_fit[i] = np.mean([hist_fitlow[i],hist_fithigh[i]])

    np.savetxt(output_file,np.transpose(finalPDF))
    if plot == 1:
        for i in range(NPDF):
            plt.plot(data_array_regrid[:,0,i],(10**data_array_regrid[:,1,i]*10**data_array_regrid[:,2,i]),color='grey',ls=':')
        plt.plot(finalPDF[0,:],(10**finalPDF[1,:]*10**finalPDF[2,:]), color='black')
        plt.plot(data_array_regrid[:,0,0], combined_fit, label='Fitted data')
        plt.gca().set_xlabel(r'log($t_{CO}$)',fontsize=14)
        plt.gca().set_ylabel(r'$dp/dt_{CO}$',fontsize=14)
        plt.savefig(plot_file, format='eps', dpi=600)
        plt.show()

if __name__ == "__main__":
    main()
