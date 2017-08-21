"""
code to find the monthly average of the pressure level integral 

of meridional eddy momentum flux convergence between two given

pressure levels from daily data

i.e. integral{ d(u'v')/dy }dp

"""

import numpy as np
from netCDF4 import Dataset,date2num
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import os
from datetime import datetime, date, time


"""
to account for the fact that pressure levels are not all equally spaced
use a 'Simpson's rule' i.e. summing up trapeziums whereby the integral 
of f(P) dP between P1 and Pn is given by 

1/2 * { f(P1)[P2-P1] + f(P2)[P3-P1] + f(P3)[P4-P2] + ... + f(Pn-1)[Pn - Pn-2] + f(Pn)[Pn - Pn-1] }

This is written as F * M * P, where F and P are vectors (f(P1),f(P2),...,F(Pn)) and (P1,P2,...,Pn) respectively,
while M is an n x n Matrix given by 

M = 0.5  *  [   -1  1   0   0   0 ...   ... ]
            [   -1  0   1   0   0 ...   ... ]
            [   0   -1  0   1   0 ...   ... ]
            [   0   0   -1  0   1 ...   ... ]
            [   ... ... ... ... ... ... ... ]
            [   ... ... ... ... -1  0   1   ]
            [   ... ... ... ... ... -1  1   ]

"""


def simpson_integral(F,X):  # where f has the same last dimension as the length of the vector x 

    # create M
    n = np.shape(X)[0]
    M = np.zeros([n,n])
    for row in np.arange(1,n-1):
        M[row,row-1] = -0.5
        M[row,row+1] = 0.5
    M[0,0] = -0.5
    M[0,1] = 0.5
    M[n-1,n-1] = 0.5
    M[n-1,n-2] = -0.5   
    
    # matrix multiplication (see description) F*M*X
    MX = np.dot(M,X)
    Integral = np.dot(F,MX)
    return Integral





# levels to integrate over
pMin = 50
pMax = 1000 

# get standard dimensions 
fileName = '/network/aopp/hera/mad/mbengue/data/ERA-INTERIM/daily/ua_2bp6/1980/1201.nc'
nc_standard = Dataset(fileName,'r')
lats = nc_standard.variables['latitude'][:]
lons = nc_standard.variables['longitude'][:]
levels = nc_standard.variables['p'][:]
nLevels, nLats, nLons = levels.shape[0], lats.shape[0], lons.shape[0]





# directories of U and V files
Udirectory = '/network/aopp/hera/mad/mbengue/data/ERA-INTERIM/daily/ua_2bp6/'
Vdirectory = '/network/aopp/hera/mad/mbengue/data/ERA-INTERIM/daily/va_2bp6/'



# count total number of months
nMonths = 0
yearList = sorted(os.listdir(Udirectory))
yearList.remove('2013')
yearList.remove('2014')
#yearList = ['2005']

for yyyy in yearList:
    months_in_yr = np.array([])
    for date in os.listdir(Udirectory+yyyy+'/'):
        months_in_yr = np.append(months_in_yr,date[0:2])      
    nMonths = nMonths + len(np.unique(months_in_yr))




# array of EMFC for each month 
EMFC_monthly_mean = np.zeros((nMonths,nLats,nLons)) 
monthlyTimes = np.array([]) # array of times for writing to Netcdf file

# pressure level normalisation
levelMask = (levels>=pMin) & (levels<=pMax)
sum_over_p_levels = - simpson_integral(np.ones(len(levels[levelMask])),levels[levelMask])    # minus is to account for pressures monotonically decreasing



month_idx = 0
for year in yearList:
    print(year)
    
    # list the dates in that year
    dates = sorted(os.listdir(Udirectory + year +'/'))
    monthStrings = np.array([])
    for date in dates:
        monthStrings = np.append(monthStrings,date[0:2])       
    
    # count the number of days in each month
    all_months, no_days_in_month = np.unique(monthStrings,return_counts=True)   
    day_count = dict(zip(all_months,no_days_in_month))
    
    for month in all_months:    
 #       print(month)
        nDays  = day_count[month]       
        EMFC_vert_int_month = np.zeros((nDays,nLats,nLons))

        for day_idx,day_string in enumerate([x for x in dates if x[0:2]==month]):
            day = day_string[2:4]   # e.g. 01 for 1st 

            # read in Uprime
            UfileName = Udirectory + year + '/' + month + day + '.nc'
            nc = Dataset(UfileName,'r')
            Uprime = nc.variables['ua'][:]
            Uprime = Uprime[0,:,:,:] # flatten in the time dimension
    
            # read in Vprime
            VfileName = Vdirectory + year + '/' + month + day + '.nc'
            nc = Dataset(VfileName,'r')
            Vprime = nc.variables['va'][:]
            Vprime = Vprime[0,:,:,:] # flatten in the time dimension

            # restrict domain to within the pressure bounds
            levelMask = (levels>=pMin) & (levels<=pMax)
            Uprime = Uprime[levelMask,:,:]
            Vprime = Vprime[levelMask,:,:]


            # calculate d(u'v')/dy (emfc) 
            R_earth = 6371000    # earth radius in m
            dTheta  = np.pi/nLats
            emfc = - np.gradient((Uprime * Vprime),axis=1) / (dTheta*R_earth)   # minus is because latitudes are listed from 90,..0, -90
                

            # calculate pressure level integral
            emfc = np.swapaxes(emfc,0,2)   # permute axes until we have the arrangement [lats,lons,levels] so that the integral function works 
            emfc = np.swapaxes(emfc,0,1)  
            EMFC_vert_int_month[day_idx,:,:] = - simpson_integral(emfc,levels[levelMask])    # minus is to account for pressures monotonically decreasing 
        
        # save monthly mean and normalise
        EMFC_monthly_mean[month_idx,:,:] = - np.mean(EMFC_vert_int_month,axis=0) / sum_over_p_levels    # minus is because we want the flux CONVERGENCE
        month_idx = month_idx + 1
        
        # save the time as midday on the 15th of that month
        d = datetime(int(year),int(month),15,12,00)
        newMonthlyTime = date2num(d,nc_standard.variables['t'].units)
        monthlyTimes = np.append(monthlyTimes,newMonthlyTime)
             
        
        """
        m = Basemap(projection='cyl',llcrnrlat=-80,urcrnrlat=80,\
            llcrnrlon=0,urcrnrlon=360)
        m.drawcoastlines()
        m.drawparallels(np.arange(-80.,81.,20.))
        m.drawmeridians(np.arange(0.,360.,20.))
        longitudes, latitudes = np.meshgrid(lons, lats)
        x, y = m(longitudes, latitudes)
        clevs = np.arange(-0.0425,0.04251,0.005)
        cs = m.contourf(x,y,EMFC_monthly_mean[month_idx,:,:],clevs,cmap='RdBu_r')
        m.colorbar()

        plt.show()
        """

   
#######################################################
# save monthly EMFC integrated over pressure levels
######################################################

writingDirectory = '/network/aopp/hera/mad/patterson/ERA-INTERIM/LiWettstein_analysis/Data/'
nc_write = Dataset(writingDirectory + 'vert_int_2bp6_EMFC_1980_2012_test2.nc','w',format='NETCDF4_CLASSIC')

# create global attributes
import time
nc_write.description = "Eddy Momentum Flux Convergence d(u'v')dy integrated between 1000hPa and 50hPa from 2-6day bandpass wind data"
nc_write.history = 'Created ' + time.ctime(time.time()) 
nc_write.source = 'ERA INTERIM daily wind 1979-2015'

# create new dimensions
nc_write.createDimension('longitude', nLons) 
nc_write.createDimension('latitude', nLats)
nc_write.createDimension('t', None ) 

# create dimension variables
lon_var = nc_write.createVariable('longitude',nc.variables['longitude'].dtype,('longitude',))
lat_var = nc_write.createVariable('latitude',nc.variables['latitude'].dtype,('latitude',))
t_var = nc_write.createVariable('t',nc_standard.variables['t'].dtype,('t',))

# add dimension attributes
for ncattr in nc.variables['longitude'].ncattrs():
    lon_var.setncattr(ncattr, nc.variables['longitude'].getncattr(ncattr))
for ncattr in nc.variables['latitude'].ncattrs():
    lat_var.setncattr(ncattr, nc.variables['latitude'].getncattr(ncattr))
for ncattr in nc_standard.variables['t'].ncattrs():
    t_var.setncattr(ncattr, nc_standard.variables['t'].getncattr(ncattr))

nc_write.variables['t'][:] = monthlyTimes
nc_write.variables['latitude'][:] = lats
nc_write.variables['longitude'][:] = lons

# create main variable
EMFC_var = nc_write.createVariable('EMFC',np.float64,('t','latitude','longitude'))
#EMFC_var.setncatts({'standard_name': "Eddy Momentum Flux Convergence",'units': "m s**-2 hPa", 'grid_type': "gaussian", 'source': "GRIB data", 'name': "EMFC", 'title': "Eddy Momentum Flux Convergence", 'date': nc.variables['eke'].getncattr('date'), 'time': nc.variables['eke'].getncattr('time')})
nc_write.variables['EMFC'][:] = EMFC_monthly_mean

# close the file :)
nc_write.close()


###############################
# CHECK THE OUTPUT IS SENSIBLE
###############################

# read in zonal wind data
fileName = "/network/aopp/hera/mad/mbengue/data/ERA-INTERIM/monthly/ua/1980/07.nc"
nc_monthly_wind = Dataset(fileName,'r')
ua = nc_monthly_wind.variables['ua'][:]

t = np.arange(396)
m = Basemap(projection='cyl',llcrnrlat=-80,urcrnrlat=80,\
            llcrnrlon=0,urcrnrlon=360)
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,20.))
m.drawmeridians(np.arange(0.,360.,20.))
longitudes, latitudes = np.meshgrid(lons, lats)
x, y = m(longitudes, latitudes)
clevs = np.arange(-0.00009,0.000091,0.00002)
cs = m.contourf(x,y,EMFC_monthly_mean[6,:,:],clevs,cmap='RdBu_r')
m.colorbar()
m.contour(x,y,np.mean(ua[0,:,:,:],axis=0),np.arange(0,61,5),color='k')

plt.show()

