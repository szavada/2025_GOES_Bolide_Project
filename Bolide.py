import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
from dataclasses import dataclass

class Bolide:
    def __init__(self, dataframe="no dataframe", index="no index", satellite="no satellite"):
        self.dataframe= dataframe
        self.index= index
        self.satellite= satellite
        self.errorlat= 4000 #we're estimating that the error on latitude and longitude measurements are half a pixel, or 4000 meters
        self.errorlong= 4000
        self.rockyrho = 3500 #kg/m^3
        self.metalrho = 7500 #kg/m^3 #constants for bolide density based on possible composition
        self.icyrho = 750 #kg/m^3
        
    def degtom (self, lat1, long1, lat2, long2, r= 6371000): #haversine equation converts change in latitude/longitude to a distance in meters
        lat1rad = np.radians(lat1) #conversion to radians since our measurements are in degrees
        long1rad = np.radians(long1)
        lat2rad = np.radians(lat2)
        long2rad = np.radians(long2)
        dlat = lat2rad - lat1rad #the difference in latitude
        dlong = long2rad - long1rad #the difference in longitude
        a= np.sin(dlat/2)**2 + np.cos(lat1rad)*np.cos(lat2rad)*np.sin(dlong/2)**2 #direct calculation broken into parts
        b= 2*np.arcsin(np.sqrt(a))
        #uncertainty estimation on degtom, each of these 4 lines is representitive of a partial derivitave with respect to that variable 
        errorla1= -r*2*(np.sin(lat1rad)*np.cos(lat2rad)*np.sin(dlong/2)**2+np.cos(dlat/2)*np.sin(dlat/2))/(np.sqrt(1-(np.cos(lat1rad)*np.cos(lat2rad)*np.sin(dlong/2)**2+np.sin(dlat/2)**2)**2))
        errorla2= r*2*(np.cos(dlat/2)*np.sin(dlat/2)-np.cos(lat1rad)*np.sin(lat2rad)*np.sin(dlong/2)**2)/(np.sqrt(1-(np.sin(dlat/2)**2+np.cos(lat1rad)*np.cos(lat2rad)*np.sin(dlong/2)**2)**2))
        errorlo1= -r*2*(np.cos(lat1rad)*np.cos(lat2rad)*np.cos(dlong/2)*np.sin(dlong/2))/(np.sqrt(1-(np.cos(lat1rad)*np.cos(lat2rad)*np.sin(dlong/2)**2+np.sin(dlat/2)**2)**2))
        errorlo2= r*2*(np.cos(lat1rad)*np.cos(lat2rad)*np.cos(dlong/2)*np.sin(dlong/2))/(np.sqrt(1-(np.cos(lat1rad)*np.cos(lat2rad)*np.sin(dlong/2)**2+np.sin(dlat/2)**2)**2))
        #this is the errors added in quadrature 
        errordegtom= np.sqrt((errorla1)**2*self.errorlat**2+(errorla2)**2*self.errorlat**2+(errorlo1)**2*self.errorlong**2+(errorlo2)**2*self.errorlong**2)
        return r*b, errordegtom

    def airrho(self, z, H=8500): #calculates density based on rho naught, height, and scale height
        rho0= 1.225 #kg/m^3                   #8.5 km is the scale height for Earth and 1.225 kg/m^3 is the density of air at sea level
        return (rho0*(1/(np.e**(z/H))))
    
    def radii(self):
        bd= pd.read_csv(self.dataframe)
        rockyradii= [] #empty lists of values we want to find
        errorrocky= []
        icyradii= []
        erroricy= []
        metalradii= []
        errormetal= []
        ilist= []            
        for i in range(len(self.index)): #for every bolide we want to analyze...
            if isinstance(bd.iloc[self.index[i]][f'lat_stereo_{self.satellite}'], float) == True:
                print(f"Satellite {self.satellite} doesn't have an entry for bolide number {self.index[i]}!")
                #pass
            else:
                lat= literal_eval(bd.iloc[self.index[i]][f'lat_stereo_{self.satellite}']) #reads the latitude value for this bolide
                long= literal_eval(bd.iloc[self.index[i]][f'lon_stereo_{self.satellite}']) #reads the longitude value
                alt= np.array(literal_eval(bd.iloc[self.index[i]][f'alt_stereo_{self.satellite}']))*1000 #convert the altitude values to meters 
                energy = literal_eval(bd.iloc[self.index[i]][f'energyJoules_stereo_{self.satellite}']) #the energy curve from each bolide
                erroralt= literal_eval(bd.iloc[self.index[i]][f'residual_dist_stereo_{self.satellite}']) #the error on the altitude is a variable in bd
                for e in range(len(energy)):
                    if energy[e] == np.max(energy):
                        imaxenergy=e #determine the index at which the bolide catastrophically destructs
                totalmass= 0 #these will increase as we sum the steos of our calculation
                totalerror= 0
                for u in range (imaxenergy): #until the bolide catastrophically destructs...
                    deltalatlong = np.abs(self.degtom(lat[u], long[u], lat[u+1], long[u+1]))[0] #calculate the distance in meters for each step in latitude/longitude
                    errordeltalatlong= self.degtom(lat[u], long[u], lat[u+1], long[u+1])[1]
                    deltaalt = np.abs(alt[u+1]-alt[u]) #the change in altitude for each step 
                    erroralt1= (alt[u]-alt[u+1])/np.abs(alt[u+1]-alt[u]) #partial derivative for error calculation
                    erroralt2= (alt[u+1]-alt[u])/np.abs(alt[u]-alt[u+1])
                    errordalt= np.sqrt((erroralt1)**2*(erroralt[u])**2+(erroralt2)**2*(erroralt[u+1])**2) #deltaalt errors added in quadrature 
                    h= np.sqrt(deltalatlong**2 + deltaalt**2) #pythagorean theorum the change in latitude, longitude, and altitude
                    errorh= np.sqrt((deltalatlong/h)**2*(errordeltalatlong)**2+(deltaalt/h)**2*(errordalt)**2) #the error value for h
                    altm = (1/2)*(alt[u]+alt[u+1]) #midpoint altitude for calculating air density
                    rho = self.airrho(altm) #air density
                    erroraltm= .5*np.sqrt(erroralt[u]**2+erroralt[u+1]**2)
                    errorrho= (rho/8500)*erroraltm #the uncertainty on air density
                    totalmass = totalmass+(h*rho) #sum up the total mass for each cylinder
                    terho= (errorrho/rho)**2 
                    teh= (errorh/h)**2
                    totalerror= totalerror+(h*rho*(np.sqrt(terho+teh)))**2 #and the total error for each cylinder
                rockyradii.append((3/4)*(totalmass/self.rockyrho)*100) #+- totalerror #units of cm
                metalradii.append((3/4)*(totalmass/self.metalrho)*100)
                icyradii.append((3/4)*(totalmass/self.icyrho)*100)
                errorrocky.append((3/4)*(np.sqrt(totalerror)/self.rockyrho)*100) #uncertainty estimation for each potential bolide
                errormetal.append((3/4)*(np.sqrt(totalerror)/self.metalrho)*100)
                erroricy.append((3/4)*(np.sqrt(totalerror)/self.icyrho)*100)
                ilist.append(self.index[i])
    #create a dictionary that can be made into a pandas dataframe
            if len(self.index) == 1:
                dict= {f'Index in {self.dataframe}' : ilist[0], 'Satellite' : f'{self.satellite}', 'Rocky Bolide Radii' : rockyradii, 'Rocky Bolide Uncertainty' : errorrocky, 'Metal Bolide Radii' : metalradii, 'Metal Bolide Uncertainty' : errormetal, 'Icy Bolide Radii' : icyradii, 'Icy Bolide Uncertainty' : erroricy}
            else:
                dict= {f'Index in {self.dataframe}' : ilist, 'Satellite' : f'{self.satellite}', 'Rocky Bolide Radii' : rockyradii, 'Rocky Bolide Uncertainty' : errorrocky, 'Metal Bolide Radii' : metalradii, 'Metal Bolide Uncertainty' : errormetal, 'Icy Bolide Radii' : icyradii, 'Icy Bolide Uncertainty' : erroricy}
            bdict = pd.DataFrame(data=dict) #dataframe of all the results
            return(bdict)