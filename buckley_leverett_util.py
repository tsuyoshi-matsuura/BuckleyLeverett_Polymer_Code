# Utilities to do the Buckley Leverett calculations

# Import some packages used in the code below

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import brentq
from scipy.spatial import ConvexHull
from scipy.integrate import quad
from bisect import bisect_left

# Some functions used in the Buckley Leverett code

def interpolate(xval, df, xcol, ycol):
    '''
    Compute xval as the linear interpolation of xval where df is a dataframe and
    df.x are the x coordinates, and df.y are the y coordinates. df.x is expected to be sorted.
    '''
    return np.interp(xval, df[xcol], df[ycol])

def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before
    
def calc_NG(kabs=None, kroe= None, muo=None, rhow=None, rhoo=None,
            qtot=None, area=None, dip=None, params_dict=None):
    '''
    Calculate the dimensionless gravity number NG

    Input:
    kabs: absolute permeability in mD
    kroe: end point oil relperm
    muo:  oil viscosity in cP
    rhow: water density in kg/m3
    rhoo: oil density in kg/m3
    qtot: total flow rate in m3/day
    area: area perpendicular to the flow direction in m2
    dip:  dip angle in degrees, positive = anti-clockwise wrt horizontal

    These parameters can either be set individually by keyword when calling calc_NG or
    using a dictionary 'params_dict'. Priority is given to setting the parameters
    via the keywords.
    '''
    # Constants
    # gravity constant in m/s2
    g = 9.81
    # Conversion from mD to m2
    mD_m2 = 9.869E-16
    # Coversion from cP to Pa.s
    cP_PaS =1E-3
    # Conversion from day to second
    d_s = 24*3600
    # degree to radian
    deg_rad = np.pi/180

    # If a dictionary is provided, use its values unless explicit values are provided
    if params_dict is not None:
        kabs  = kabs  if kabs  is not None else params_dict.get('kabs')
        kroe  = kroe  if kroe  is not None else params_dict.get('kroe')
        muo   = muo   if muo   is not None else params_dict.get('muo')
        rhow  = rhow  if rhow  is not None else params_dict.get('rhow')
        rhoo  = rhoo  if rhoo  is not None else params_dict.get('rhoo')
        qtot  = qtot  if qtot  is not None else params_dict.get('qtot')
        area  = area  if area  is not None else params_dict.get('area')
        dip   = dip   if dip   is not None else params_dict.get('dip')

    # Check for None values and raise an error specifying which parameter is missing
    missing_params = [name for name, value in zip(
        ['kabs', 'kroe', 'muo', 'rhow', 'rhoo', 'qtot', 'area', 'dip'],
        [kabs, kroe, muo, rhow, rhoo, qtot, area, dip]) if value is None]

    if missing_params:
        raise ValueError(f"Missing parameter(s): {', '.join(missing_params)}")
    
    # Parameter checking
    errors = {}
    if kabs <=0:
        errors['kabs'] = 'kabs must be larger than 0'
    if kroe <=0:
        errors['kroe'] = 'kroe must be larger than 0'
    if muo <=0:
        errors['muo'] = 'muo must be larger than 0'
    if rhow <=0:
        errors['rhow'] = 'rhow must be larger than 0'
    if rhoo <=0:
        errors['rhoo'] = 'rhoo must be larger than 0'
    if qtot <0: 
        errors['qtot'] = 'qtot must be larger than 0'
    if area <0: 
        errors['qtot'] = 'qtot must be larger than 0'

    if errors:
        raise ValueError('\n'.join([f'{param}: {msg}' for param, msg in errors.items()]))

    return  (kabs*mD_m2) * kroe * g * (rhow-rhoo) / (muo*cP_PaS) * area / (qtot/d_s) * np.sin(dip*deg_rad)
    
# Main component of the Buckley Leverett utilities

class BuckleyLeverett(object):
    '''
    BuckleyLeverett is used to set up a Buckley Leverett model and determine 
    fractional flow solutions
    
    Input:
    The following parameters are required to set up the Buckley Leverett model:
        krwe, nw, Scw:  water relperm
        kroe, no, Sorw: oil relperm
        muo, muw:       viscosities of oil, water and polymer solution 
                        polymer solution viscosity is at injection concentration
        nSw:            number of water saturation values used in the look-up
                        table for rarefaction wave calculations. By default nSw=5001.
        GravOn:         flag to indicate if gravity effects are included
        NG:             dimensionless gravity number, including dip angle
        eps*, xtol:     numerical parameters, normally the default values are fine.
        debug:          flag to print debug information

    For the input of the viscosity it is not relevant which unit is used, as long
    as for both viscosities the same unit is used.
        
    These parameters can either be set individually by keyword when calling
    BuckleyLeverett or using a dictionary 'params_dict'. Priority is given to setting
    the parameters via the keywords.

    When the Buckley Leverett model is set up all calculations required to determine
    the fractional flow solution are performed assuming an initial water saturation
    equal to the connate water saturation: Swi = Scw.
    If the solution at another initial water saturation is required, it can be changed using
    the method 'set_Swi(Swi)' which will change the initial water saturation in the instance of
    FracFlow to the desired value and update all calculations to reflect this change.
    N.B.: if gravity is included, it is not allowed to change Swi to another value than Scw.

    Note that changing any of the other parameters in an instance of BuckleyLeverett will not
    trigger a re-calculation, but can mess up the calculations. Therefore, it is recommended
    to create a new instance of FracFlow if any of the parameters need to be changed.
    '''

    def __init__(self, krwe=None, nw=None, Scw=None, kroe=None, no=None, Sorw=None,
                 muo=None, muw=None, GravOn= False, NG=0, nSw=5001, 
                 eps_Sw1=1e-8, epsSvel =1e-8, xtol=1e-15,
                 params_dict=None, debug = False):

        # If a dictionary is provided, use its values unless explicit values are provided
        if params_dict is not None:
            krwe = krwe if krwe is not None else params_dict.get('krwe')
            nw   = nw   if nw   is not None else params_dict.get('nw')
            Scw  = Scw  if Scw  is not None else params_dict.get('Scw')
            kroe = kroe if kroe is not None else params_dict.get('kroe')
            no   = no   if no   is not None else params_dict.get('no')
            Sorw = Sorw if Sorw is not None else params_dict.get('Sorw')
            muo  = muo  if muo  is not None else params_dict.get('muo')
            muw  = muw  if muw  is not None else params_dict.get('muw')
            # For the following parameters, keep the default if not provided
            GravOn  = params_dict.get('GravOn', GravOn)
            NG      = params_dict.get('NG', NG)
            nSw     = params_dict.get('nSw', nSw) 
            eps_Sw1 = params_dict.get('eps_Sw1', eps_Sw1)
            epsSvel = params_dict.get('epsSvel', epsSvel)
            xtol    = params_dict.get('xtol', xtol)

        # Check for None values and raise an error specifying which parameter is missing
        missing_params = [name for name, value in zip(
            ['krwe', 'nw', 'Scw', 'kroe', 'no', 'Sorw', 'muo', 'muw'],
            [krwe, nw, Scw, kroe, no, Sorw, muo, muw]) if value is None]

        if missing_params:
            raise ValueError(f"Missing parameter(s): {', '.join(missing_params)}")
        
        self.krwe   = krwe  # end point water relperm
        self.nw     = nw    # water Corey exponent
        self.Scw    = Scw   # connate water saturation
        self.kroe   = kroe  # end point oil relperm
        self.no     = no    # oil Corey exponent
        self.Sorw   = Sorw  # residual oil saturation
        self.muo    = muo   # oil viscosity 
        self.muw    = muw   # water viscosity
        self.GravOn = GravOn # flag to indicate gravity effects are included
        self.NG     = NG    # dimensionless gravity number
        self.nSw    = nSw   # number of saturation values in the lookup tables for rarefaction

        # Parameter checking
        errors = {}
        if self.krwe <=0:
            errors['krwe'] = 'krwe must be larger than 0'
        if self.nw <1:
            errors['nw'] = 'nw must be greater than or equal to 1'
        if self.Scw <0 or self.Scw >= 1:
            errors['Scw'] = 'Scw must be in the range [0,1)'
        if self.kroe <=0:
            errors['kroe'] = 'kroe must be larger than 0'
        if self.no <1:
            errors['no'] = 'no must be greater than or equal to 1'
        if self.Sorw <=0 or self.Sorw >1:
            errors['Sorw'] = 'Sorw must be in the range [0,1)'
        if self.muo <=0:
            errors['muo'] = 'muo must be larger than 0'
        if self.muw <=0:
            errors['muw'] = 'muw must be larger than 0'
        if self.Scw >= 1-self.Sorw:
            errors['Scw_Sorw'] = 'Scw must be less than 1-Sorw'

        if errors:
            raise ValueError('\n'.join([f'{param}: {msg}' for param, msg in errors.items()]))
        
        # Numerical parameters:
        self.eps_Sw1 = eps_Sw1 # used to avoid the spurious solution Sw=Scw when determining Sw1
        self.epsSvel = epsSvel # used to avoid having a singularity when Sw = Swst in the function shock_vel
        self.xtol    = xtol    # tolerance parameter for the solver brentq

        # Allows printing of intermediate results for debugging
        self.debug = debug

        # Set default Swi to Scw
        self.Swi = self.Scw

        # Calculate the end point mobility ratio = displacing fluid mobility / displaced fluid mobility
        self.MobWat = krwe/muw / (kroe/muo)

        # The dataframes ff is used as lookup table for generating the rarefaction solution
        Ss = np.linspace(self.Scw,1.0-self.Sorw,self.nSw)
        fws  = self.fw(Ss)
        dfws = self.dfw_S(Ss)
        self.ff = pd.DataFrame(index=Ss, data={'fw':fws,'dfw':dfws})

        # Set injection saturation
        # In this code the water fractional flow is taken to be 1 at the injection point (xD=0)
        #
        if not GravOn:
            self.Swinj = 1.0 - self.Sorw
        else:
            # if gravity in included, the fractional flow curve can become larger than 1
            # for negative dip angles. Setting Swinj to 1-Sorw in these cases will result
            # in backflow of water. To avoid this the injection saturation is set to the lowest
            # saturation value for which the fractional flow is 1 (note we rely on the fact that sw in 
            # ordered in inceasing order)
            self.Swinj = interpolate(1.0, self.ff.reset_index(names=['sw']), 'fw', 'sw')
        if debug:
            print(f'__init__: Swinj = {self.Swinj:.8e}')
        
        # The general solution of the Buckley Leverett fractional flow problem takes one of
        # the following three forms:
        #   1) shock wave
        #   2) rarefaction wave
        #   3) rarefaction wave followed by a shock. Transition from rarefaction to
        #     shock takes place in Sw=Sw1
        self.Sw1, self.events = self.BL_events(self.Swinj,self.Swi,eps_Sw1=self.eps_Sw1,epsSvel=self.epsSvel,xtol=self.xtol)
        # Generate convex hull solution
        self.hull, self.box, self.vertices, self.wavetype, self.events_hull = self.convex_hull(self.Swinj, self.Swi)
        # Calculate shock front mobility ratio
        if self.events == 'shock':
            self.MobWat_shf = self.shock_mobwat(self.Swinj)
        elif self.events == 'rare + shock':
            self.MobWat_shf = self.shock_mobwat(self.Sw1)
        else:
            # there is no shock, set equal to MobWat
            self.MobWat_shf = self.MobWat

    def set_Swi(self, SwiNew):
        '''
        set_Swi is used to change the initial water saturation in an instance of BuckleyLeverett.
        It will trigger the re-calculation of the fraction flow solution.
        '''
        
        if (SwiNew < self.Scw) or (SwiNew >= self.Swinj):
            raise ValueError(f'Swi must be in the range [{self.Scw},{self.Swinj})')
        # Update the initial water saturation
        self.Swi = SwiNew
        #
        # The general solution of the Buckley Leverett fractional flow problem takes one of
        # the following three forms:
        #   1) shock wave
        #   2) rarefaction wave
        #   3) rarefaction wave followed by a shock. Transition from rarefaction to
        #     shock takes place in Sw=Sw1
        self.Sw1, self.events = self.BL_events(self.Swinj,self.Swi,eps_Sw1=self.eps_Sw1,epsSvel=self.epsSvel,xtol=self.xtol)
        # Generate convex hull solution
        self.hull, self.box, self.vertices, self.wavetype, self.events_hull = self.convex_hull(self.Swinj, self.Swi)
        # Calculate shock front mobility ratio
        if self.events == 'shock':
            self.MobWat_shf = self.shock_mobwat(self.Swinj)
        elif self.events == 'rare + shock':
            self.MobWat_shf = self.shock_mobwat(self.Sw1)
        else:
            # there is no shock, set equal to MobWat
            self.MobWat_shf = self.MobWat

        return self.Swi

    def S_D(self,Sw):
        '''
        The function S_D returns the normalized water saturation = (Sw-Scw) / (1-Scw-Sorw).

        The normalized water saturation is used in the calculation of the fractional
        flow function and its derivatives
        '''
        return (Sw-self.Scw) / (1.0-self.Scw-self.Sorw)

    def fw(self,Sw):
        '''
        The function fw returns the fractional flow value = q_wat / (q_wat + q_oil) as function
        of the water saturation Sw.
        '''
        S = self.S_D(Sw)
        fws = self.MobWat*S**self.nw/(self.MobWat*S**self.nw + (1-S)**self.no)
        if self.GravOn:
            fws = fws * (1.0 - self.NG * (1-S)**self.no)
        return fws
    
    def dfw_S(self,Sw):
        '''
        The function dfw_S returns the derivative of the fractional flow function wrt Sw.
        It depends on the water saturation.
        '''
        S = self.S_D(Sw)
        num = self.MobWat * S**(self.nw-1) * (1-S)**(self.no-1) * (self.nw*(1-S)+self.no*S)
        denom = self.MobWat*S**self.nw + (1-S)**self.no
        dfws = num/denom**2
        if self.GravOn:
            dfws = dfws * (1.0 - self.NG * (1-S)**self.no)
            add_term = self.MobWat*S**self.nw / denom * self.NG * self.no*(1-S)**(self.no-1)
            dfws = dfws + add_term
        return dfws / (1.0-self.Scw-self.Sorw)
    
    def dfw_S2(self,Sw):
        '''
        The function dfw_S2 returns the second derivative of the fractional flow function wrt Sw.
        It depends on the water saturation Sw.

        This function is used to test convexity / concavity of the fractional flow curve in 
        the function BL_events.
        It could also be used to calculate the S-wave rarefaction solution as an alternative to using
        the lookup method.
        '''
        S = self.S_D(Sw)
        num1 = (1-S)**self.no * (self.nw**2*(1-S)**2 + self.no*(1+self.no)*S**2 + self.nw*(1-S)*(-1+(1+2*self.no)*S))
        num2 = self.MobWat*S**self.nw * (self.nw**2*(1-S)**2 + self.no*(self.no-1)*S**2 + self.nw*(1-S)*(1+(2*self.no-1)*S))
        num = self.MobWat * (1-S)**(self.no-2) * S**(self.nw-2) * (num1 - num2)
        denom = (1-S)**self.no + self.MobWat*S**self.nw
        dfws2 = num / denom**3
        if self.GravOn:
            dfws2 = dfws2 * (1.0 - self.NG * (1-S)**self.no)
            add_term1 = 2*self.MobWat * S**(self.nw-1) * (1-S)**(self.no-1) * (self.nw*(1-S)+self.no*S)
            add_term1 = add_term1 * self.NG * self.no * (1-S)**(self.no-1) / denom**2
            add_term2 = self.MobWat * S**self.nw / denom * self.NG * self.no * (self.no-1) * (1-S)**(self.no-2)
            dfws2 = dfws2 + add_term1 - add_term2
        return dfws2 / (1.0-self.Scw-self.Sorw)**2
    
    def shock_mobwat(self,Sw_shf):
        '''
        This function calculates the shock front mobility ratio:
              displacing fluid mobility / displaced fluid mobility
        '''
        S_shf = self.S_D(Sw_shf)
        S_wi  = self.S_D(self.Swi)
        displacing_mob = (self.krwe*S_shf**self.nw/self.muw) + (self.kroe*(1.0-S_shf)**self.no/self.muo)
        displaced_mob  = (self.krwe*S_wi**self.nw /self.muw) + (self.kroe*(1.0-S_wi)**self.no /self.muo)
        return displacing_mob / displaced_mob
    
    def shock_vel(self, Sw, Swst, epsSvel=1e-8):
        '''
        The function shock_vel calculates the speed of the shock between Sw and Swst.

        The parameter epsSvel is used to avoid having a singularity when Sw = Swst by returning
        the derivative value dfw/dSw(Swst) if (Sw-Swst) < epsSvel.

        The function is able to handle both a float and pandas series / numpy array as input
        for the parameter Sw.
        '''
        if isinstance(Sw,(pd.core.series.Series,np.ndarray)):
            select = (np.abs(Sw-Swst)<epsSvel)
            result = np.empty_like(Sw)
            result[select] = self.dfw_S(Swst)
            result[~select] = (self.fw(Sw[~select])-self.fw(Swst))/(Sw[~select]-Swst)
            return result
        else:
            if np.abs(Sw-Swst)<epsSvel:
                return self.dfw_S(Swst)
            else:
                return (self.fw(Sw)-self.fw(Swst))/(Sw-Swst)
    
    def lam_shock(self, Sw,Swi,epsSvel=1e-8):
        '''
        The function lam_shock is used to determine the value for Sw
        where the rarefaction speed matches the speed of the shock between Sw and Swi.

        This function is used in the function find_events

        See function shock_vel for the parameter epsSvel
        '''
        return self.dfw_S(Sw) - self.shock_vel(Sw, Swi, epsSvel)
    
    def BL_events(self,Swinj,Swi,eps_Sw1=1e-8, epsSvel=1e-8, xtol=1e-15):
        '''
        This function determines how the solution between Sw=Swinj and Sw=Swi looks like.
        Three possible scenarios depending on the Swi value:
           1) shock wave         : 'events' = 'shock'
           2) rarefaction wave   : 'events' = 'rare'
           3) rarefaction + shock: 'events' = 'rare + shock'. Transition from rarefaction to shock
              takes place at Sw=Sw1

        The function returns the scenario corresponding to the Swi value in 'events'. It also calculates
        an Sw1 value, which is only relevant in the 'rare + shock' scenario.

        The parameter 'eps_Sw1' is used to avoid a 'rare + shock' solution with a very small
        rarefaction part. It is also used to avoid the spurious solution Sw=Swi.
        See function shock_vel for the parameter epsSvel
        '''

        if (self.no==1) and (self.nw==1) and (self.MobWat==1) and ((self.NG==0) or (self.GravOn==False)):
            # Special case of linear fractional flow
            # Contact discontinuity modelled as a shock
            Sw1 = Swi
            events = 'shock'
        else:
            # ept_Sw1 is introduced to avoid the spurious solution Sw=Swi and to avoid a 'rare + shock' solution
            # with a very small rarefaction part
            if (not self.GravOn) or (self.NG<=0):
                # In this case the fractional flow curve is monotonically increasing between Swi and Swinj
                # Keep in mind that for negative NG values, Swinj can be lower that 1-Sorw and the 
                # part where the fractional flow curve is decreasing is for Sw>Swinj.
                Sw_lo = Swi + eps_Sw1
                Sw_hi = Swinj - eps_Sw1
            else:
                # If gravity is included and NG>0, there can be a part of the fractional flow curve
                # where it is decreasing which can result in a two intersection points between the
                # shock speed and the rarefaction speed (the zeroes of lam_shock). We are looking 
                # for the intersection point with the larger Sw value. This is achieved by setting the 
                # lower bound for brentq to the Sw value where the derivative of fw is zero.
                try:
                    Sw_lo = brentq(self.dfw_S, Swi+eps_Sw1,Swinj-eps_Sw1, xtol=xtol)
                    if self.debug:
                        print(f'BL_events: Sw_lo = {Sw_lo:.8e}')
                except ValueError as e:
                    Sw_lo = Swi + eps_Sw1
                    if self.debug:
                        print(f'BL_events: Sw_lo = {Sw_lo:.8e}, Error from brentq: ', e)
                Sw_hi = Swinj - eps_Sw1
            try:
                Sw1 = brentq(lambda Sw: self.lam_shock(Sw,Swi=Swi,epsSvel=epsSvel),Sw_lo,Sw_hi,xtol=xtol)
                events = 'rare + shock'
                if self.debug:
                    print(f'BL_events: Sw1 = {Sw1:.8e}, events = {events}')
            except ValueError as e:
                Sw1 = Swi      
                # Second derivative of fw is used to check concavity (Swi<Sw2) or convexity (Swi>=Sw2)
                if (self.dfw_S2(0.5*(Swi+Swinj))>=0):
                    events = 'shock'
                else:
                    events = 'rare'
                if self.debug:
                    print(f'BL_events: Sw1 = {Sw1:.8e}, events = {events}, Error from brentq: ', e)   
        return Sw1, events
    
    def convex_hull(self, Sl, Sr, nS=5001):
        '''
        Determineq the convex hull of the fractional flow curve
        Assumes Sr < Sl
        '''

        # S range for which the convex hull needs to be created
        # {(S,y): sr <= S <= Sl and y <= fw(S)}
        # S range in reverse order
        Srange = np.linspace(Sl,Sr,nS)

        # Determine bottom of convex hull
        fwmin = np.min(self.fw(Srange))-0.1

        # Create box for the convex hull: fraction flow curve + two bottom points
        box = np.append(np.array(((Srange,self.fw(Srange)))).T, [[Sr,fwmin]],axis=0)
        box = np.append(box,[[Sl,fwmin]],axis=0)
        #
        # Determine the hull of the box
        hull=ConvexHull(box)
        vertices = hull.vertices
        #
        # Re-order the vertices so that it starts with 0 = (Sl,fw(Sl))
        # Note that the vertices are circular and in counter clockwise direction
        for i in range(vertices.shape[0]):
            if vertices[0]!=0:
                keep = vertices[0]
                vertices = np.delete(vertices,0)
                vertices = np.append(vertices,keep)
        # The last two vertices are the ones that were added at the bottom: remove them
        last = vertices.shape[0]-1
        vertices = np.delete(vertices,[last-1,last])

        # Determine rarefactions and shocks
        # Consecutive vertices form rarefactions, jumps in vertices are shocks
        shock    = {}
        raref    = {}
        wavetype = {}
        # Start by assuming that the first event is a rarefaction. This is checked later 
        # by removing rarefaction with only a single vertex
        val_new        = vertices[0]
        raref[val_new] = val_new
        wavetype[0]    = 'r'
        # Loop over the vertices:
        #   1) jumps in vertices are shocks
        #   2) consecutive vertices are rarefactions
        # shock[i] = j means that there is a shock from vertex i to j
        # raref[i] = j means that there is a rarefaction from vertex i to j
        for i in range(vertices.shape[0]-1):
            val_curr = vertices[i]
            val_next = vertices[i+1]
            if (val_next-val_curr)>1:
                # Jump in vertices: shock
                shock[val_curr] = val_next
                #
                val_new        = val_next
                raref[val_new] = val_next
                #
                wavetype[val_curr] = 's'
                wavetype[val_new]  = 'r'
            else:
                raref[val_new]=val_next
        # Remove rarefactions with only single point
        keys = []
        for k, v in raref.items():
            if k==v:
                keys.append(k)
        for k in keys:
            del raref[k]
        # Remove spurious rarefaction at the end
        if (nS-1) in wavetype:
            del wavetype[nS-1]

        # Now put every thing into events
        events = {}
        for i, (k,v) in enumerate(wavetype.items()):
            if v =='s':
                Sl = box[k,0]
                Sr = box[shock[k],0]
                events[i] = ('s', Sl, Sr)
                wavetype[k] =('s', shock[k])
            else:
                Sl = box[k,0]
                Sr = box[raref[k],0]
                events[i] = ('r', Sl, Sr)
                wavetype[k] =('r', raref[k])
                
        return hull, box, vertices, wavetype, events
    
    def rare_wave(self, ksi, Sl, Sr):
        '''
        This function returns the rarefaction solution Sw(ksi) for ksi = xD/tD where
        xD is the dimensionless position at dimensionless time tD.
        Sl and Sr are the saturation boundaries of the rarefaction wave, where Sl is
        the slower saturation (Sr<Sl)

        The solution is determined by inverting the relation for the rarefaction:
                ksi = x/t = dfw/dSw(Sw(ksi))
        using a lookup in a table of (Sw,dfw/dSw) values.
        '''
        # To ensure that the lookup is uniquely determined we restrict the
        # saturations in the table to be between Sr and Sl.
        smin = take_closest(self.ff.index,Sr)
        smax = take_closest(self.ff.index,Sl)
      
        data = self.ff[smin:smax]
        # Change the table index (contains Sw values) to a table column with name 'Sw'
        data = data.reset_index(names=['Sw'])
        # Perform the lookup: given ksi=dfw, find Sw
        Sw = interpolate(ksi, data.sort_values(by='dfw'), 'dfw', 'Sw')
        return Sw
    
    def calc_Sol(self,x,t):
        '''
        The function calc_Sol calculates the water saturation Sw at dimensionless position xD=x and
        dimensionless time tD=t for an instance of BuckleyLeverett.
        '''
        # The solution is one of the following scenarios:
        #   a) rarefaction from Swinj to Swi
        #   b) shock from Swinj to Swi
        #   c) rarefaction from Swinjwto Sw1, followed by shock from Sw1 to Swi
        #
        if t<=0:
            if x<=0:
                return self.Swinj
            else:
                return self.Swi
        else: 
            ksi = x/t

            if ksi <= 0:
                return self.Swinj
            else:
                if self.events == 'rare':
                    if ksi <= self.dfw_S(self.Swi):
                        # rarefaction wave from Swinj to Swi
                        return self.rare_wave(ksi, Sl=self.Swinj, Sr=self.Swi)
                    else:
                        return self.Swi
                elif self.events == 'shock':
                    if ksi <= self.shock_vel(self.Swinj,self.Swi):
                        # Sw2 upto the S-wave shock (C=0) from Sw2 to Swi occurs
                        return self.Swinj
                    else:
                        return self.Swi
                else: # self.events == 'rare + shock'
                    if ksi <= self.dfw_S(self.Sw1):
                        # rarefaction wave from Swinj to Sw1
                        # Note that the front of this rarefaction wave has the same speed
                        # as the shock from Sw1 to Swi, i.e dfw_S(Sw1) = shock_vel(Sw1,Swi)
                        return self.rare_wave(ksi, Sl=self.Swinj, Sr=self.Sw1)
                    else:
                        return self.Swi
                    
    def calc_Sol_hull(self, x,t):
        if t<=0:
            if x<=0:
                return self.Swinj
            else:
                return self.Swi
        else:
            ksi = x/t
            last_ev = list(self.wavetype)[-1]
            for k,v in self.wavetype.items():
                sr = self.box[v[1],0]
                sl = self.box[k,0]
                if v[0] == 's':
                    vel_s = self.shock_vel(sl,sr)
                    if ksi <= vel_s:
                        return sl
                    elif k==last_ev:
                        return sr
                elif v[0]=='r':
                    if ksi <= self.dfw_S(sl):
                        return sl
                    elif ksi <= self.dfw_S(sr):
                        return self.rare_wave(ksi,Sl=sl,Sr=sr)
                    elif k==last_ev:
                        return sr
                    
    def calc_tB(self):
        '''
        This function calculated the breakthrough time
        '''
        if self.events == 'shock':
            # Solution C
            v = self.shock_vel(self.Swinj, self.Swi)
            tB = 1.0/v
        elif self.events == 'rare + shock':
            # Solution B
            v = self.shock_vel(self.Sw1,self.Swi)
            tB = 1.0/v
        else:
            # Solution A
            v = self.dfw_S(self.Swi)
            tB = 1.0/v
        return tB

    def calc_Savg(self,t):
        '''
        This function calculates the average water saturation in the interval 0 <= xD <= 1 at tD=t.

        The expressions are based on analytical calculations.
        '''
        tB = self.calc_tB()
        #
        if t <= tB:
            return self.Swi + (self.fw(self.Swinj)-self.fw(self.Swi))*t
        else:
            Sw = self.calc_Sol(1,t)
            return Sw + (self.fw(self.Swinj) - self.fw(Sw))*t
            
    def calc_RF(self,t,wrtSwi=False):
        '''
        This function calculates the recovery factor at tD=t.
        Note that tD is equivalent to # of PV injected.

        The flag 'wrtSwi' determines the reference initial condition for the RF calculation:
        1) if True, the RF is calculated wrt Swi
        2) if False, the RF is calculated wrt Scw
        '''
        if wrtSwi:
            Sini = self.Swi
        else:
            Sini = self.Scw
        return (self.calc_Savg(t) - Sini) / (1.0 - Sini)

    def plot_profile(self, PVs, xstart=0.0, xend=1.0, ns = 101, eps=1e-4, sizex = 6, sizey = 4):
        '''
        This function plots the profile of the water saturation Sw
        for the BuckleyLeverett instance in the interval xstart <= xD <= xend 
        at the PV(s) injected specified by PVs.

        Input:
        PVs:    PV injected at which the profiles should be plotted, can be a single number or 
                a list, series or array of PVs
        xstart: xD value at which the profile should start
        xend:   xD value at which the profile should end
        ns:     number of xD points in the profile

        The function returns:
        data:       the Sw profile data for the last profile plotted
        fig, ax:    figure and axes data of the last plot
        '''
        # If only one number is input for PVs, convert to list
        if isinstance(PVs, (int, float)):
            PVs = [PVs]

        for PV in PVs:
            fig, ax = plt.subplots(figsize=(sizex,sizey))
            data = pd.Series()
            t = PV
            points = self.calc_quad_points(t, xstart, xend)
            if len(points)>=1:
                xs = np.linspace(xstart,points[0]-eps,ns)
                xs = np.append(xs,[points[0]])
                xs = np.append(xs,np.linspace(points[0]+eps,xend,10))
            else:
                xs = np.linspace(xstart,xend,ns)
            # xs = np.linspace(xstart,xend,ns)
            for x in xs:
                data[x] = self.calc_Sol(x,t)
            ax.plot(data, 'b-', label='Sw profile')
            ax.set_ylim(-0.05,1.05)
            ax.set_xlabel('xD')
            ax.set_ylabel('Sw (frac)')
            ax.set_title(f'{PV:4.2f} PV injected')
            ax.legend()
            ax.grid()
            plt.show()

        return data, fig, ax
    
    def calc_quad_points(self, t, xstart=0.0, xend=1.0):
        '''
        This function generates the 'points' input data for the SciPy integration 
        routine 'quad' used in 'plot_Savg_Integration'.
        It specifies the discontinuitiess in the water saturation profile.
        '''
        if self.events == 'shock':
            # Solution C
            v = self.shock_vel(self.Swinj, self.Swi)
            xs = [v*t]
        elif self.events == 'rare + shock':
            # Solution B
            v = self.shock_vel(self.Sw1,self.Swi)
            xs = [v*t]
        else:
            # Solution A   
            v = self.dfw_S(self.Swi)
            xs=[v*t]
        points = []
        for x in xs:
            if (x>xstart) and (x<xend): points.append(x)
        return points
    
    def plot_Savg_Integration(self, PVstart=0.0, PVend=2.0, eps=1e-4, ns = 21, quad_eps=1e-5):
        '''
        This function plots the average water saturation as function of PV injected for
        an instance of BuckleyLeverett. The average water saturation is calculated by integrating
        the Sw profile over the range 0 <= xD <= 1. The integation is performed by
        the SciPy integration routine 'quad'.

        Input:
        PVstart:    start value of PV injected in the plot
        PVend:      end value of PV injected in the plot
        ns:         number of "PV injected" points in the plot
        quad_eps:   tolerance parameter value used in the SciPy integration routine 'quad'

        The function returns:
        Savg:       Savg vs PV data in the plot
        fig, ax:    figure and axes data of the plot
        '''
        tB = self.calc_tB()
        if (tB>PVstart) and (tB<PVend):
            PVs = np.linspace(PVstart,tB-eps,5)
            PVs = np.append(PVs,[tB])
            PVs = np.append(PVs,np.linspace(tB+eps,PVend,ns))
        else:
            PVs = np.linspace(PVstart,PVend,ns)
        # PVs = np.linspace(PVstart,PVend,ns)
        Savg = pd.Series()
        for PV in PVs:
            points = self.calc_quad_points(PV)
            Savg[PV] = quad(self.calc_Sol,0,1,args=(PV,), points=points,epsrel=quad_eps, epsabs=quad_eps)[0]

        fig, ax = plt.subplots()
        ax.plot(Savg, 'r*-', label = 'Average water saturation')
        ax.set_xlabel('PV injected')
        ax.set_ylabel('Average water saturation (frac)')
        ax.set_title('Average water saturation')
        ax.legend()
        ax.grid()

        return Savg, fig, ax
    
    def plot_Savg_brute_force(self, PVend=2.0, ns=2001):
        '''
        This function plots the average water saturation as function of PV injected for
        an instance of BuckleyLeverett. The average water saturation is calculated by performing
        a 'brute force' integration of the 'net injected porevolume'

        Input:
        PVend:      end value of PV injected in the plot
        ns:         number of "PV injected" points in the plot. It also determines the 
                    integration step size, so it should not be too small.

        The function returns:
        Savg:       Savg vs PV data in the plot
        fig, ax:    figure and axes data of the plot
        '''
        # This function only works from PVstart = 0.0
        PVstart = 0.0
        # DelPV = integration step size
        PVs, DelPV = np.linspace(PVstart,PVend,ns, retstep=True)
        NetInj = pd.Series()
        for PV in PVs:
            # NetInj = net injected porevolume in the period PV to PV+DelPV
            NetInj[PV] = (1.0 - self.fw(self.calc_Sol(1.0,PV)))*DelPV
        # .cumsum carries out the integration
        Savg = self.Swi + NetInj.cumsum()

        fig, ax = plt.subplots()
        ax.plot(Savg, 'b-', label = 'Average water saturation')
        ax.set_xlabel('PV injected')
        ax.set_ylabel('Average water saturation (frac)')
        ax.set_title('Average water saturation')
        ax.legend()
        ax.grid()

        return Savg, fig, ax
    
    def plot_Savg(self, PVstart=0.0, PVend=2.0, eps=1e-4, ns = 101):
        '''
        This function plots the average water saturation as function of PV injected for
        an instance of BuckleyLeverett. The average water saturation is calculated using the
        function 'calc_Savg'

        Input:
        PVstart:    start value of PV injected in the plot
        PVend:      end value of PV injected in the plot
        ns:         number of "PV injected" points in the plot

        The function returns:
        Savg:       Savg vs PV data in the plot
        fig, ax:    figure and axes data of the plot
        '''
        tB = self.calc_tB()
        if (tB>PVstart) and (tB<PVend):
            PVs = np.linspace(PVstart,tB-eps,10)
            PVs = np.append(PVs,[tB])
            PVs = np.append(PVs,np.linspace(tB+eps,PVend,ns))
        else:
            PVs = np.linspace(PVstart,PVend,ns)
        # PVs = np.linspace(PVstart,PVend,ns)
        #
        Savg = pd.Series()
        for PV in PVs:
            Savg[PV] = self.calc_Savg(PV)
        
        fig, ax = plt.subplots()
        ax.plot(Savg, 'b-', label = 'Average water saturation')
        ax.set_xlabel('PV injected')
        ax.set_ylabel('Average water saturation (frac)')
        ax.set_title('Average water saturation')
        ax.legend()
        ax.grid()
        return Savg, fig, ax
    
    def plot_RF(self, PVstart=0.0, PVend=2.0, eps=1e-4, ns = 201):
        '''
        This function plots the RF and BSW as function of PV injected for
        an instance of BuckleyLeverett. The RF is calculated wrt to both Swi and Scw as
        initial condition.

        Input:
        PVstart:    start value of PV injected in the plot
        PVend:      end value of PV injected in the plot
        ns:         number of "PV injected" points in the plot

        The function returns:
        RF_Scw:     RF wrt Scw vs PV data in the plot
        RF_Swi:     RF wrt Swi vs PV data in the plot
        BSW:        BSW vs PV data in the plot
        fig, ax:    figure and axes data of the plot

        '''
        tB = self.calc_tB()
        if (tB>PVstart) and (tB<PVend):
            PVs = np.linspace(PVstart,tB-eps,10)
            PVs = np.append(PVs,[tB])
            PVs = np.append(PVs,np.linspace(tB+eps,PVend,ns))
        else:
            PVs = np.linspace(PVstart,PVend,ns)
        # PVs = np.linspace(PVstart,PVend,ns)
        RF_Scw = pd.Series()
        RF_Swi = pd.Series()
        BSW = pd.Series()

        for PV in PVs:
            RF_Scw[PV] = self.calc_RF(PV)
            RF_Swi[PV] = self.calc_RF(PV,wrtSwi=True)
            BSW[PV] = self.fw(self.calc_Sol(1,PV))

        fig, ax = plt.subplots()
        ax.plot(RF_Scw, 'r-', label = f'RF wrt Scw = {self.Scw:4.2f}')
        ax.plot(RF_Swi, 'g--', label = f'RF wrt Swi = {self.Swi:4.2f}')
        ax.plot(BSW, 'b-', label = 'BSW')
        ax.set_xlabel('PV injected')
        ax.set_ylabel('RF or BSW (frac)')
        ax.set_title('Recovery factor vs PV injected')
        ax.legend()
        ax.set_ylim(-0.01,1.05)
        ax.set_yticks(np.arange(0,1.05,0.1))
        ax.grid()

        return RF_Scw, RF_Swi, BSW, fig, ax
    
    def plot_solution(self, sizex=12, sizey=6):
        '''
        This function plots the fractional flow solution for an instance of BuckleyLeverett.

        The solution is one of the following scenarios
            a) rarefaction from Swinj to Swi
            b) shock (from Swinj to Swi
            c) rarefaction from Swinj to Sw1, followed by a shock from Sw1 to Swi

        Input:
        sizex:  x-dimension of the plot
        sizey:  y-dimension of the plot

        The function returns:
        fig, ax:    figure and axes data of the plot
        '''
        fig, ax = plt.subplots(figsize=(sizex,sizey))

        Ss = np.linspace(self.Scw,1.0-self.Sorw,101)
        ax.plot(Ss,self.fw(Ss), 'b-', label = 'fw')

        ax.plot(self.Swinj,self.fw(self.Swinj),'ro', label=f'Swinj={self.Swinj:6.4f}', markersize=8, fillstyle='none')

        if self.events == 'rare':
            Ss = np.linspace(self.Swinj,self.Swi,10)
            ax.plot(Ss,self.fw(Ss),'r.', label = 'rarefaction')
            vel = self.dfw_S(self.Swi)
            print(f'Mobility ratio = {self.MobWat:4.2e}')
            print(f'rarefaction speed, front = {vel:4.2e}')
            print(f'Breakthrough after {1/vel:4.2e} PV injected')
        elif self.events == 'shock':
            ax.plot((self.Swinj,self.Swi),(self.fw(self.Swinj),self.fw(self.Swi)),'g-', label = 'shock', lw=2)
            vel = self.shock_vel(self.Swinj,self.Swi)
            print(f'Mobility ratio             = {self.MobWat:4.2e}')
            print(f'Shock front mobility ratio = {self.MobWat_shf:4.2e}')
            print(f'shock speed = {vel:4.2f}')
            print(f'Breakthrough after {1/vel:4.2e} PV injected')
        else: # self.events == 'rare + shock'
            Ss = np.linspace(self.Swinj,self.Sw1,10)
            ax.plot(Ss,self.fw(Ss),'r.', label = 'rarefaction')
            ax.plot(self.Sw1,self.fw(self.Sw1),'go', label = f'Sw1={self.Sw1:6.4f}', markersize=8,fillstyle='none')
            ax.plot((self.Sw1,self.Swi),(self.fw(self.Sw1),self.fw(self.Swi)),'g-', label = 'shock', lw=2)
            print(f'Sw1 = {self.Sw1:6.4f}')
            vel = self.shock_vel(self.Sw1,self.Swi)
            print(f'Mobility ratio             = {self.MobWat:4.2e}')
            print(f'Shock front mobility ratio = {self.MobWat_shf:4.2e}')
            print(f'rarefaction speed, front = shock speed = {vel:4.2e}')
            print(f'Breakthrough after {1/vel:4.2e} PV injected')

        ax.plot(self.Swi,self.fw(self.Swi),'bo', label=f'Swi={self.Swi:6.4f}', markersize=8, fillstyle='none')

        ax.set_xlabel('Sw')
        ax.set_ylabel('fw')
        ax.grid()
        ax.legend()
        
        return fig, ax