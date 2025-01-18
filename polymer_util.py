# Utilities to do the polymer calculations

# Import some packages used in the code below

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import brentq
from scipy.integrate import quad
from bisect import bisect_left

# Some functions used in the polymer flooding code

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

# Main component of the polymer utilities

class FracFlow(object):
    '''
    FracFlow is used to set up a polymer model and determine fractional flow solutions
    
    Input:
    The following parameters are required to set up the polymer model:
        krwe, nw, Scw:  water relperm
        kroe, no, Sorw: oil relperm
        muo, muw, mup:  viscosities of oil, water and polymer solution 
                        polymer solution viscosity is at injection concentration
        ad1, ad2:       adsorption parameter for Langmuir isotherm
        nSw:            number of water saturation values used in the look-up
                        table for rarefaction wave calculations. By default nSw=5001.
        eps*, xtol:     numerical parameters, normally the default values are fine.
        debug:          flag to print debug information

    For the input of the viscosity it is not relevant which unit is used, as long
    as for all three viscosities the same unit is used.
        
    These parameters can either be set individually by keyword when calling FracFlow or
    using a dictionary 'params_dict'. Priority is given to setting the parameters
    via the keywords.

    When the polymer model is set up all calculations required to determine the fractional
    flow solution are performed assuming an initial water saturation equal to the connate
    water saturation: Swi = Scw.
    If the solution at another initial water saturation is required, it can be changed using
    the method 'set_Swi(Swi)' which will change the initial water saturation in the instance of
    FracFlow to the desired value and update all calculations to reflect this change.

    The polymer concentration used in this class is the normalized polymer concentration:
    C = polymer concentration / injection concentration

    Note that changing any of the other parameters in an instance of FracFlow will not
    trigger a re-calculation, but can mess up the calculations. Therefore, it is recommended
    to create a new instance of FracFlow if any of the parameters need to be changed.
    '''
    
    def __init__(self, krwe=None, nw=None, Scw=None, kroe=None, no=None, Sorw=None,
                 muo=None, muw=None, mup=None, ad1=None, ad2=None, nSw=5001, 
                 eps_Sw1=1e-8, eps_Sw2=1e-8, eps_Sw3=1e-8, epsAds=1e-8, epsSvel=1e-8, xtol=1e-15,
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
            mup  = mup  if mup  is not None else params_dict.get('mup')
            ad1  = ad1  if ad1  is not None else params_dict.get('ad1')
            ad2  = ad2  if ad2  is not None else params_dict.get('ad2')
            # For the following parameters, keep the default if not provided
            nSw     = params_dict.get('nSw', nSw) 
            eps_Sw1 = params_dict.get('eps_Sw1', eps_Sw1)
            eps_Sw2 = params_dict.get('eps_Sw1', eps_Sw2)
            eps_Sw3 = params_dict.get('eps_Sw3', eps_Sw3) 
            epsAds  = params_dict.get('epsAds', epsAds) 
            epsSvel = params_dict.get('epsSvel', epsSvel)
            xtol    = params_dict.get('xtol', xtol)
            

        # Check for None values and raise an error specifying which parameter is missing
        missing_params = [name for name, value in zip(
            ['krwe', 'nw', 'Scw', 'kroe', 'no', 'Sorw', 'muo', 'muw', 'mup', 'ad1', 'ad2'],
            [krwe, nw, Scw, kroe, no, Sorw, muo, muw, mup, ad1, ad2]) if value is None]

        if missing_params:
            raise ValueError(f"Missing parameter(s): {', '.join(missing_params)}")

        self.krwe = krwe  # end point water relperm
        self.nw   = nw    # water Corey exponent
        self.Scw  = Scw   # connate water saturation
        self.kroe = kroe  # end point oil relperm
        self.no   = no    # oil Corey exponent
        self.Sorw = Sorw  # residual oil saturation
        self.muo  = muo   # oil viscosity 
        self.muw  = muw   # water viscosity
        self.mup  = mup   # viscosity of polymer solution at injection concentration
        self.ad1  = ad1   # adsorption parameter Langmuir isotherm
        self.ad2  = ad2   # adsorption parameter Langmuir isotherm
        self.nSw  = nSw   # number of saturation values in the lookup tables for rarefaction

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
        if self.mup <=0:
            errors['mup'] = 'mup must be larger than 0'
        if self.ad1 <0: 
            errors['ad1'] = 'ad1 must be larger than or equal to 0'
        if (self.ad1 >0) and  (self.ad2 <=0):
            errors['ad2'] = 'ad2 must be larger than 0 if ad1 > 0'
        if self.Scw >= 1-self.Sorw:
            errors['Scw_Sorw'] = 'Scw must be less than 1-Sorw'
        if self.muw > self.mup:
            errors['muw_mup'] = 'muw must be smaller than or equal to mup'

        if errors:
            raise ValueError('\n'.join([f'{param}: {msg}' for param, msg in errors.items()]))

        # Numerical parameters:
        self.eps_Sw1 = eps_Sw1 # used to avoid the spurious solution Sw=Scw when determining Sw1
        self.eps_Sw2 = eps_Sw2 # used to avoid the spurious solution Sw=1-Sorw when determining Sw2
        self.eps_Sw3 = eps_Sw3 # used to avoid the spurious solution Sw=Swi when determining Sw3
                               # also to avoid 'rare+shock' solutions with a very short rarefaction part
        self.epsAds  = epsAds  # used to avoid having a singularity when C = Cst in the function Dads
        self.epsSvel = epsSvel # used to avoid having a singularity when Sw = Swst in the function shockS_vel
        self.xtol    = xtol    # tolerance parameter for the solver brentq

        # Allows printing of intermediate results for debugging
        self.debug = debug
       
        # Set default Swi to Scw
        self.Swi = self.Scw
        # lowSwi is a flag to keep track which branch of the fractional flow solution should be used
        # for the S-wave (C=0) part of the polymer solution.
        self.lowSwi = True

        # Set injection saturation. This corresponds to an injection fractional flow of 1.
        self.Swinj = 1.0 - self.Sorw

        # Calculate the end point mobility ratio for the pure water case.
        # MobWat = = displacing fluid mobility / displaced fluid mobility = water mobility / oil mobility 
        self.MobWat = krwe/muw / (kroe/muo)
        # Calculate the relative change in aqueous phase viscosity due to polymer
        self.delMu = (mup - muw)/muw

        # The dataframes ffC0 and ffC1 are used as lookup tables when generating the S-wave rarefaction
        # solution for C=0 resp. C=1
        Ss = np.linspace(self.Scw,1.0-self.Sorw,self.nSw)
        fws  = self.fw(Ss,C=0)
        dfws = self.dfw_S(Ss,C=0)
        self.ffC0 = pd.DataFrame(index=Ss, data={'fw':fws,'dfw':dfws})
        fws  = self.fw(Ss,C=1)
        dfws = self.dfw_S(Ss,C=1)
        self.ffC1 = pd.DataFrame(index=Ss, data={'fw':fws,'dfw':dfws})

        # The general solution of the polymer fractional flow problem consists of the following parts:
        #  1) S-wave rarefaction (C=1) from (Sw=1-Sorw,C=1) to (Sw=Sw1,C=1)
        #  2) C-wave shock from (Sw=Sw1,C=1) to (Sw=Sw2,C=0)
        #  3) S-wave (C=0) from (Sw=Sw2,C=0) to (Sw=Swi,C=0). This part depends on Swi and can be of three types:
        #     a) S-wave shock (C=0)
        #     b) S-wave rarefaction (C=0)
        #     c) S-wave rarefaction (C=0) followed by S-wave shock. Transition from rarefaction to
        #        shock takes place in Sw=Sw3
        # However, for large Swi (Swi > Sw2_alt) the solution is different:
        #  1) S-wave rarefaction (C=1) from (Sw=1-Sorw,C=1) to (Sw=Sw1_hi,C=1)
        #  2) C-wave shock from (Sw=Sw1_hi,C=1) to (Sw=Swi,C=0)
        #
        # The following steps are performed to determine the parts of the solution:
        #  A) Determine the intersection of the S-wave integral curve through (Sw=1-Sorw,C=1) and a C-wave Hugoniot locus, such that
        #     the speeds at the intersection point match: Sw = Sw1.
        #     This is the starting point of the C-wave shock. Note that the shock speed matches the
        #     front speed of the preceding S-wave rarefaction (C=1)
        #  B) Determine the two potential 'landing' points of the C-wave shock by solving the Rankine-Hugoniot condition for
        #     a C-wave shock from (Sw=Sw1,C=1) to (Sw=*,C=0):
        #     1) Sw2 (<=Sw1) is the endpoint of the 'admissible' shock.
        #     2) Sw2_alt (>Sw1) is not 'admissible' because of speed requirements. It is used to distinguish between
        #        two solution branches which follow the C-wave shock (see below)
        #  C) Determine the solution after the C-wave shock. There are two potential scenarios depending on Swi:
        #     1) If Swi <= Sw2_alt, lowSwi = True, the traditional 'tangent' solution a la Buckley Leverett is found.
        #        There are three potential scenarios depending on Swi:
        #          a) S-wave shock (C=0)
        #          b) S-wave rarefaction (C=0)
        #          c) S-wave rarefaction (C=0) followed by S-wave shock. Transition from rarefaction to
        #             shock takes place in Sw=Sw3
        #     2) If Swi > Sw2_alt, lowSwi = False, the solution shocks directly from the S-wave C=1 rarefaction via a C-wave
        #        shock to (Sw=Swi,C=0): (Sw=Sw1_hi,C=1) -> (Sw=Swi,C=0). To find Sw1_hi, the Rankine-Hugoniot condition is solved
        #        for a C-wave shock from (Sw=Sw1_hi,C=1) to (Sw=Swi,C=0)
        #
        # For no>1, the scheme described above always holds. However, for no=1 the intersection point Sw1 does not always exist,
        # because dfw/dSw(Sw=1-Sorw,C=1) is not equal to zero. In this case, the solution is as follows:
        #  1) C-wave shock from (Sw=1-Sorw,C=1) to (Sw=Sw2,C=0). Sw2 is determined as described above.
        #  2) S-wave (C=0) from (Sw=Sw2,C=0) to (Sw=Swi,C=0). This part depends on Swi and can be of three types:
        #     a) S-wave shock (C=0)
        #     b) S-wave rarefaction (C=0)
        #     c) S-wave rarefaction (C=0) followed by S-wave shock. Transition from rarefaction to
        #        shock takes place in Sw=Sw3
        #
        if self.no > 1:
            # Step A
            # Determine the intersection of:
            #  1) the S-wave integral curve through (Sw=1-Sorw,C=1) and a C-wave Hugoniot locus, such that
            #     the speeds at the intersection point match: Sw = Sw1.
            #  2) the S-wave integral curve through (Sw=1-Sorw,C=1) and a C-wave integral curve, such that
            #     the speeds at the intersection point match Sw = Sw1_IC.
            #     Sw1_IC is not used in the fractional flow solution
            #
            # eps_Sw1 is introduced to avoid the spurious solution Sw=Scw
            self.Sw1    = brentq(lambda Sw: self.lamS_shockC(Sw,epsAds=1e-8),self.Scw+self.eps_Sw1,1.0-self.Sorw,xtol=self.xtol)
            self.Sw1_IC = brentq(lambda Sw: self.lamS_eq_lamC(Sw, C=1.0),    self.Scw+self.eps_Sw1,1.0-self.Sorw,xtol=self.xtol)
            # The flag S_rare_C1 is used to determine if the S-wave rarefaction (C=1) should be included in the solution
            self.S_rare_C1 = True
            if self.debug:
                print(f'no>1 branch: Sw1  = {self.Sw1:.8e}')
                print(f'no>1 branch: 1-S1 = {(1-(self.Sw1-self.Scw)/(1-self.Scw-self.Sorw)):.16e}')

            # Step B (part1)
            # Determine Sw2 (Sw2<=Sw1), the endpoint of the 'admissible' shock.
            #
            # Determine Sw2 by solving the Rankine-Hugoniot condition for a C-wave shock from (Sw=Sw1,C=1) to (Sw=Sw2,C=0) with
            # the boundary condition that Sw2 <= Sw1
            self.Sw2 = brentq(lambda Sw: self.hugo(Sw,C=0,Sst=self.Sw1,Cst=1.0,epsAds=self.epsAds), self.Scw, self.Sw1,xtol=self.xtol)
            if self.debug:
                print(f'no>1 branch: Sw2  = {self.Sw2:.8e}')
                print(f'no>1 branch: 1-S2 = {(1-(self.Sw2-self.Scw)/(1-self.Scw-self.Sorw)):.8e}')

            # Step C
            # For no>1, Sw1 should be smaller than 1-Sorw, hence also Sw2 should be smaller than 1-Sorw (Sw2<=Sw1).
            # However, due to numerical inaccuracies, Sw1 and Sw2 can become rounded off to 1-Sorw which can give
            # problems for step C. Hence this if block.
            if self.Sw2 < 1.0-self.Sorw:
                # For Swi = Scw we always have lowSwi = True. So, once Sw2 is found, determine the solution after the C-shock wave.
                # Three potential scenarios depending on Swi:
                #  1) S-wave shock (C=0)
                #  2) S-wave rarefaction (C=0)
                #  3) S-wave rarefaction (C=0) followed by S-wave shock. Transition from rarefaction to
                #     shock takes place in Sw=Sw3
                #
                #  'events' contains the solution type: 'shock', 'rare' or 'shock+rare'.
                self.Sw3, self.events  = self.events_Swave(self.Sw2,self.Swi,eps_Sw3=self.eps_Sw3,epsSvel=self.epsSvel,xtol=self.xtol)
            else:
                # Problem occurs if Sw2=1-Sorw. In this case, for no>1 we have lamS(Sw2=1-Sorw,C=0) = 0. Note that lamS(Sw2,C=0) is the
                # speed of the back of the S-wave (C=0) rarefaction which would be smaller than the speed of the preceding C-wave shock.
                # Therefore, from speed considerations only a S-wave shock solution (C=0) is allowed after the C-wave shock.
                self.Sw3 = self.Scw
                self.events = 'shock'
            if self.debug:
                print('no>1 branch: events =', self.events)
                print(f'no>1 branch: Sw3 = {self.Sw3:.8e}') 
                
        elif self.no == 1:
            # Step A
            # Determine the intersection of the S-wave integral curve through (Sw=1-Sorw,C=1) and a C-wave Hugoniot locus, such that
            # the speeds at the intersection point match: Sw = Sw1.
            #
            # For no=1 there is not always such an intersection, in particular for larger values of Mob(C=1). If there is no intersection,
            # Sw1 is set to 1-Sorw and the flag S_rare_C1 is set to False.
            try:
                # eps_Sw1 is introduced to avoid the spurious solution Sw=Scw
                self.Sw1 = brentq(lambda Sw: self.lamS_shockC(Sw,epsAds=self.epsAds),self.Scw+self.eps_Sw1,1.0-self.Sorw,xtol=self.xtol)
                self.S_rare_C1 = True
                if self.debug:
                    print(f'no=1 branch: Sw1  = {self.Sw1:.8e}')
                    print(f'no=1 branch: 1-S1 = {(1-(self.Sw1-self.Scw)/(1-self.Scw-self.Sorw)):.8e}')
            except ValueError as e:
                self.Sw1 = 1.0-self.Sorw
                self.S_rare_C1 = False
                if self.debug:
                    print(f'no=1 branch: Sw1 = {self.Sw1:.8e}, Error from brentq: ', e)

            if self.S_rare_C1:
                # If Sw1 is found, steps B & C are performed
                if debug:
                    print('no=1 branch: S_rareC_1 =', self.S_rare_C1)

                # Step B (part 1)
                # Determine Sw2 (Sw2<=Sw1), the endpoint of the 'admissible' shock.
                #
                # Determine Sw2 by solving the Rankine-Hugoniot condition for a C-wave shock from (Sw=Sw1,C=1) to (Sw=Sw2,C=0) with
                # the boundary condition that Sw2 <= Sw1
                self.Sw2 = brentq(lambda Sw: self.hugo(Sw,C=0,Sst=self.Sw1,Cst=1.0,epsAds=self.epsAds), self.Scw, self.Sw1,xtol=self.xtol)
                if self.debug:
                    print(f'no=1 branch: Sw2  = {self.Sw2:.8e}')
                    print(f'no=1 branch: 1-S2 = {(1-(self.Sw2-self.Scw)/(1-self.Scw-self.Sorw)):.8e}')

                # Step C
                # Contrary to the no>1 case, lamS(Sw2=1-Sorw,C=0) is not zero. Therefore, no need to force a S-wave shock solution for Sw2=1-Sorw
                #
                # For Swi = Scw we always have lowSwi = True. So, once Sw2 is found, determine the solution after the C-shock wave.
                # Three potential scenarios depending on Swi:
                #  1) S-wave shock (C=0)
                #  2) S-wave rarefaction (C=0)
                #  3) S-wave rarefaction (C=0) followed by S-wave shock. Transition from rarefaction to
                #     shock takes place in Sw=Sw3
                #
                #  'events' contains the solution type: 'shock', 'rare' or 'shock+rare'.
                self.Sw3, self.events  = self.events_Swave(self.Sw2,self.Swi,eps_Sw3=self.eps_Sw3,epsSvel=self.epsSvel,xtol=self.xtol)
                if self.debug:
                    print('no=1 branch: events =', self.events)
                    print(f'no=1 branch: Sw3 = {self.Sw3:.8e}') 
            else:
                # If no solution for Sw1 was found, there is no initial S-wve rarefaction (C=1) and the solution starts with a C-wave shock
                # from (Sw=1-Sorw,C=1) to (Sw=Sw2,C=0) (like step B). The solution after the C-wave shock is determined as in the no>1 case (step C).
                if debug:
                    print('no=1 branch: S_rareC_1 =', self.S_rare_C1)
                #
                # Step B (part 1)
                # Determine Sw2, the endpoint of the 'admissible' shock.
                #
                # Determine Sw2 by solving the Rankine-Hugoniot condition for a C-wave shock from (Sw=1-Sorw,C=1) to (Sw=Sw2,C=0) with
                # the boundary condition that Sw2 <= 1-Sorw
                try:
                    # eps_Sw2 is introduced to avoid the spurious solution Sw=1-Sorw
                    self.Sw2 =  brentq(lambda Sw: self.hugo(Sw,C=0.0,Sst=(1.0-self.Sorw),Cst=1.0,epsAds=self.epsAds), self.Scw, 1.0-self.Sorw-self.eps_Sw2)
                except ValueError as e:
                    self.Sw2 = 1.0 - self.Sorw
                if self.debug:
                    print(f'no=1 branch: Sw2  = {self.Sw2:.8e}')
                    print(f'no=1 branch: 1-S2 = {(1-(self.Sw2-self.Scw)/(1-self.Scw-self.Sorw)):.8e}')
                #
                # Step C
                # Three potential scenarios depending on Swi:
                #  1) S-wave shock (C=0)
                #  2) S-wave rarefaction (C=0)
                #  3) S-wave rarefaction (C=0) followed by S-wave shock. Transition from rarefaction to
                #     shock takes place in Sw=Sw3
                #
                #  'events' contains the solution type: 'shock', 'rare' or 'shock+rare'.
                self.Sw3, self.events  = self.events_Swave(self.Sw2,self.Swi,eps_Sw3=self.eps_Sw3,epsSvel=self.epsSvel,xtol=self.xtol)
                if self.debug:
                    print('no=1 branch: events =', self.events)
                    print(f'no=1 branch: Sw3 = {self.Sw3:.8e}') 

        # Step B (part2)
        # Determine the 'non-admissible landing' point, Sw2_alt, of the C-wave shock by solving the Rankine-Hugoniot condition for
        # a C-wave shock from (Sw=Sw1,C=1) to (Sw=*,C=0):
        #   Sw2_alt (>Sw1) is not 'admissible' because of speed requirements. It is used to distinguish between
        #   two solution branches which follow the C-wave shock
        try:
            # Determine Sw2_alt by solving the Rankine-Hugoniot condition for a C-wave shock from (Sw=Sw1,C=1) to (Sw=Sw2,C=0) with
            # the boundary condition that Sw2 > Sw1
            self.Sw2_alt = brentq(lambda Sw: self.hugo(Sw,C=0,Sst=self.Sw1,Cst=1.0,epsAds=self.epsAds), self.Sw1, 1.0-self.Sorw,xtol=self.xtol)
            if self.debug:
                print(f'Sw2_alt = {self.Sw2_alt:.8e}')
        except ValueError as e:
            self.Sw2_alt = 1.0-self.Sorw
            if self.debug:
                print(f'Sw2_alt = {self.Sw2_alt:.8e}, Error from brentq: ', e)

    def set_Swi(self, SwiNew):
        '''
        set_Swi is used to change the initial water saturation in an instance of FracFlow.
        It will trigger the re-calculation of the fraction flow solution.
        '''
        if (SwiNew < self.Scw) or (SwiNew >= 1.0-self.Sorw):
            raise ValueError(f'Swi must be in the range [{self.Scw},{1.0-self.Sorw})')
        # Update the initial water saturation
        self.Swi = SwiNew
        #
        if (self.Swi < self.Sw2_alt):
            # Determine the solution after the C-wave shock to (Sw=Sw2,C=0). Three potential scenarios depending on Swi:
            #  1) S-wave shock (C=0)
            #  2) S-wave rarefaction (C=0)
            #  3) S-wave rarefaction (C=0) followed by S-wave shock. Transition from rarefaction to
            #     shock takes place in Sw=Sw3
            #
            # self.events contains the solution type: 'shock', 'rare' or 'shock+rare'.
            self.Sw3, self.events  = self.events_Swave(self.Sw2,self.Swi,eps_Sw3=self.eps_Sw3,epsSvel=self.epsSvel,xtol=self.xtol)
            # lowSwi is a flag to keep track which branch of the fractional flow
            # solution should be used. For Swi <= Sw2_alt it is True, otherwise False
            self.lowSwi = True
            if self.debug:
                print(f'set_Swi: Sw3 = {self.Sw3:.8e}')
                print('set_Swi: lowSwi = ', self.lowSwi)
        else:
            # 'High Swi' solution: S-wave rarefaction (C=1) + C-wave shock to (Sw=Swi,C=0)
            # The shock is from (Sw=Sw1_hi,C=1) to (Sw=Swi,C=0). Note that the shock speed is higher
            # than the front speed of the S-wave rarefaction
            #
            # To find Sw1_hi, the Rankine-Hugoniot condition is solved for a C-wave shock
            # from (Sw=Sw1_hi,C=1) to (Sw=Swi,C=0)
            try:
                self.Sw1_hi = brentq(lambda Sw: self.hugo(Sw,C=1.0,Sst=self.Swi,Cst=0.0), self.Sw1, self.Swi,xtol=self.xtol)
                if self.debug:
                    print(f'set_Swi: Sw1_hi = {self.Sw1_hi:.8e}')
            except ValueError as e:
                # if Swi is close to Sw2_alt, numerical inaccuracies can prevent finding Sw1_hi
                self.Sw1_hi = self.Sw1
                if self.debug:
                    print(f'set_Swi: Sw3 = {self.Sw3:.8e}, Error from brentq: ', e)
            # # Flag that the solution is of the high Swi type
            self.lowSwi = False
            if self.debug:
                print('set_Swi: lowSwi = ', self.lowSwi)
        return self.Swi, self.lowSwi

    def Mob(self,C):
        '''
        The function Mob calculates the ratio (oil mobility / water mobility) as a function of
        the normalized polymer concentration, C. Note that this is the inverse of the standard 
        mobility ratio = displacing fluid mobility / displaced fluid mobility.

        C = Normalized polymer concentration = polymer concentration / injection concentration.

        A linear relation between polymer solution viscosity and normalized polymer concentration is
        assumed. 
        '''
        return (1.0/self.MobWat)*(1+self.delMu*C)

    def dMob_C(self,C):
        '''
        The function dMob_C calculates the derivative of the mobility ratio wrt of the normalized 
        polymer concentration.
        '''
        return (1.0/self.MobWat)*self.delMu
    
    def S_D(self,Sw):
        '''
        The function S_D returns the normalized water saturation = (Sw-Scw) / (1-Scw-Sorw).

        The normalized water saturation is used in the calculation of the fractional
        flow function and its derivatives
        '''
        return (Sw-self.Scw) / (1.0-self.Scw-self.Sorw)

    def fw(self,Sw,C):
        '''
        The function fw returns the fractional flow value = q_wat / (q_wat + q_oil) as function
        of the water saturation Sw and normalized polymer concentration C.
        '''
        S = self.S_D(Sw)
        return S**self.nw/(S**self.nw + self.Mob(C)*(1-S)**self.no)

    def dfw_S(self,Sw,C):
        '''
        The function dfw_S returns the derivative of the fractional flow function wrt Sw.
        It depends on the water saturation Sw and normalized polymer concentration C.
        '''
        S = self.S_D(Sw)
        num = self.Mob(C) * S**(self.nw-1) * (1-S)**(self.no-1) * (self.nw*(1-S)+self.no*S)
        num = num / (1.0-self.Scw-self.Sorw)
        denom = ( S**self.nw + self.Mob(C)*(1-S)**self.no )**2
        return num / denom

    def dfw_C(self,Sw,C):
        '''
        The function dfw_C returns the derivative of the fractional flow function wrt C.
        It depends ont the water saturation Sw and normalized polymer concentration C.
        '''
        S = self.S_D(Sw)
        num = -self.dMob_C(C) * S**self.nw * (1-S)**self.no
        denom = ( S**self.nw + self.Mob(C)*(1-S)**self.no )**2
        return num / denom

    def dfw_S2(self,Sw,C):
        '''
        The function dfw_S2 returns the second derivative of the fractional flow function wrt Sw.
        It depends on the water saturation Sw and normalized polymer concentration C.

        This function is used to test convexity / concavity of the fractional flow curve in 
        the function events_Swave.
        It can also be used to calculate the S-wave rarefaction solution as an alternative to using
        the lookup method.
        The ODE to be solved is: dS/dksi = 1 / dfw_S2(Sw(ksi),C=constant)
        '''
        S = self.S_D(Sw)
        num1 = self.Mob(C)*(1-S)**self.no * (self.nw**2*(1-S)**2 + self.no*(1+self.no)*S**2 + self.nw*(1-S)*(-1+(1+2*self.no)*S))
        num2 = S**self.nw * (self.nw**2*(1-S)**2 + self.no*(self.no-1)*S**2 + self.nw*(1-S)*(1+(2*self.no-1)*S))
        num = self.Mob(C) * (1-S)**(self.no-2) * S**(self.nw-2) * (num1 - num2)
        num = num / (1.0-self.Scw-self.Sorw)**2
        denom = ( S**self.nw + self.Mob(C)*(1-S)**self.no )**3
        return num / denom

    def ad(self,C):
        '''
        The function ad returns the Langmuir isotherm a function of the normalized polymer concentration C
        '''
        return self.ad1*C / (1+self.ad2*C)

    def dad_C(self,C):
        '''
        The function dad_C returns the derivative of the Langmuir isotherm wrt C.
        '''
        return self.ad1 / (1+self.ad2*C)**2

    def dad_C2(self,C):
        '''
        The function dad_C2 returns the second derivative of the Langmuir isotherm wrt C.

        It is currently not used in the fractional flow solution
        '''
        return -2*self.ad1*self.ad2 / (1+self.ad2*C)**3
    

    def lamS(self,Sw,C):
        '''
        The function lamS returns the S-wave eigenvalue of the polymer conservation equation as
        a function of the water saturation Sw and normalized polymer concentration C.

        It corresponds to the speed of the rarefaction wave on the S-wave integral curve.
        '''
        return self.dfw_S(Sw,C)

    def lamC(self,Sw,C):
        '''
        The function lamC returns the C-wave eigenvalue of the polymer conservation equation as
        a function of the water saturation Sw and normalized polymer concentration C.

        It corresponds to the speed of the rarefaction wave on the C-wave integral curve.
        '''
        return self.fw(Sw,C) / (Sw + self.dad_C(C))
    
    def dS_dC(self,Sw,C):
        '''
        The function dS_dC calculates the RHS of the ODE dS/dC = dS_dC for the C-wave integral curve.
        It is a function of the water saturation Sw and normalized polymer concentration C.

        This function is not used in the fractional flow solution, but is used in the detailed
        analysis of the solution construction.
        '''
        S = self.S_D(Sw)
        D = S**self.nw+self.Mob(C)*(1-S)**self.no
        num = -self.dMob_C(C) * S * (1-S)**self.no * (Sw+self.dad_C(C))
        denom = S*D - self.Mob(C)/(1.0-self.Scw-self.Sorw)*(1-S)**(self.no-1)*(self.nw*(1-S)+self.no*S)*(Sw+self.dad_C(C))
        return num/denom

    def dC_dS(self,C,Sw):
        '''
        The function dC_dS calculates the RHS of the ODE dC/dS = dC_dS for the C-wave integral curve.
        It is a function of the water saturation Sw and normalized polymer concentration C.

        This function is not used in the fractional flow solution, but is used in the detailed
        analysis of the solution construction.
        '''
        S = self.S_D(Sw)
        D = S**self.nw+self.Mob(C)*(1-S)**self.no
        num = S*D - self.Mob(C)/(1.0-self.Scw-self.Sorw)*(1-S)**(self.no-1)*(self.nw*(1-S)+self.no*S)*(Sw+self.dad_C(C))
        denom = -self.dMob_C(C) * S * (1-S)**self.no * (Sw+self.dad_C(C))
        return num/denom
    
    def IC_ODE(self, x,ksi):
        '''
        The function IC_ODE sets up a set of coupled ODE's for the C-wave integral curve, viz.:
            dSw/dksi = dS_dksi
            dC/dksi  = dC_dksi

        It can, for example, be solved using the SciPy routine scipy.integrate.odeint:
            x0  = [Sw0,C0]
            ksis = np.linspace(0,ksi_max, nksi)
            Sw, C = odeint(ff.IC_ODE, x0, ksis)
        where ff is an instance of FracFlow, x0 is a point on the C-wave integral curve and ksis
        is a range of ksi values for which the solution is sought.

        This function is not used in the fractional flow solution, but is used in the detailed
        analysis of the solution construction.
        '''
        S = x[0]
        C = x[1]

        dS = self.dS_dksi(S,C)
        dC = self.dC_dksi(S,C)

        return [dS, dC]
    
    def dS_dksi(self, Sw,C):
        '''
        The function dS_dksi calculates the derivative dSw/dksi as input to the coupled ODE's
        used in IC_ODE for determining the C-wave integral curve.
        '''
        S = (Sw-self.Scw)/(1.0-self.Scw-self.Sorw)
        return -self.dMob_C(C) * S * (1-S)**self.no * (Sw+self.dad_C(C))

    def dC_dksi(self, Sw,C):
        '''
        The function dC_dksi calculated the derivative dC/dksi as input to the coupled ODE's
        used in IC_ODE for determining the C-wave integral curve.
        '''
        scale = (1.0-self.Scw-self.Sorw)
        S = (Sw-self.Scw)/scale
        D = (S**self.nw + self.Mob(C)*(1-S)**self.no)
        return S*D - self.Mob(C)/scale*(1-S)**(self.no-1)*(self.nw*(1-S)+self.no*S)*(Sw+self.dad_C(C))
  
    def hugo(self, Sw, C, Sst, Cst, epsAds=1e-8):
        '''
        The function hugo is used to determine the C-wave Hugoniot locus of the state (Sw=Sst,C=Cst).

        Given C, Sst and Cst solving the equation:
                hugo(Sw, C, Sst, Cst) = 0
        finds Sw, so that the state (Sw,C) is on the Hugoniot locus of the state (Sw=Sst,C=Cst).

        It is used to solve the C-wave Rankine-Hugoniot condition:
            fw(Sst,Cst) / (Sst + (ad(Cst)-ad(C)/(Cst-C)) = fw(S,C) / (S + (ad(Cst)-ad(C)/(Cst-C))

        See function Dads for the parameter epsAds
        '''
        part1 = self.fw(Sw,C) / (Sw + self.Dads(C,Cst,epsAds))
        part2 = self.fw(Sst,Cst) / (Sst + self.Dads(C,Cst,epsAds))
        return part1 - part2

    def shockS_vel(self, Sw,Swst,Cst,epsSvel=1e-8):
        '''
        The function shockS_vel calculates the speed of the S-wave shock between states (Sw,Cst) and
        (Swst,Cst). Note that S-wave shocks do not change C.

        The parameter epsSvel is used to avoid having a singularity when Sw = Swst by returning
        the derivative value dfw/dSw(Swst,Cst) if (Sw-Swst) < epsSvel.

        The function is able to handle both a float and pandas series / numpy array as input
        for the parameter Sw.
        '''
        if isinstance(Sw,(pd.core.series.Series,np.ndarray)):
            select = (np.abs(Sw-Swst)<epsSvel)
            result = np.empty_like(Sw)
            result[select] = self.dfw_S(Swst,Cst)
            result[~select] = (self.fw(Sw[~select],Cst)-self.fw(Swst,Cst))/(Sw[~select]-Swst)
            return result
        else:
            if np.abs(Sw-Swst)<epsSvel:
                return self.dfw_S(Swst,Cst)
            else:
                return (self.fw(Sw,Cst)-self.fw(Swst,Cst))/(Sw-Swst)
        
    def Dads(self, C, Cst,epsAds=1e-8):
        '''
        The function Dads evaluates the following expression:
              (ad(C) - ad(Cst)) / (C - Cst) 

        The parameter epsAds is used to avoid having a singularity when C = Cst by returning
        the derivative value da/dC(Cst) if (C-Cst).

        The function is able to handle both a float and pandas series / numpy array as input
        for the parameter C.
        '''
        if isinstance(C, (pd.Series, np.ndarray)):
            select = (np.abs(C - Cst) < epsAds)
            result = np.empty_like(C)
            result[select] = self.dad_C(Cst)
            result[~select] = (self.ad(C[~select]) - self.ad(Cst)) / (C[~select] - Cst)
            return result
        else:
            if np.abs(C - Cst) < epsAds:
                return self.dad_C(C)
            else:
                return (self.ad(C) - self.ad(Cst)) / (C - Cst)
        
    def shockC_vel(self, C,Swst,Cst,epsAds=1e-8):
        '''
        The function shockC_vel returns the speed of the C-wave shock between states (Sw,C) and (Swst,Cst).

        Note that this function assumes that (Sw,C) is on the Hugoniot locus of (Swst,Cst) and therefore 
        does not require Sw to be input.

        See function Dads for the parameter epsAds
        '''
        return self.fw(Swst,Cst)/(Swst+self.Dads(C,Cst,epsAds))
    
    def lamS_eq_lamC(self,Sw,C):
        '''
        The function lamS_eq_lamC is used to determine the intersection of the S-wave integral curve
        for the specified C and a C-wave integral curve, such that the speeds match at the intersection point, i.e.
                lamS(Sw, C) = lamC(Sw,C)
        '''
        return self.lamS(Sw,C) - self.lamC(Sw,C)
    
    
    def lamS_shockC(self, Sw,epsAds=1e-8):
        '''
        The function lamS_shockC is used to determine the intersection of the S-wave integral curve C=1 and a
        C-wave Hugoniot locus going through a state with C=0, such that the front speed of the S-wave rarefaction
        matches the speed of the shock from (Sw,C=1) to (*,C=0), i.e.
                lamS(Sw,C=1) = fw(Sw,C=1)/(Sw + (ad(C=1)-ad(C=0))/(1-0)) =shockC_vel(C=0,Swst=Sw,Cst=1)

        See function Dads for the parameter epsAds
        '''
        return self.lamS(Sw,1) - self.shockC_vel(C=0,Swst=Sw,Cst=1,epsAds=epsAds)
    

    def lamS_shockS(self, Sw,Swi,C,epsSvel=1e-8):
        '''
        The function lamS_shockS is used to determine the value for Sw
        where the S-wave rarefaction speed matches the speed of the S-wave shock
        between the states (Sw,C) and (Swi,C).

        This function is used in the function events_Swave

        See function shockS_vel for the parameter epsSvel
        '''
        return self.dfw_S(Sw,C) - self.shockS_vel(Sw, Swi, C, epsSvel)

    def events_Swave(self,Sw2,Swi,eps_Sw3=1e-8, epsSvel=1e-8, xtol=1e-15):
        '''
        This function determines how the S-wave solution between (Sw=Sw2,C=0) and (Sw=Swi,C=0) looks like.
        (Sw2,C=0) is the endpoint of the C-wave shock that connects the C=1 S-wave rarefaction to the
        C=0 S-wave solution.
        Three possible scenarios depending on the Swi value:
           1) S-wave shock: 'events' = 'shock'
           2) S-wave rarefaction: 'events' = 'rare'
           3) S-wave rarefaction + S-wave shock: 'events' = 'rare + shock'. Transition from rarefaction to shock
              takes place at Sw=Sw3
        This is equivalent to the Buckley Leverett solution.

        Note this function only works for a C=0 S-wave.

        The function returns the scenario corresponding to the Swi value in 'events'. It also calculates
        an Sw3 value, which is only relevant in the 'rare + shock' scenario.

        The parameter 'eps_Sw3' is used to avoid a 'rare + shock' solution with a very small
        rarefaction part. It is also used to avoid the spurious solution Sw=Swi.
        See function shockS_vel for the parameter epsSvel
        The parameter xtol is used in the brentq solver.
        '''

        # Set C value to 0, as this function is used for the C=0 S-wave
        C = 0

        if (self.no==1) and (self.nw==1) and (self.MobWat==1):
            # Special case of linear fractional flow
            # Contact discontinuity modelled as a shock
            Sw3 = Swi
            events = 'shock'
        else:
            # ept_Sw3 is introduced to avoid the spurious solution Sw=Swi and to avoid a 'rare + shock' solution
            # with a very small rarefaction part
            if Swi < Sw2:
                Sw_lo = Swi + eps_Sw3
                Sw_hi = Sw2 - eps_Sw3
            else:
                Sw_lo = Sw2 + eps_Sw3
                Sw_hi = Swi - eps_Sw3
            try:
                Sw3 = brentq(lambda Sw: self.lamS_shockS(Sw,Swi=Swi,C=C,epsSvel=epsSvel),Sw_lo,Sw_hi,xtol=xtol)
                events = 'rare + shock'
                if self.debug:
                    print(f'events_Swave: Sw3 = {Sw3:.8e}, events = {events}')
            except ValueError as e:
                Sw3 = Swi      
                # Second derivative of fw is used to check concavity (Swi<Sw2) or convexity (Swi>=Sw2)
                if ((Swi < Sw2 ) and (self.dfw_S2(0.5*(Swi+Sw2),0)>=0)) or ((Swi >= Sw2) and (self.dfw_S2(0.5*(Swi+Sw2),0)<=0)):
                    events = 'shock'
                else:
                    events = 'rare'
                if self.debug:
                    print(f'events_Swave: Sw3 = {Sw3:.8e}, events = {events}, Error from brentq: ', e)   
        return Sw3, events

    def rare_Swave(self, ksi, Czero, Sl, Sr):
        '''
        This function returns the S-wave rarefaction solution Sw(ksi) for ksi = xD/tD where
        xD is the dimensionless position at dimensionless time tD.
        The flag Czero selects between the C=0 (Czero=True) and C=1 (Czero=False)
        S-wave rarefaction solution.
        Sl and Sr are the saturation boundaries of the rarefaction wave, where Sl is
        the slower saturation.

        The solution is determined by inverting the relation for the S-wave rarefaction:
                ksi = x/t = dfw/dSw(Sw(ksi), C=0 or 1)
        using a lookup in a table of (Sw,dfw/dSw) values.
        '''
        # Select the correct look up table depending of the Czero flag
        if Czero:
            fracflow = self.ffC0
        else:
            fracflow = self.ffC1
        # To ensure that the lookup is uniquely determined we restrict the
        # saturations in the table to be between Sr and Sl.
        if Sr < Sl:
            smin = take_closest(fracflow.index,Sr)
            smax = take_closest(fracflow.index,Sl)
        else: 
            smin = take_closest(fracflow.index,Sl)
            smax = take_closest(fracflow.index,Sr)        
        data = fracflow[smin:smax]
        # Change the table index (contains Sw values) to a table column with name 'Sw'
        data = data.reset_index(names=['Sw'])
        # Perform the lookup: given ksi=dfw, find Sw
        Sw = interpolate(ksi, data.sort_values(by='dfw'), 'dfw', 'Sw')
        return Sw
    

    # ****************************************************************************************
    # Functions for post processing and visualising the  polymer results

    def calc_Sol(self,x,t):
        '''
        The function calc_Sol calculates the water saturation Sw at dimensionless position xD=x and
        dimensionless time tD=t for an instance of FracFlow.
        '''
        if self.lowSwi:
            # This branch is for Swi <= Sw2_alt
            # The solution consists of:
            #   1) S-wave rarefaction (C=1) from Swinj to Sw1: is skipped if 'S_rare_C1' is False in which case
            #      Sw1 is the same as Swinj
            #   2) C-wave shock from (Sw=Sw1,C=1) to (Sw=Sw2,C=0). Note that the speed of the S-wave 
            #     rarefactin (C=1) above (if present) matches the speed of the C-wave shock.
            #   3) One of the following scenarios depending on Swi
            #       a) S-wave rarefaction (C=0) from Sw2 to Swi
            #       b) S-wave shock (C=0) from Sw2 to Swi
            #       c) S-ware rarefaction (C=0) from Sw2 to Sw3, followed by S-wave shock (C=0) from Sw3 to Swi
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
                elif ksi <= self.shockC_vel(C=1,Swst=self.Sw2,Cst=0):
                    # This part is at the back end of the C-wave shock from (Sw=Sw1,C=1) to (Sw=Sw2,C=0)
                    if self.S_rare_C1:
                        # S-wave rarefaction C=1, runs from Swinj to Sw1
                        # Note that the speed of the front of the S-wave raraefaction is the same as the
                        # speed of the C-wave shock from (Sw=Sw1,C=1) to (Sw=Sw2,C=0)
                        # The C-wave shock takes place at the front of the S-wave rarefaction (C=1)
                        return self.rare_Swave(ksi, Czero=False, Sl=self.Swinj, Sr=self.Sw1)
                    else:
                        # No S-wave rarefaction (C=1) is present
                        # Back end of the shock from Swinj to Sw2
                        return self.Swinj
                else:
                    if self.events == 'rare':
                        if ksi <= self.lamS(self.Sw2,C=0):
                            # Sw2 up to start of S-wave rarefaction (C=0)
                            return self.Sw2
                        elif ksi <= self.lamS(self.Swi,C=0):
                            # S-wave rarefaction wave (C=0) from Sw2 to Swi
                            return self.rare_Swave(ksi, Czero=True, Sl=self.Sw2, Sr=self.Swi)
                        else:
                            return self.Swi
                    elif self.events == 'shock':
                        if ksi <= self.shockS_vel(self.Sw2,self.Swi,Cst=0):
                            # Sw2 upto the S-wave shock (C=0) from Sw2 to Swi occurs
                            return self.Sw2
                        else:
                            return self.Swi
                    else: # self.events == 'rare + shock'
                        if ksi <= self.lamS(self.Sw2,C=0):
                            # Sw2 up to start of S-wave rarefaction (C=0)
                            return self.Sw2
                        elif ksi <= self.lamS(self.Sw3,C=0):
                            # S-wave rarefaction wave (C=0) from Sw2 to Sw3
                            # Note that the front of this rarefaction wave has the same speed
                            # as the shock from Sw3 to Swi, i.e lamS(Sw3,C=0) = shockS_vel(Sw3,Swi,Cst=0)
                            return self.rare_Swave(ksi, Czero=True, Sl=self.Sw2, Sr=self.Sw3)
                        else:
                            return self.Swi
        else:
            # This branch is for Swi > Sw2_alt
            # The solution consists of:
            #   1) S-wave rarefaction (C=1) from Swinj to Sw1_hi
            #   2) C-wave shock from (Sw=Sw1_hi,C=1) to (Sw=Swi,C=0). Note that the C-wave shock
            #      is faster than the front of the S-wave rarefaction.
            if t<=0:
                if x<=0:
                    return self.Swinj
                else:
                    return self.Swi
            else:
                ksi =x/t
                if ksi <= self.lamS(self.Swinj,C=1):
                    # Note self.lam1(self.Sinj,C=1) = 0, so Swinj is stationary
                    return self.Swinj
                elif ksi <= self.lamS(self.Sw1_hi,C=1):
                    # S-wave rarefaction C=1, runs from Swinj to Sw1_hi
                    return self.rare_Swave(ksi, Czero=False, Sl=self.Swinj, Sr=self.Sw1_hi)
                elif ksi <= self.shockC_vel(C=1,Swst=self.Swi,Cst=0):
                    # Sw1_hi upto the C-wave shock from (Sw=Sw1_hi,C=1) to (Sw=Swi,C=0) occurs
                    # Note that the polymer front moves with the same speed as this C-wave shock.
                    return self.Sw1_hi
                else:
                    return self.Swi

    def calc_SolC(self,x,t):
        '''
        The function calc_SolC calculates the normalized polymer concentration C at dimensionless
        position xD=x and dimensionless time tD=t for an instance of FracFlow.
        '''
        if self.lowSwi:
            # This branch is for Swi <= Sw2_alt
            # The normalized polymer concentrion shocks from 1 to 0 at the C-wave shock from
            # (Sw=Sw1,C=1) to (Sw=Sw2,C=0).
            if t<=0:
                if x<=0:
                    return 1
                else:
                    return 0
            else:
                ksi=x/t
                if ksi <= self.shockC_vel(C=1,Swst=self.Sw2,Cst=0):
                    return 1
                else:
                    return 0
        else:
            # This branch is for Swi > Sw2_alt
            # The normalized polymer concentrion shocks from 1 to 0 at the C-wave shock from
            # (Sw=Sw1_hi,C=1) to (Sw=Swi,C=0).
            if t<=0:
                if x<=0:
                    return 1
                else:
                    return 0
            else:
                ksi = x/t
                if ksi <= self.shockC_vel(C=1,Swst=self.Swi,Cst=0):
                    return 1
                else:
                    return 0

    def calc_Savg(self,t):
        '''
        This function calculates the average water saturation in the interval 0 <= xD <= 1 at tD=t.

        The expressions are based on analytical calculations.
        '''
        def Sa(t):
            return self.Swi + (1.0 - self.fw(self.Swi,C=0))*t
        def Sb(t):
            return self.Sw2 + (1.0 - self.fw(self.Sw2,C=0))*t
        def Sc(t):
            Sw = self.calc_Sol(1,t)
            return Sw + (1.0 - self.fw(Sw,C=1))*t
        def Sd(t):
            Sw = self.calc_Sol(1,t)
            return Sw + (1.0 - self.fw(Sw,C=0))*t
        def Se(t):
            return self.Sw1_hi + (1.0 - self.fw(self.Sw1_hi,C=1))*t
        if self.lowSwi:
            # This branch is for Swi <= Sw2_alt
            if self.events == 'shock':
                # Solution A
                v1 = self.shockC_vel(C=1,Swst=self.Sw2,Cst=0)
                v2 = self.shockS_vel(self.Sw2, self.Swi,Cst=0)
                tB1 = 1.0/v1
                tB2 = 1.0/v2
                if t <= tB2:
                    return Sa(t)
                elif t <= tB1:
                    return Sb(t)
                else:
                    return Sc(t)
            if self.events == 'rare + shock':
                # Solution B
                v1 = self.shockC_vel(C=1,Swst=self.Sw2,Cst=0)
                v3 = self.lamS(self.Sw2,C=0)
                v4 = self.shockS_vel(self.Sw3,self.Swi,Cst=0)
                tB1 = 1.0/v1
                tB3 = 1.0/v3
                tB4 = 1.0/v4
                if t <= tB4:
                    return Sa(t)
                elif t<= tB3:
                    return Sd(t)
                elif t <= tB1:
                    return Sb(t)
                else:
                    return Sc(t)
            else:
                # Solution C    
                v1 = self.shockC_vel(C=1,Swst=self.Sw2,Cst=0)
                v3 = self.lamS(self.Sw2,C=0)
                v5 = self.lamS(self.Swi,C=0)
                tB1 = 1.0/v1
                tB3 = 1.0/v3
                tB5 = 1.0/v5
                if t <= tB5:
                    return Sa(t)
                elif t <= tB3:
                    return Sd(t)
                elif t <= tB1:
                    return Sb(t)
                else:
                    return Sc(t)
        else:
            # This branch is for Swi > Sw2_alt
            v5 = self.lamS(self.Sw1_hi,C=1)
            v6 = self.shockC_vel(C=1,Swst=self.Swi,Cst=0)
            tB5 = 1.0/v5
            tB6 = 1.0/v6
            if t<= tB6:
                return Sa(t)
            elif t<= tB5:
                return Se(t)
            else:
                return Sc(t)
        
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

    def plot_profile(self, PVs, xstart=0.0, xend=1.0, ns = 101, sizex = 6, sizey = 4):
        '''
        This function plots the profiles of the water saturation Sw and normalized polymer
        concentration C for the FracFlow instance in the interval xstart <= xD <= xend 
        at the PV(s) injected specified by PVs.

        Input:
        PVs:    PV injected at which the profiles should be plotted, can be a single number or 
                a list, series or array of PVs
        xstart: xD value at which the profile should start
        xend:   xD value at which the profile should end
        ns:     number of xD points in the profile

        The function returns:
        data:       the Sw profile data for the last profile plotted
        dataC:      the C profile date for the last profile plotted
        fig, ax:    figure and axes data of the last plot
        '''
        # If only one number is input for PVs, convert to list
        if isinstance(PVs, (int, float)):
            PVs = [PVs]
        
        xs = np.linspace(xstart,xend,ns)

        for PV in PVs:
            fig, ax = plt.subplots(figsize=(sizex,sizey))
            data = pd.Series()
            dataC = pd.Series()
            t = PV
            for x in xs:
                data[x] = self.calc_Sol(x,t)
                dataC[x] = self.calc_SolC(x,t)
            ax.plot(data, 'b-', label='Sw profile')
            ax.plot(dataC, 'g--', label='C profile', lw=2)
            ax.set_ylim(-0.05,1.05)
            ax.set_xlabel('xD')
            ax.set_ylabel('Sw (frac) or C/Cinj (frac)')
            ax.set_title(f'{PV:4.2f} PV injected')
            ax.legend()
            ax.grid()
            plt.show()

        return data, dataC, fig, ax

    def calc_quad_points(self, t,eps=1e-8):
        '''
        This function generates the 'points' input data for the SciPy integration 
        routine 'quad' used in 'plot_Savg_Integration'.
        It specifies the discontinuitiess in the water saturation profile.
        '''
        if self.lowSwi:
            # This branch is for Swi <= Sw2_alt
            if self.events == 'shock':
                # Solution A
                v1 = self.shockC_vel(C=1,Swst=self.Sw2,Cst=0)
                v2 = self.shockS_vel(self.Sw2, self.Swi,Cst=0)
                xs = [v1*t,v2*t]
            if self.events == 'rare + shock':
                # Solution B
                v1 = self.shockC_vel(C=1,Swst=self.Sw2,Cst=0)
                v3 = self.lamS(self.Sw2,C=0)
                v4 = self.shockS_vel(self.Sw3,self.Swi,Cst=0)
                xs = [v1*t,v3*t,v4*t]
            else:
                # Solution C    
                v1 = self.shockC_vel(C=1,Swst=self.Sw2,Cst=0)
                v3 = self.lamS(self.Sw2,C=0)
                v5 = self.lamS(self.Swi,C=0)
                xs=[v1*t,v3*t,v5*t]
        else:
            # This branch is for Swi > Sw2_alt
            v5 = self.lamS(self.Sw1_hi,C=1)
            v6 = self.shockC_vel(C=1,Swst=self.Swi,Cst=0)
            xs = [v5*t,v6*t]
        points = []
        for x in xs:
            if (x>0) and (x<1): points.append(x)
        return points

    def plot_Savg_Integration(self, PVstart=0.0, PVend=2.0, ns = 41, quad_eps=1e-5):
        '''
        This function plots the average water saturation as function of PV injected for
        an instance of FracFlow. The average water saturation is calculated by integrating
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
        PVs = np.linspace(PVstart, PVend, ns)
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
        an instance of FracFlow. The average water saturation is calculated by performing
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
            NetInj[PV] = (1.0 - self.fw(self.calc_Sol(1.0,PV),self.calc_SolC(1.0,PV)))*DelPV
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

    def plot_Savg(self, PVstart=0.0, PVend=2.0, ns = 101):
        '''
        This function plots the average water saturation as function of PV injected for
        an instance of FracFlow. The average water saturation is calculated using the
        function 'calc_Savg'

        Input:
        PVstart:    start value of PV injected in the plot
        PVend:      end value of PV injected in the plot
        ns:         number of "PV injected" points in the plot

        The function returns:
        Savg:       Savg vs PV data in the plot
        fig, ax:    figure and axes data of the plot
        '''
        PVs = np.linspace(PVstart, PVend, ns)
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

    def plot_RF(self, PVstart=0.0, PVend=2.0, ns = 201):
        '''
        This function plots the RF and BSW as function of PV injected for
        an instance of FracFlow. The RF is calculated wrt to both Swi and Scw as
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
        PVs = np.linspace(PVstart, PVend, ns)

        RF_Scw = pd.Series()
        RF_Swi = pd.Series()
        BSW = pd.Series()

        for PV in PVs:
            RF_Scw[PV] = self.calc_RF(PV)
            RF_Swi[PV] = self.calc_RF(PV,wrtSwi=True)
            BSW[PV] = self.fw(self.calc_Sol(1,PV),self.calc_SolC(1,PV))

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
        This function plots the fractional flow solution for an instance of FracFlow.

        For Swi <= Sw2_alt the solution consists of:
        1) S-wave rarefaction (C=1) from Swinj to Sw1 (this part is skipped if S_rare_C1 is False)
        2) C-wave shock from (Sw=Sw1,C=1) to (Sw=Sw2,C=0). Note that the C-wave shock speed matches
            the front speed of the S-wave rarefactin (C=1) above
        3) One of the following scenarios depending on Swi
            a) S-wave rarefaction (C=0) from Sw2 to Swi
            b) S-wave shock (C=0) from Sw2 to Swi
            c) S-ware rarefaction (C=0) from Sw2 to Sw3, followed by S-wave shock (C=0) from Sw3 to Swi
        For Swi > Sw2_alt the solution consists of:
            1) S-wave rarefaction (C=1) from Swinj to Sw1_hi
            2) C-wave shock from (Sw=Sw1_hi,C=1) to (Sw=Swi,C=0). Note that the C-wave shock
                is faster than the front of the S-wave rarefaction.
            For this case also a zoomed out plot is generated for a more detailed view of
            the final C-wave shock

        Input:
        sizex:  x-dimension of the plot
        sizey:  y-dimension of the plot

        The function returns:
        fig, ax:    figure and axes data of the plot
        '''
        if self.lowSwi:
            fig, ax = plt.subplots(figsize=(sizex,sizey))
            Swinj = self.Swinj; Cinj=1.0
            Ss = np.linspace(self.Scw,1.0-self.Sorw,101)
            ax.plot(Ss,self.fw(Ss,0), 'b-', label = 'fw, C=0')
            ax.plot(Ss,self.fw(Ss,1), 'r-', label = 'fw, C=1')

            ax.plot(Swinj,self.fw(Swinj,Cinj),'ro', label = f'Swinj=1-Sorw={self.Swinj:6.4f}, C=1', markersize=8,fillstyle='none')
            if self.S_rare_C1:
                Ss = np.linspace(Swinj,self.Sw1,10)
                ax.plot(Ss,self.fw(Ss,Cinj),'g.', label='S-wave rarefaction, C=1')
                ax.plot(self.Sw1,self.fw(self.Sw1,Cinj),'go', label = f'Sw1={self.Sw1:6.4f}, C=1', markersize=8,fillstyle='none')
                print(f'Sw1 = {self.Sw1:6.4f}')
                print(f'Sw2 = {self.Sw2:6.4f}')
                print(f'lamS(Sw=Sw1,C=1) = {self.lamS(Sw=self.Sw1,C=1):6.4f}')
                print(f'C-wave shock speed (Sw1,C=1)->(Sw2,C=0) = {self.shockC_vel(C=1,Swst=self.Sw2,Cst=0):6.4f}')
            else:
                print(f'Sw1 = 1-Sorw = {self.Sw1:6.4f}')
                print(f'Sw2 = {self.Sw2:6.4f}')
                print(f'C-wave shock speed (1-Sorw,C=1)->(Sw2,C=0) = {self.shockC_vel(C=1,Swst=self.Sw2,Cst=0):6.4f}')

            Ci=0.0
            ax.plot((self.Sw1,self.Sw2),(self.fw(self.Sw1,Cinj),self.fw(self.Sw2,Ci)),'k-', label = 'C-wave shock', lw=3)
            ax.plot(self.Sw2,self.fw(self.Sw2,Ci),'ko', label = f'Sw2={self.Sw2:6.4f}, C=0', markersize=8,fillstyle='none')

            if self.events == 'rare':
                Ss = np.linspace(self.Sw2,self.Swi,10)
                ax.plot(Ss,self.fw(Ss,Ci),'r.', label = 'S-wave rarefaction, C=0')
                print(f'S-wave rarefaction speed, back = {self.lamS(self.Sw2,C=0):6.4f}')
                print(f'S-wave rarefaction speed, front = {self.lamS(self.Swi,C=0):6.4f}')
            elif self.events == 'shock':
                ax.plot((self.Sw2,self.Swi),(self.fw(self.Sw2,0),self.fw(self.Swi,0)),'g-', label = 'S-wave shock, C=0', lw=2)
                print(f'S-wave shock speed = {self.shockS_vel(self.Sw2,self.Swi,Cst=0):6.4f}')
            else: # self.events == 'rare + shock'
                Ss = np.linspace(self.Sw2,self.Sw3,10)
                ax.plot(Ss,self.fw(Ss,0),'r.', label = 'S-wave rarefaction, C=0')
                ax.plot(self.Sw3,self.fw(self.Sw3,Ci),'o', label = f'Sw3={self.Sw3:6.4f}, C=0', markeredgecolor='maroon', markersize=8,fillstyle='none')
                ax.plot((self.Sw3,self.Swi),(self.fw(self.Sw3,0),self.fw(self.Swi,0)),'g-', label = 'S-wave shock, C=0', lw=2)
                print(f'S-wave rarefaction speed, back = {self.lamS(self.Sw2,C=0):6.4f}')
                print(f'Sw3 = {self.Sw3:6.4f}')
                print(f'S-wave rarefaction speed, front = S-wave shock speed = {self.shockS_vel(self.Sw3,self.Swi,Cst=0):6.4f}')

            ax.plot(self.Swi,self.fw(self.Swi,0),'bo', label=f'Swi={self.Swi:6.4f}, C=0', markersize=8, fillstyle='none')

            D=self.ad(1)-self.ad(0)
            ax.plot((-D,self.Sw2,self.Sw1),(0,self.fw(self.Sw2,0),self.fw(self.Sw1,1)),'r-.', lw=1)

            ax.set_xlabel('Sw')
            ax.set_ylabel('fw')
            ax.grid()
            ax.legend()
        else:
            fig, ax = plt.subplots(1,2,figsize=(sizex,sizey))
            Swinj = self.Swinj; Cinj=1.0
            Ci = 0.0
            D = self.ad(1)-self.ad(0)

            Ss = np.linspace(self.Scw,1.0-self.Sorw,101)
            ax[0].plot(Ss,self.fw(Ss,0), 'b-', label = 'fw, C=0')
            ax[0].plot(Ss,self.fw(Ss,1), 'r-', label = 'fw, C=1')
            #
            ax[0].plot(Swinj,self.fw(Swinj,Cinj),'ro', label = f'Swinj={Swinj:6.4f}', markersize=8,fillstyle='none')
            Ss = np.linspace(Swinj,self.Sw1_hi,10)
            ax[0].plot(Ss,self.fw(Ss,Cinj),'g.', label='S-wave rarefaction, C=1')
            ax[0].plot(self.Sw1_hi,self.fw(self.Sw1_hi,Cinj),'go', label = f'Sw1_hi={self.Sw1_hi:6.4f}, C=1', markersize=8,fillstyle='none')
            #
            ax[0].plot((self.Sw1_hi,self.Swi),(self.fw(self.Sw1_hi,Cinj),self.fw(self.Swi,Ci)),'k-', label = 'C-wave shock', lw=3)
            ax[0].plot(self.Swi,self.fw(self.Swi,Ci),'bo', label = f'Swi={self.Swi:6.4f}, C=0', markersize=8,fillstyle='none')
            #
            ax[0].plot((-D,self.Swi,self.Sw1_hi),(0,self.fw(self.Swi,0),self.fw(self.Sw1_hi,1)),'r-.', lw=1)
            #
            ax[0].set_xlabel('Sw')
            ax[0].set_ylabel('fw')
            ax[0].legend()
            ax[0].grid()

            print(f'Sw1_hi = {self.Sw1_hi:6.4f}')
            print(f'lamS(Sw=Sw1_hi,C=1) = {self.lamS(Sw=self.Sw1_hi,C=1):6.4f}')
            print(f'C-wave shock speed = {self.shockC_vel(C=1,Swst=self.Swi,Cst=0):6.4f}')

            Ss = np.linspace(self.Scw,1.0-self.Sorw,101)
            ax[1].plot(Ss,self.fw(Ss,0), 'b-', label = 'fw, C=0')
            ax[1].plot(Ss,self.fw(Ss,1), 'r-', label = 'fw, C=1')
            #
            ax[1].plot(Swinj,self.fw(Swinj,Cinj),'ro', label = f'Swinj={Swinj:6.4f}', markersize=8,fillstyle='none')
            Ss = np.linspace(Swinj,self.Sw1_hi,10)
            ax[1].plot(Ss,self.fw(Ss,Cinj),'g.', label='S-wave rarefaction, C=1')
            ax[1].plot(self.Sw1_hi,self.fw(self.Sw1_hi,Cinj),'go', label = f'Sw1_hi={self.Sw1_hi:6.4f}, C=1', markersize=8,fillstyle='none')
            #
            ax[1].plot((self.Sw1_hi,self.Swi),(self.fw(self.Sw1_hi,Cinj),self.fw(self.Swi,Ci)),'k-', label = 'C-wave shock', lw=3)
            ax[1].plot(self.Swi,self.fw(self.Swi,Ci),'bo', label = f'Swi={self.Swi:6.4f}, C=0', markersize=8,fillstyle='none')
            #
            ax[1].plot((-D,self.Swi,self.Sw1_hi),(0,self.fw(self.Swi,0),self.fw(self.Sw1_hi,1)),'r-.', lw=1)
            #
            ax[1].set_xlim(self.Sw1_hi*0.9,1.05)
            ax[1].set_ylim(0.85,1.05)
            ax[1].set_xlabel('Sw')
            ax[1].set_ylabel('fw')
            ax[1].legend()
            ax[1].grid()
        
        return fig, ax
