import numpy as np
from scipy import constants
import pandas as pd
import matplotlib.pyplot as plt

class PBR:
    def __init__(self, FCH4, FH2O, FCO2, FCO, FH2, FN2, T, P, z, d, n_w, n_t):
        """
        Parameters
        ----------
        FCH4 : array size of n_w
            Flow of CH4 into the reactor in mol/s.
        FH2O : array size of n_w
            Flow of H2O into the reactor in mol/s.
        FCO2 : array size of n_w
            Flow of CO2 into the reactor in mol/s.
        FCO : array size of n_w
            Flow of CO into the reactor in mol/s.
        FH2 : array size of n_w
            Flow of H2 into the reactor in mol/s.
        FN2 : array size of n_w
            Flow of N2 into the reactor in mol/s.
        T : array size of n_w
            Inlet temperature in degree Celsius.
        P : array size of n_w
            Inlet pressure in bar.
        T : array size of n_w
            Inlet temperature in degree Celsius.
        z : int
            Length of the reactor in m.
        d : float
            Diameter of the reactor in m.
        n_w : int
            number of spatial increment
        n_t : int
            number of temporal increment       

        """       
        self.z = z                                  # length of the reactor in m
        self.d = d                                  # diameter in m
        self.A = np.pi * (self.d**2)/4              # Area of the reactor
        self.dHrxn1 = 206000                        # Heat of reaction of CH4 + H2O --> CO + 3H2
        self.dHrxn2 = -41000                        # Heat of reaction of CO + H2O --> CO2 + H2
        self.dHrxn3 = 165000                        # Heat of reaction of CH4 + 2H2O --> CO2 + 4H2
        
        self.poros = 0.4                            # Bed porosity
        self.Dp = 0.02                              # Catalyst particle diameter
        self.cat_dens = 550                         # Catalyst density in kg/m3
        self.cat_cp  = 670                          # catalyst heat capacity in J/kg.K

        self.g = constants.g                        # gravity acceleration
        self.R = constants.physical_constants['molar gas constant'][0] #Ideal Gas Constant
        self.T_a = 1000+273.15                      # Temperature of heating medium

        self.volume = np.pi*self.d**2 /4 * self.z   # total volume of reactor
        self.weight = self.volume*(self.cat_dens*(1-self.poros))
        self.n_w = n_w                              # number of weight increment
        self.n_t = n_t                              # number of time increment
        self.w = 0
        self.t = 0
        self.dw = self.weight/self.n_w
        self.dz = self.dw/(self.cat_dens)*4/(np.pi*self.d**2)
        self.dt = 1

        self.FCH4 = FCH4
        self.FH2O = FH2O         
        self.FH2 = FH2         
        self.FCO = FCO        
        self.FCO2 = FCO2        
        self.FN2 = FN2
        self.T = T + 273.15
        self.P = self._convert_to_Pa(P)  

        self.u = self._velocity()                   # Initial superficial velocity    

    def _velocity(self):
        # Density of each component
        d_CH4 = self._density('CH4')
        d_CO = self._density('CO')
        d_CO2 = self._density('CO2')
        d_H2 = self._density('H2')
        d_H2O = self._density('H2O')
        d_N2 = self._density('N2')

        Ftotal = self.FCH4[self.w-1] + self.FCO2[self.w-1] + self.FCO[self.w-1] + self.FH2[self.w-1] + self.FH2O[self.w-1] + self.FN2[self.w-1]

        # Average molecular weight
        M = (self.FCH4[self.w-1] * 16.04 + self.FCO2[self.w-1] * 44.01 + self.FCO[self.w-1] * 28.01 + self.FH2[self.w-1] * 2.02 + self.FH2O[self.w-1] * 18.02 + self.FN2[self.w-1] * 28.02) / Ftotal

        # Fraction of each component
        y_CH4 = self.FCH4[self.w-1] / Ftotal
        y_CO2 = self.FCO2[self.w-1] / Ftotal
        y_CO = self.FCO[self.w-1] / Ftotal
        y_H2 = self.FH2[self.w-1] / Ftotal
        y_H2O = self.FH2O[self.w-1] / Ftotal
        y_N2 = self.FN2[self.w-1] / Ftotal

        density = (d_CH4 * y_CH4 + d_CO * y_CO + d_CO2 * y_CO2 + d_H2 * y_H2 + d_H2O * y_H2O + d_N2 * y_N2)

        Q = Ftotal * M/1000/density                    # Volumetric flowrate in m3/s
        u = Q/(3.14*self.d**2/4)                       # Velocity in m/s

        return u

    def _convert_to_Pa(self, P):                    # Convert atm to Pa
        return P*101325

    def _r1(self):                                  # CH4 + H2O --> CO + 3H2
        if self.P_H2 > 0 and self.P_CH4 > 0 and self.P_CO > 0 and self.P_CO2 > 0 and self.P_H2O > 0:
            k1 = 1.17e12*np.exp(-240100/(self.R*self.T[self.w-1]))
            K1 = 4.71e12*np.exp(-224000/(self.R*self.T[self.w-1]))
            den = self._DEN()            
            r_1 = k1*(self.P_CH4*self.P_H2O/self.P_H2**2.5-self.P_H2**0.5*self.P_CO/K1)/den**2 
        else:
            r_1 = 0
        return r_1

    def _r2(self):                                  # CO + H2O --> CO2 + H2
        if self.P_H2 > 0 and self.P_CH4 > 0 and self.P_CO > 0 and self.P_CO2 > 0 and self.P_H2O > 0:
            k2 = 5.42e2*np.exp(67130/(self.R*self.T[self.w-1]))
            K2 = 1.14e-2*np.exp(37300/(self.R*self.T[self.w-1]))
            den = self._DEN()            
            r_2 = k2*(self.P_CO*self.P_H2O/self.P_H2-self.P_CO2/K2)/den**2
        else:
            r_2 = 0
        return r_2

    def _r3(self):                                  # CH4 + 2H2O --> CO2 + 4H2
        if self.P_H2 > 0 and self.P_CH4 > 0 and self.P_CO > 0 and self.P_CO2 > 0 and self.P_H2O > 0:
            k3 = 2.83e11*np.exp(-243900/(self.R*self.T[self.w-1]))
            K1 = 4.71e12*np.exp(-224000/(self.R*self.T[self.w-1]))
            K2 = 1.14e-2*np.exp(37300/(self.R*self.T[self.w-1]))
            den = self._DEN()
            K3 = K1*K2
            r_3 = k3*(self.P_CH4*self.P_H2O**2/self.P_H2**3.5-self.P_H2**0.5*self.P_CO2/K3)/den**2
        else:
            r_3 = 0
        return r_3

    def _DEN(self):                                 # Denominator of the kinetics of reaction
        KCO = 8.25e-5*np.exp(70650/(self.R*self.T[self.w-1]))
        KH2 = 6.12e-9*np.exp(82900/(self.R*self.T[self.w-1]))
        KCH4 = 6.65e-4*np.exp(38280/(self.R*self.T[self.w-1]))
        KH2O = 1.77e5*np.exp(-88680/(self.R*self.T[self.w-1]))
        den = 1 + KCO*self.P_CO + KH2*self.P_H2 + KCH4*self.P_CH4 + KH2O*self.P_H2O/self.P_H2
        return den

    def _partial_presure(self, flow, total_flow):   # To convert partial pressure
        P = flow/total_flow*self.P[self.w-1]/(self._convert_to_Pa(1))
        return P

    def _cp_T(self, component):                     # Cp = a + bT + cT^2 + dT^3 + e/T^2 in J/mol.K
        A = {'CH4': 20.44,
             'CO2': 19.84,
             'CO': 31.68,
             'H2': 27.03,
             'H2O': 31.96,
             'N2': 0.9125}

        B = {'CH4': 0.04486,
             'CO2': 0.07251,
             'CO': -0.01577,
             'H2': 0.009772,
             'H2O': 0.01073,
             'N2': 0.000321}

        C = {'CH4': 2.321e-5,
             'CO2': -5.35e-5,
             'CO': 2.999e-5,
             'H2': -1.467e-5,
             'H2O': -7.181e-6,
             'N2': -6.723e-8}

        D = {'CH4': -1.66e-8,
            'CO2': 1.538e-8,
            'CO': -1.254e-8,
            'H2': 8.085e-9,
            'H2O': 5.34e-9,
            'N2': 0}

        E = {'CH4': 0.7584,
             'CO2': 0.3506,
             'CO': 0.4907,
             'H2': 0.1309,
             'H2O': 0.8265,
             'N2': 0}

        Cp = A[component] + B[component]*self.T[self.w-1] + C[component]*self.T[self.w-1]**2 + D[component]*self.T[self.w-1]**3 + E[component]/self.T[self.w-1]**2
        return Cp

    def _density(self, component):                  # dens = p00 + p10*P + p01*T + p20*P^2 + p11*P*T + p02*T^2 + p30*P^3 + p21*P^2*T + p12*P*T^2 in kg/m3
        p00 = {'CH4': 6.322,
             'CO2': 17.72,
             'CO': 10.85,
             'H2': 0.7793,
             'H2O': 8.013,
             'N2': 10.86}

        p10 = {'CH4': 4.125e-6,
             'CO2': 1.141e-5,
             'CO': 7.119e-6,
             'H2': 5.158e-7,
             'H2O': 5.027e-6,
             'N2': 7.132e-6}

        p01 = {'CH4': -0.01348,
             'CO2': -0.03792,
             'CO': -0.023,
             'H2': -0.001658,
             'H2O': -0.01771,
             'N2': -0.02305}

        p20 = {'CH4': -4.779e-15,
             'CO2': -1.168e-14,
             'CO': -1.092e-14,
             'H2': -4.497e-16,
             'H2O': 1.14e-14,
             'N2': 1.039e-14}

        p11 = {'CH4': -2.175e-9,
             'CO2': -6.062e-9,
             'CO': -3.712e-9,
             'H2': -2.708e-10,
             'H2O': -2.86e-9,
             'N2': -3.725e-9}

        p02 = {'CH4': 7.09e-6,
             'CO2': 2.002e-5,
             'CO': 1.203e-5,
             'H2': 8.701e-7,
             'H2O': 9.644e-6,
             'N2': 1.207e-5}
        
        dens = p00[component] + p10[component]*(self.P[self.w-1]) + p01[component]*self.T[self.w-1] + p20[component]*(self.P[self.w-1])**2 + p11[component]*(self.P[self.w-1])*self.T[self.w-1] + p02[component]*self.T[self.w-1]**2
        return dens

    def _viscosity(self, component):                # viscosity = A + BT + CT^2 in Pa.s
        A = {'CH4': 1132,
             'CO2': -352.1,
             'CO': 5342,
             'H2': 3.943e4,
             'H2O': -1748,
             'N2': 3135}

        B = {'CH4': 36.62,
             'CO2': 55.62,
             'CO': 42.67,
             'H2': -92.13,
             'H2O': 43.98,
             'N2': 53.37}

        C = {'CH4': -0.01032,
             'CO2': -0.01398,
             'CO': -0.001366,
             'H2': 0.07781,
             'H2O': -0.009722,
             'N2': -0.01521}

        visc = (A[component] + B[component]*self.T[self.w-1] + C[component]*self.T[self.w-1]**2)/1e9
        return visc
    
    def _k(self, component):          # Thermal conductivity in W/m.K
        A = {'CH4': 0.01283,
             'CO2': 0.02397,
             'CO': 0.02708,
             'H2': 0.195,
             'H2O': 0.04318,
             'N2': 0.02908}

        B = {'CH4': 0.0002125,
             'CO2': 5.992e-5,
             'CO': 5.243e-5,
             'H2': 0.0003066,
             'H2O': 6.09e-5,
             'N2': 4.977e-5}

        C = {'CH4': 2.503e-5,
             'CO2': 2.662e-5,
             'CO': 1.72e-5,
             'H2': 2.667e-5,
             'H2O': 7.315e-5,
             'N2': 1.571e-5}

        convert = 1/self._convert_to_Pa(1)

        k = A[component] + B[component] * (self.T[self.w-1]-273.15) + C[component] * (self.P[self.w-1] * convert)
        return k
    
    def _UA(self):
        Ftotal = self.FCH4[self.w-1] + self.FCO2[self.w-1] + self.FCO[self.w-1] + self.FH2[self.w-1] + self.FH2O[self.w-1] + self.FN2[self.w-1]

        # Fraction of each component
        y_CH4 = self.FCH4[self.w-1] / Ftotal
        y_CO2 = self.FCO2[self.w-1] / Ftotal
        y_CO = self.FCO[self.w-1] / Ftotal
        y_H2 = self.FH2[self.w-1] / Ftotal
        y_H2O = self.FH2O[self.w-1] / Ftotal
        y_N2 = self.FN2[self.w-1] / Ftotal

        # Density of each component
        d_CH4 = self._density('CH4')
        d_CO = self._density('CO')
        d_CO2 = self._density('CO2')
        d_H2 = self._density('H2')
        d_H2O = self._density('H2O')
        d_N2 = self._density('N2')

        # Viscosity of each component
        m_CH4 = self._viscosity('CH4')
        m_CO = self._viscosity('CO')
        m_CO2 = self._viscosity('CO2')
        m_H2 = self._viscosity('H2')
        m_H2O = self._viscosity('H2O')
        m_N2 = self._viscosity('N2')

        # Heat capacity
        cp_CH4 = self._cp_T('CH4')
        cp_CO = self._cp_T('CO')
        cp_CO2 = self._cp_T('CO2')
        cp_H2 = self._cp_T('H2')
        cp_H2O = self._cp_T('H2O')
        cp_N2 = self._cp_T('N2')

        # Thermal conductivity
        k_CH4 = self._k('CH4')
        k_CO = self._k('CO')
        k_CO2 = self._k('CO2')
        k_H2 = self._k('H2')
        k_H2O = self._k('H2O')
        k_N2 = self._k('N2')

        # Average properties
        density = (d_CH4 * y_CH4 + d_CO * y_CO + d_CO2 * y_CO2 + d_H2 * y_H2 + d_H2O * y_H2O + d_N2 * y_N2)
        miu = (m_CH4 * y_CH4 + m_CO * y_CO + m_CO2 * y_CO2 + m_H2 * y_H2 + m_H2O * y_H2O + m_N2 * y_N2)
        cp = (cp_CH4 * y_CH4 + cp_CO * y_CO + cp_CO2 * y_CO2 + cp_H2 * y_H2 + cp_H2O * y_H2O + cp_N2 * y_N2)
        k = (k_CH4 * y_CH4 + k_CO * y_CO + k_CO2 * y_CO2 + k_H2 * y_H2 + k_H2O * y_H2O + k_N2 * y_N2)

        # convection inside the tube
        Re_tube = density*self.u*self.d/miu
        Pr_tube = miu*cp/k

        h_tube = 0.027*Re_tube*Pr_tube**0.33*k/self.d

        # Conduction of tube
        k_tube = 53           # thermal conductivity of tube in W/m.K assuming carbon steel
        R_conduction = self.d*np.log(self.d/(self.d-0.049*2))/(2*k_tube)
        UA = 1/(1/h_tube + R_conduction)
        return UA
    
    def _FCp(self):
        component = ['CH4', 'CO2', 'CO', 'H2', 'H2O', 'N2']
        flow = [self.FCH4[self.w-1], self.FCO2[self.w-1], self.FCO[self.w-1], self.FH2[self.w-1], self.FH2O[self.w-1], self.FN2[self.w-1]]
        Fcp = []

        # Calculating Flow x Cp for each component
        for i in range(len(component)):
            cpi = self._cp_T(component[i])
            fcp = flow[i]*cpi
            Fcp.append(fcp)

        Fcp = sum(Fcp)
        return Fcp

    def _dP_dz(self):                           # Momentum balance based on Ergun Equation
        term1 = 150 * (1 - self.poros) / (self.poros**3 * self.Dp)
        term2 = 1.75 * (1 - self.poros) / self.poros
        Ftotal = self.FCH4[self.w-1] + self.FCO2[self.w-1] + self.FCO[self.w-1] + self.FH2[self.w-1] + self.FH2O[self.w-1] + self.FN2[self.w-1]

        # Fraction of each component
        y_CH4 = self.FCH4[self.w-1] / Ftotal
        y_CO2 = self.FCO2[self.w-1] / Ftotal
        y_CO = self.FCO[self.w-1] / Ftotal
        y_H2 = self.FH2[self.w-1] / Ftotal
        y_H2O = self.FH2O[self.w-1] / Ftotal
        y_N2 = self.FN2[self.w-1] / Ftotal

        # Density of each component
        d_CH4 = self._density('CH4')
        d_CO = self._density('CO')
        d_CO2 = self._density('CO2')
        d_H2 = self._density('H2')
        d_H2O = self._density('H2O')
        d_N2 = self._density('N2')

        # Viscosity of each component
        m_CH4 = self._viscosity('CH4')
        m_CO = self._viscosity('CO')
        m_CO2 = self._viscosity('CO2')
        m_H2 = self._viscosity('H2')
        m_H2O = self._viscosity('H2O')
        m_N2 = self._viscosity('N2')

        # Average density and viscosity
        density = (d_CH4 * y_CH4 + d_CO * y_CO + d_CO2 * y_CO2 + d_H2 * y_H2 + d_H2O * y_H2O + d_N2 * y_N2)
        miu = (m_CH4 * y_CH4 + m_CO * y_CO + m_CO2 * y_CO2 + m_H2 * y_H2 + m_H2O * y_H2O + m_N2 * y_N2)

        dP_dz = -term1 * (1 - self.poros) * miu * (self.u / self.poros) + term2 * (1 - self.poros) * density * self.u**2
        return dP_dz
    
    def solve(self):                                 # Main function in this class
        CH4_result = np.zeros((self.n_w,self.n_t))
        H2O_result = np.zeros((self.n_w,self.n_t))
        CO_result = np.zeros((self.n_w,self.n_t))
        CO2_result = np.zeros((self.n_w,self.n_t))
        H2_result = np.zeros((self.n_w,self.n_t))
        N2_result = np.zeros((self.n_w,self.n_t))
        T_result = np.zeros((self.n_w,self.n_t))

        for self.t in range(1, self.n_t):
            FCH4_old = self.FCH4.copy()
            FH2O_old = self.FH2O.copy()
            FCO_old = self.FCO.copy()
            FCO2_old = self.FCO2.copy()
            FH2_old = self.FH2.copy()
            FN2_old = self.FN2.copy()
            T_old = self.T.copy()

            for self.w in range(1, self.n_w):
                tot_flow  = self.FCH4[self.w-1] + self.FH2O[self.w-1] + self.FCO[self.w-1] + self.FCO2[self.w-1] + self.FH2[self.w-1] + self.FN2[self.w-1]

                self.P_CH4 = self._partial_presure(self.FCH4[self.w-1], tot_flow)
                self.P_H2O = self._partial_presure(self.FH2O[self.w-1], tot_flow)
                self.P_CO = self._partial_presure(self.FCO[self.w-1], tot_flow)
                self.P_CO2 = self._partial_presure(self.FCO2[self.w-1], tot_flow)
                self.P_H2 = self._partial_presure(self.FH2[self.w-1], tot_flow)

                r_1 = self._r1()
                r_2 = self._r2()
                r_3 = self._r3()

                r_CH4 = -r_1 - r_3
                r_H2O = -r_1 - r_2 - 2*r_3
                r_CO = r_1 - r_2
                r_CO2 = r_2 + r_3
                r_H2 = 3*r_1 + r_2 +4*r_3
                r_N2 = 0

                self.u = self._velocity()
                FCp = self._FCp()

                f1 = 1/(self.z*self.A*self.cat_dens)

                self.FCH4[self.w] = self.FCH4[self.w-1] + self.dw * (r_CH4 + f1 * (FCH4_old[self.w] - FCH4_old[self.w-1])/self.dt)
                self.FH2O[self.w] = self.FH2O[self.w-1] + self.dw * (r_H2O + f1 * (FH2O_old[self.w] - FH2O_old[self.w-1])/self.dt)
                self.FCO[self.w] = self.FCO[self.w-1] + self.dw * (r_CO + f1 * (FCO_old[self.w] - FCO_old[self.w-1])/self.dt)
                self.FCO2[self.w] = self.FCO2[self.w-1] + self.dw * (r_CO2 + f1 * (FCO2_old[self.w] - FCO2_old[self.w-1])/self.dt)
                self.FH2[self.w] = self.FH2[self.w-1] + self.dw * (r_H2 + f1 * (FH2_old[self.w] - FH2_old[self.w-1])/self.dt)
                self.FN2[self.w] = self.FN2[self.w-1] + self.dw * (r_N2 + f1 * (FN2_old[self.w] - FN2_old[self.w-1])/self.dt)
                self.T[self.w] = self.T[self.w-1] + self.dw/(FCp+self.weight*self.cat_cp) * ((-1/(self.u*self.A*self.cat_dens)*FCp*(T_old[self.w]-T_old[self.w-1])/self.dt) + (r_1*-self.dHrxn1 + r_2*-self.dHrxn2 + r_3*-self.dHrxn3) + self._UA()*(self.T_a - self.T[self.w-1]))
                self.P[self.w] = self.P[self.w-1] - self._dP_dz() * self.dz

                CH4_result[self.w-1,self.t-1] = self.FCH4[self.w]
                H2O_result[self.w-1,self.t-1] = self.FH2O[self.w]
                CO_result[self.w-1,self.t-1] = self.FCO[self.w]
                CO2_result[self.w-1,self.t-1] = self.FCO2[self.w]
                H2_result[self.w-1,self.t-1] = self.FH2[self.w]
                N2_result[self.w-1,self.t-1] = self.FN2[self.w]
                T_result[self.w-1,self.t-1] = self.T[self.w]           
               

        profile = pd.DataFrame({'CH4 (mol/s)': self.FCH4,
                              'H2O (mol/s)': self.FH2O,
                              'CO (mol/s)' : self.FCO,
                              'H2 (mol/s)' : self.FH2,
                              'CO2 (mol/s)': self.FCO2,
                              'N2 (mol/s)' : self.FN2,
                              'T (deg C)' : self.T - 273.15,
                              'P (bar)':self.P/(self._convert_to_Pa(1))})  
                              
        return profile
        # return CH4_result, H2O_result, CO_result, CO2_result, H2_result, N2_result,T_result