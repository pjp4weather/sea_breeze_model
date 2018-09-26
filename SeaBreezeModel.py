# Exercise 1.14 Atmospheric Dynamics 2018 (page 96): 
# http://www.staff.science.uu.nl/~delde102/AtmosphericDynamics[2018]Ch1.pdf

# https://docs.python.org/2.7/
# IMPORT MODULES
import numpy as np  # http://www.numpy.org
import matplotlib.pyplot as plt   # http://matplotlib.org


class seaBreezeModel:
    """ A simple Sea breeze model"""
    
    def __init__(self,lat,A,phase,gamma,dpdx_synop):
        """ set the parameters of the model"""
        
        # constants
        self.omega = 0.000072792  # angular velocity Earth [s^-1]
        self.rho = 1.25 # density
        self.phi = 0.0 # phase of the pressure gradient
        self.earth_radius = 6.3e6
        self.g = 9.81
        
        self.phase = phase # phase of surface pressure gradient in time
        self.A = A       # parameters
        self.lat = lat # latitude of IJmuiden in degress
        self.latRad = self.lat * np.pi/180
        self.gamma = gamma
        

        
        # infered parameters
        self.fcor = 2 * self.omega *  np.sin(lat * np.pi/180)  # Coriolis parameter
        self.vg = 1./(self.rho *self.fcor)* dpdx_synop        
        
        self.C1= - A / ( self.fcor * self.rho ) * ( ( self.omega**2 / ( self.fcor**2 - self.omega**2 ) ) + 1)
        self.C3 = self.A * self.omega / (self. rho * ( self.fcor**2 - self.omega**2 ) )
        
    def fIdeal(self,t,u,v,*arg):
        """evoltuiion equation for the sea breeze model in an ideal case without 
        friction, geostrophic wind and the tan(lat) term"""
        return self.fuIdeal(t,u,v), self.fvIdeal(t,u,v)
    
    def fuIdeal(self,t,u,v):
        """evolution equation for the x-component"""
        return self.fcor*v - (self.A/self.rho)* np.cos((self.omega*t)+self.phase) 

    def fvIdeal(self,t,u,v):
        """evolution equation for the y-component"""
        return -self.fcor * u
    
    
    def f(self,t,u,v,vg):
        """evoltuiion equation for the sea breeze model"""
        return self.fu(t,u,v,vg), self.fv(t,u,v)
    
    def fu(self,t,u,v,vg):
        """evolution equation for the x-component"""
        return  self.fcor*(v-vg) \
                - (self.A/self.rho)* np.cos((self.omega*t)+self.phase) \
                - self.gamma * u \
                + u*v*np.tan(self.latRad)/self.earth_radius 
     

    def fv(self,t,u,v):
        """evolution equation for the y-component"""
        return - self.fcor * u \
               - self.gamma * v \
               - u**2 * np.tan(self.latRad)/self.earth_radius 
    
class integration:
    """ a object that contains different methods for the integration of a 2D model"""
    
    def __init__(self,dt,tmax_h,corrector_step=False,scheme="rk4"):
        """ 
        define the setup of the integration
        
        Parameters
        ----------
        dt : float
           time step for the integration scheme
        tmax_h : float
            maximum integration time in hours
        corrector_step : bool 
            use a corrector step in the integration scheme (default False)
        scheme : str
            chose a integration scheme, either "rk4" or "euler" (default euler)
        
        """
        
        self.dt = dt
        self.tmax_h = tmax_h
        self.corrector_step = corrector_step
        self.int_scheme = eval("self."+scheme)
        
        self.iters = int(tmax_h * 3600/dt) 
        self.time = np.arange(0,tmax_h * 3600,dt)
        self.u,self.v = np.zeros(self.iters), np.zeros(self.iters)

    def euler(self,x_start,y_start,f,*args):
        """ 
        euler forward method
        
        Parameters
        ----------
        x_start, y_start :  float
            starting values for the integration scheme
        f : function
            function that is used to integrate the x and y values
        *args: list of arguments for f
            arguments of the used function
        """
        dxdt, dydt = f(*args) 
        
        retx = x_start + dxdt*self.dt
        rety = y_start + dydt*self.dt
        
        return retx, rety
    
    
    def rk4(self,x_start,y_start,f,*args):
        """
        runge-kutta 4 method
        
        Parameters
        ----------
        x_start, y_start :  float
            starting values for the integration scheme
        f : function
            function that is used to integrate the x and y values
        *args: list of arguments for f
            arguments of the used function
        """
        
        h = self.dt
        

        k11,k12 = f(args[0],
                    args[1], 
                    args[2],
                    args[3])
        
        k21,k22 = f(args[0] + h/2., 
                    args[1] + h/2. * k11, 
                    args[2] + h/2. * k12,
                    args[3])
        
        k31,k32 = f(args[0] + h/2.,
                    args[1] + h/2. * k21, 
                    args[2] + h/2. * k22,
                    args[3])
        
        k41,k42 = f(args[0] + h,
                    args[1] + h * k31, 
                    args[2] + h * k32,
                    args[3])
        
        retx = x_start + h/6. * ( k11 + 2.*k21 + 2.*k31 + k41)
        rety = y_start + h/6. * ( k12 + 2.*k22 + 2.*k32 + k42)
        
        return retx, rety
    
    def integrate2D(self,f,vg,u_init=0.,v_init=0.):
        """ 
        scheme to integrate a 2-dimensional ODE
        (note that the integration scheme is already chosen at this point)
        
        Parameters
        ----------
        f: function
            function to use for the integration
        u_init, v_init:  float
            initial values
        """
        
        self.u[0] = u_init
        self.v[0] = v_init
        
        for t in range(self.iters-1): 
            self.u[t+1], self.v[t+1] = self.int_scheme(self.u[t], self.v[t], f ,  self.time[t],self.u[t],self.v[t],vg[t])
            
            
            if  self.corrector_step:
                self.u[t+1],self.v[t+1] = self.int_scheme(self.u[t], self.v[t], f ,  self.time[t],self.u[t+1],self.v[t+1],vg[t])


#%% =============================================================================
# Main
# =============================================================================
if __name__=="__main__":
    plt.close("all")
    # define integration method
    dt = 360.
    tmax_h = 48.
    corrector_step = False
    
    method = "euler"
    # define model
    lat = 52
    A = 0.001
    phase = 0.
    gamma = 0.
    
    
    integrator = integration(dt,tmax_h,corrector_step,method)
    vg = np.zeros(integrator.iters)
    sb = seaBreezeModel(lat,A,phase,gamma,vg)
    integrator.integrate2D(sb.fIdeal,sb.vg)
    u_euler,v_euler,time = integrator.u,integrator.v, integrator.time 
    
    corrector_step = True
    
    integrator = integration(dt,tmax_h,corrector_step,method)
    integrator.integrate2D(sb.fIdeal, sb.vg)
    u_matsuno,v_matsuno,time = integrator.u,integrator.v, integrator.time 
    
    corrector_step = False
    method = "rk4"
    
    integrator = integration(dt,tmax_h,corrector_step,method)
    integrator.integrate2D(sb.fIdeal,vg)
    u_rk4,v_rk4,time = integrator.u,integrator.v, integrator.time 
        
    
    # Analytical solution
    u_ana = np.zeros((integrator.iters))# analytical solution x-component velocity
    u_ana = sb.C1 * np.sin(sb.fcor * time) +  sb.C3 * np.sin((sb.omega * time) + sb.phase) 
    
    # MAKE PLOT of evolution in time of u and u_ana
    time_axis = time / 3600.
    
    plt.figure()
    plt.plot(time_axis, u_euler, label = 'u (numerical solution) euler')
    plt.plot(time_axis, u_matsuno, label = 'u (numerical solution) matsuno')
    plt.plot(time_axis, u_rk4, label = 'u (numerical solution) rk4')

    plt.plot(time_axis, u_ana, 'k:', label = 'u (analytical solution)')
    plt.axis([0,time_axis[integrator.iters-1],-25.0,25.0])  # define axes 
    plt.xticks(np.arange(0,time_axis[integrator.iters-1],6), fontsize=12) 
    plt.yticks(np.arange(-25.0,25.0,5), fontsize=12) 
    plt.xlabel('time [hours]', fontsize=14) # label along x-axes
    plt.ylabel('u [m/s]', fontsize=14) # label along x-axes
    plt.legend()
    plt.grid(True)
    plt.savefig("img/task1a.png") # save plot as png-file
    plt.show() # show plot on screen
    
    plt.figure()
    plt.plot(time_axis, v_euler, label = 'v (numerical solution) euler')
    plt.plot(time_axis, v_matsuno, label = 'v (numerical solution) matsuno')
    plt.plot(time_axis, v_rk4, label = 'v (numerical solution) rk4')

    #plt.plot(time_axis, v_ana, 'k:', label = 'v (analytical solution)')
    plt.axis([0,time_axis[integrator.iters-1],-25.0,25.0])  # define axes 
    plt.xticks(np.arange(0,time_axis[integrator.iters-1],6), fontsize=12) 
    plt.yticks(np.arange(-25.0,25.0,5), fontsize=12) 
    plt.xlabel('time [hours]', fontsize=14) # label along x-axes
    plt.ylabel('v [m/s]', fontsize=14) # label along x-axes
    plt.legend()
    plt.grid(True)
    plt.savefig("img/task1b.png") # save plot as png-file
    plt.show() # show plot on screen
