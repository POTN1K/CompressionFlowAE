import numpy as np
import sys
sys.path.append('..')
from SampleFlows.ParentClass import Model

#data = Model.data_reading(re=40, nx=24, nu=2)

def incompressibility(_3x3x2):
    def central_difference(before, after):
        return (after - before)/2
    # return der of u wrt x + der of v wrt y
    return central_difference(_3x3x2[1][0][0], _3x3x2[1][2][0]) \
        + central_difference(_3x3x2[0][1][1], _3x3x2[2][1][1])

def navier_stokes(_3x3x2, f):
    # I use the incompressibale navier-stokes equaiton
    # f should be a vector of 2 values, in x and y
    def central_difference(before, after):
        return (after - before)/2
    
    def central_difference_second(before, at, after):
        # assumed grid size of 1
        return after - 2 * at + before
    # since the grid is uniform but he size is unkwon, density could be anything
    # I assume a grid size of one
    # dynamic pressure
    def q_over_rho_0(vel): # where u and v are the velocity components
        vel = np.array(vel)
        return 0.5*np.dot(vel, vel)
    # in the incompressible navier stokes equation you have p/rho_0 from which p = p_0 + q
    # the dynamic pressure is enough, because the static does not change, therefore its derivativa is 0-> drops out

    # we can calculate this value each time and checl for consistency over the grid by std
    # partial differentitaion wrt time is 0 as we only look at one image

    # convection
    def convection(_3x3x2):
                            # u                            du/dx                                       v                 du/dy
        return np.array([_3x3x2[1][1][0] * central_difference(_3x3x2[1][0][0], _3x3x2[1][2][0]) + _3x3x2[1][1][1] * central_difference(_3x3x2[0][1][0], _3x3x2[2][1][0]),
                            # u                            dv/dx                                       v                 dv/dy
                        _3x3x2[1][1][0] * central_difference(_3x3x2[1][0][1], _3x3x2[1][2][1]) + _3x3x2[1][1][1] * central_difference(_3x3x2[0][1][1], _3x3x2[2][1][1])])
    
    # diffusion # contains kinetic viscosity, which is not given
    def diffusion_over_kinetic_viscosity(_3x3x2):
                                                # d^2 u / dx^2                                                d^2 u /dy^2      
        return np.array([central_difference_second(_3x3x2[1][0][0], _3x3x2[1][1][0], _3x3x2[1][2][0]) + central_difference_second(_3x3x2[0][1][0], _3x3x2[1][1][0], _3x3x2[2][1][0]),
                         #                       d^2 v / dx^2                                                d^2 v /dy^2      
                         central_difference_second(_3x3x2[1][0][1], _3x3x2[1][1][1], _3x3x2[1][2][1]) + central_difference_second(_3x3x2[0][1][1], _3x3x2[1][1][1], _3x3x2[2][1][1])])

    def internal_source(_3x3x2):
                            #                               d q/rho_0 / dx
        return np.array([   -1 * (central_difference(q_over_rho_0(_3x3x2[1][0]), q_over_rho_0(_3x3x2[1][2]))) ,
                            #                               d q/rho_0 / dy
                            -1 * (central_difference(q_over_rho_0(_3x3x2[0][1]), q_over_rho_0(_3x3x2[2][1]))) ])

    # since the equation is a a vector equaiton and we have one unkown, the dynamic viscosity we can verify wheter the equation holds -> return error
    kinetic_viscosity_x = (convection(_3x3x2)[0] - internal_source(_3x3x2)[0] - f[0]) / diffusion_over_kinetic_viscosity(_3x3x2)[0]
    kinetic_viscosity_y = (convection(_3x3x2)[1] - internal_source(_3x3x2)[1] - f[1]) / diffusion_over_kinetic_viscosity(_3x3x2)[1]

    return abs(kinetic_viscosity_x - kinetic_viscosity_y)
