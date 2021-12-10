# -*- coding: utf-8 -*-

__author__ = "Giovani Hidalgo Ceotto, Oscar Mauricio Prada Ramirez, edited by Matthew Anderson Hendricks"
__copyright__ = "Copyright 20XX, Projeto Jupiter"
__license__ = "MIT"

import re
import math
import bisect
import warnings
import time
from datetime import datetime, timedelta
from inspect import signature, getsourcelines
from collections import namedtuple

import numpy as np
from scipy import integrate
from scipy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from .Function import Function


class HybridMotor:
    """Class to specify characteristics and useful operations for solid
    motors.

    Attributes
    ----------

        Geometrical attributes:
        Motor.nozzleRadius : float
            Radius of motor nozzle outlet in meters.
        Motor.throatRadius : float
            Radius of motor nozzle throat in meters.
        Motor.grainNumber : int
            Number of solid grains.
        Motor.LiquidgrainNumber : int
            Number of liquid grains.
        Motor.grainSeparation : float
            Distance between two grains in meters.
        Motor.grainDensity : float
            Density of each solid grain in kg/meters cubed.
        Motor.LiquidgrainDensity: float
            Density of each liquid grain in kg/meters cubed.
        Motor.grainOuterRadius : float
            Outer radius of each grain in meters.
        Motor.LiquidgrainOuterRadius: float
            Outer radius of each liquid grain in meters.
        Motor.grainInitialInnerRadius : float
            Initial inner radius of each grain in meters.
        Motor.grainInitialHeight : float
            Initial height of each grain in meters.
        Motor.LiquidgrainInitialHeight: float
            Initial height of each liquid grain in meters.
        Motor.grainInitialVolume : float
            Initial volume of each grain in meters cubed.
        Motor.LiquidgrainInitialVolume: float
            Initial volume of each liquid grain in meters cubed.
        Motor.grainInnerRadius : Function
            Inner radius of each grain in meters as a function of time.
        # liquid grain inner radius is zero so will be ignored
        Motor.grainHeight : Function
            Height of each solid grain in meters as a function of time.
        Motor.LiquidgrainHeight: Function
            Height of each liquid grain in meters as a function of time.

        Mass and moment of inertia attributes:
        Motor.grainInitialMass : float
            Initial mass of each grain in kg.
        Motor.propellantInitialMass : float
            Total propellant initial mass in kg.
        Motor.mass : Function
            Propellant total mass in kg as a function of time.
        Motor.massDot : Function
            Time derivative of propellant total mass in kg/s as a function
            of time.
        Motor.inertiaI : Function
            Propellant moment of inertia in kg*meter^2 with respect to axis
            perpendicular to axis of cylindrical symmetry of each grain,
            given as a function of time.
        Motor.inertiaIDot : Function
            Time derivative of inertiaI given in kg*meter^2/s as a function
            of time.
        Motor.inertiaZ : Function
            Propellant moment of inertia in kg*meter^2 with respect to axis of
            cylindrical symmetry of each grain, given as a function of time.
        Motor.inertiaDot : Function
            Time derivative of inertiaZ given in kg*meter^2/s as a function
            of time.

        Thrust and burn attributes:
        Motor.thrust : Function
            Motor thrust force, in Newtons, as a function of time.
        Motor.totalImpulse : float
            Total impulse of the thrust curve in N*s.
        Motor.maxThrust : float
            Maximum thrust value of the given thrust curve, in N.
        Motor.maxThrustTime : float
            Time, in seconds, in which the maximum thrust value is achieved.
        Motor.averageThrust : float
            Average thrust of the motor, given in N.
        Motor.burnOutTime : float
            Total motor burn out time, in seconds. Must include delay time
            when the motor takes time to ignite. Also seen as time to end thrust
            curve.
        Motor.exhaustVelocity : float
            Propulsion gases exhaust velocity, assumed constant, in m/s.
        Motor.burnArea : Function
            Total burn area considering all grains, made out of inner
            cylindrical burn area and grain top and bottom faces. Expressed
            in meters squared as a function of time.
        Motor.Kn : Function
            Motor Kn as a function of time. Defined as burnArea divided by
            nozzle throat cross sectional area. Has no units.
        Motor.burnRate : Function
            Propellant burn rate in meter/second as a function of time.
        Motor.interpolate : string
            Method of interpolation used in case thrust curve is given
            by data set in .csv or .eng, or as an array. Options are 'spline'
            'akima' and 'linear'. Default is "linear".
    """

    def __init__(
        self,
        thrustSource,
        massSource,
        LiquidmassSource,
        burnOut,
        grainNumber,
        grainDensity,
        LiquidgrainDensity,
        grainOuterRadius,
        LiquidgrainOuterRadius,
        grainInitialInnerRadius,
        grainInitialHeight,
        LiquidgrainInitialHeight,
        tankInertiaI,
        tankInertiaZ,
        LiquidgrainNumber=1,
        grainSeparation=0,
        LiquidgrainSeparation=0,
        nozzleRadius=0.0335,
        throatRadius=0.0114,
        reshapeThrustCurve=False,
        interpolationMethod="linear",
    ):
        """Initialize Motor class, process thrust curve and geometrical
        parameters and store results.

        Parameters
        ----------
        thrustSource : int, float, callable, string, array
            Motor's thrust curve. Can be given as an int or float, in which
            case the thrust will be considered constant in time. It can
            also be given as a callable function, whose argument is time in
            seconds and returns the thrust supplied by the motor in the
            instant. If a string is given, it must point to a .csv or .eng file.
            The .csv file shall contain no headers and the first column must
            specify time in seconds, while the second column specifies thrust.
            Arrays may also be specified, following rules set by the class
            Function. See help(Function). Thrust units are Newtons.
        massSource : int, float, callable, string, array
            Motor's solid mass curve. Can be given as an int or float, in which
            case the thrust will be considered constant in time. It can
            also be given as a callable function, whose argument is time in
            seconds and returns the thrust supplied by the motor in the
            instant. If a string is given, it must point to a .csv or .eng file.
            The .csv file shall contain no headers and the first column must
            specify time in seconds, while the second column specifies thrust.
            Arrays may also be specified, following rules set by the class
            Function. See help(Function). Mass units are in Kgs.
        LiquidmassSource : int, float, callable, string, array
            Motor's liquid mass curve. Can be given as an int or float, in which
            case the thrust will be considered constant in time. It can
            also be given as a callable function, whose argument is time in
            seconds and returns the thrust supplied by the motor in the
            instant. If a string is given, it must point to a .csv or .eng file.
            The .csv file shall contain no headers and the first column must
            specify time in seconds, while the second column specifies thrust.
            Arrays may also be specified, following rules set by the class
            Function. See help(Function). Mass units are in Kgs.
        burnOut : int, float
            Motor burn out time in seconds.
        grainNumber : int
            Number of solid grains
        LiquidgrainNumber: int
            Number of liquid grains
        grainDensity : int, float
            Solid grain density in kg/m3.
        LiquidgrainDensity: int, float
            Liquid grain density in kg/m3.
        grainOuterRadius : int, float
            Solid grain outer radius in meters.
        grainInitialInnerRadius : int, float
            Solid grain initial inner radius in meters.
        grainInitialHeight : int, float
            Solid grain initial height in meters.
        grainSeparation : int, float, optional
            Distance between grains (in this implementation
            it is the distance between the solid
            grain and the center of propellant mass), in meters. Default is 0.
        LiquidgrainSeparation : int, float, optional
            Distance between grains(
                In this implementation it is the distance between the liquid gain 
                and the propellant center of mass
            ), in meters. Default is 0.
        nozzleRadius : int, float, optional
            Motor's nozzle outlet radius in meters. Used to calculate Kn curve.
            Optional if the Kn curve is not interesting. Its value does not impact
            trajectory simulation.
        throatRadius : int, float, optional
            Motor's nozzle throat radius in meters. Its value has very low
            impact in trajectory simulation, only useful to analyze
            dynamic instabilities, therefore it is optional.
        reshapeThrustCurve : boolean, tuple, optional
            If False, the original thrust curve supplied is not altered. If a
            tuple is given, whose first parameter is a new burn out time and
            whose second parameter is a new total impulse in Ns, the thrust
            curve is reshaped to match the new specifications. May be useful
            for motors whose thrust curve shape is expected to remain similar
            in case the impulse and burn time varies slightly. Default is
            False.
        interpolationMethod : string, optional
            Method of interpolation to be used in case thrust curve is given
            by data set in .csv or .eng, or as an array. Options are 'spline'
            'akima' and 'linear'. Default is "linear".

        Returns
        -------
        None
        """
        # Thrust parameters
        self.interpolate = interpolationMethod
        self.burnOutTime = burnOut

        # Check if thrustSource is csv, eng, function or other
        if isinstance(thrustSource, str):
            # Determine if csv or eng
            if thrustSource[-3:] == "eng":
                # Import content
                comments, desc, points = self.importEng(thrustSource)
                # Process description and points
                # diameter = float(desc[1])/1000
                # height = float(desc[2])/1000
                # mass = float(desc[4])
                # nozzleRadius = diameter/4
                # throatRadius = diameter/8
                # grainNumber = grainnumber
                # grainVolume = height*np.pi*((diameter/2)**2 -(diameter/4)**2)
                # grainDensity = mass/grainVolume
                # grainOuterRadius = diameter/2
                # grainInitialInnerRadius = diameter/4
                # grainInitialHeight = height
                thrustSource = points
                self.burnOutTime = points[-1][0]

        # Create thrust function
        self.thrust = Function(
            thrustSource, "Time (s)", "Thrust (N)", self.interpolate, "zero"
        )

        self.mass = Function(
            massSource, "Time (s)", "Mass (kg)", self.interpolate, "zero"
        )
        self.Liquidmass = Function(
            LiquidmassSource, "Time (s)", "Liquid Mass (kg)", self.interpolate, "zero"
        )

        if callable(thrustSource) or isinstance(thrustSource, (int, float)):
            self.thrust.setDiscrete(0, burnOut, 50, self.interpolate, "zero")

        # Reshape curve and calculate impulse
        if reshapeThrustCurve:
            self.reshapeThrustCurve(*reshapeThrustCurve)
        else:
            self.evaluateTotalImpulse()

        # Define motor attributes
        # Grain and nozzle parameters
        self.nozzleRadius = nozzleRadius
        self.throatRadius = throatRadius
        self.grainNumber = grainNumber
        self.LiquidgrainNumber = LiquidgrainNumber
        self.grainSeparation = grainSeparation
        self.LiquidgrainSeparation = LiquidgrainSeparation
        self.grainDensity = grainDensity
        self.LiquidgrainDensity = LiquidgrainDensity
        self.grainOuterRadius = grainOuterRadius
        self.LiquidgrainOuterRadius = LiquidgrainOuterRadius
        self.grainInitialInnerRadius = grainInitialInnerRadius
        self.grainInitialHeight = grainInitialHeight
        self.LiquidgrainInitialHeight = LiquidgrainInitialHeight
        self.tankInertiaI = tankInertiaI
        self.tankInertiaZ = tankInertiaZ
        # Other quantities that will be computed
        self.massDot = None
        self.exhaustVelocity = None
        self.grainInnerRadius = None
        self.grainHeight = None
        self.LiquidgrainHeight = None
        self.burnArea = None
        self.Kn = None
        self.burnRate = None
        self.inertiaI = None
        self.LiquidinertiaI = None
        self.inertiaIDot = None
        self.LiquidinertiaIDot = None
        self.inertiaZ = None
        self.LiquidinertiaZ = None
        self.inertiaZDot = None
        self.LiquidinertiaZDot = None
        self.maxThrust = None
        self.maxThrustTime = None
        self.averageThrust = None

        # Compute uncalculated quantities
        # Thrust information - maximum and average
        self.maxThrust = np.amax(self.thrust.source[:, 1])
        maxThrustIndex = np.argmax(self.thrust.source[:, 1])
        self.maxThrustTime = self.thrust.source[maxThrustIndex, 0]
        self.averageThrust = self.totalImpulse / self.burnOutTime
        # Grains initial geometrical parameters
        self.grainInitialVolume = (
            self.grainInitialHeight
            * np.pi
            * (self.grainOuterRadius ** 2 - self.grainInitialInnerRadius ** 2)
        )
        self.grainInitialMass = self.grainDensity * self.grainInitialVolume
        self.propellantInitialMass = self.grainNumber * self.grainInitialMass
        # Dynamic quantities
        self.evaluateMassDot()
        self.evaluateGeometry()
        self.evaluateLiquidGeometry()
        self.evaluateInertia()
        self.evaluateLiquidInertia()
        self.evaluateTotalInertia()

        self.maxExhaustVelocity = np.amax(self.exhaustVelocity.source[:, 1])

    def reshapeThrustCurve(
        self, burnTime, totalImpulse, oldTotalImpulse=None, startAtZero=True
    ):
        """Transforms the thrust curve supplied by changing its total
        burn time and/or its total impulse, without altering the
        general shape of the curve. May translate the curve so that
        thrust starts at time equals 0, without any delays.

        Parameters
        ----------
        burnTime : float
            New desired burn out time in seconds.
        totalImpulse : float
            New desired total impulse.
        oldTotalImpulse : float, optional
            Specify the total impulse of the given thrust curve,
            overriding the value calculated by numerical integration.
            If left as None, the value calculated by numerical
            integration will be used in order to reshape the curve.
        startAtZero: bool, optional
            If True, trims the initial thrust curve points which
            are 0 Newtons, translating the thrust curve so that
            thrust starts at time equals 0. If False, no translation
            is applied.

        Returns
        -------
        None
        """
        # Retrieve current thrust curve data points
        timeArray = self.thrust.source[:, 0]
        thrustArray = self.thrust.source[:, 1]
        # Move start to time = 0
        if startAtZero and timeArray[0] != 0:
            timeArray = timeArray - timeArray[0]

        # Reshape time - set burn time to burnTime
        self.thrust.source[:, 0] = (burnTime / timeArray[-1]) * timeArray
        self.burnOutTime = burnTime
        self.thrust.setInterpolation(self.interpolate)

        # Reshape thrust - set total impulse
        if oldTotalImpulse is None:
            oldTotalImpulse = self.evaluateTotalImpulse()
        self.thrust.source[:, 1] = (totalImpulse / oldTotalImpulse) * thrustArray
        self.thrust.setInterpolation(self.interpolate)

        # Store total impulse
        self.totalImpulse = totalImpulse

        # Return reshaped curve
        return self.thrust

    def evaluateTotalImpulse(self):
        """Calculates and returns total impulse by numerical
        integration of the thrust curve in SI units. The value is
        also stored in self.totalImpulse.

        Parameters
        ----------
        None

        Returns
        -------
        self.totalImpulse : float
            Motor total impulse in Ns.
        """
        # Calculate total impulse
        self.totalImpulse = self.thrust.integral(0, self.burnOutTime)

        # Return total impulse
        return self.totalImpulse

    def evaluateMassDot(self):
        """Calculates and returns the time derivative of solid propellant
        mass by assuming nonconstant exhaust velocity.The values are 
        obtained by differentiating the mass curve. The
        result is a function of time, object of the Function class,
        which is stored in self.massDot.

        Parameters
        ----------
        None

        Returns
        -------
        self.massDot : Function
            Time derivative of total propellant mas as a function
            of time.
        """
        # Retrieve mass dot curve data
        t = self.mass.source[:, 0]
        y = self.mass.source[:, 1]

        # Set initial conditions
        T = [0]
        ydot = [(y[1]-y[0])/(t[1]-t[0])]

        # Solve for each time point
        for i in range(1, len(t)-1):
            T += [t[i]]
            ydot += [(y[i]-y[i-1])/(t[i]-t[i-1])]


        # Create mass dot Function
        self.massDot = Function(
            np.concatenate(([T], [ydot])).transpose(),
            "Time (s)",
            "Propellant Total Mass Rate (kg s-1)",
            self.interpolate,
            "constant",
        )
        self.massDot.setOutputs("Mass Dot (kg/s)")
        self.massDot.setExtrapolation("zero")

        # Return Function
        return self.massDot

    def evaluateLiquidMassDot(self):
        """Calculates and returns the time derivative of solid propellant
        mass by assuming nonconstant exhaust velocity.The values are 
        obtained by differentiating the mass curve. The
        result is a function of time, object of the Function class,
        which is stored in self.massDot.

        Parameters
        ----------
        None

        Returns
        -------
        self.massDot : Function
            Time derivative of total propellant mas as a function
            of time.
        """
        # Retrieve mass dot curve data
        t = self.Liquidmass.source[:, 0]
        y = self.Liquidmass.source[:, 1]

        # Set initial conditions
        T = [0]
        ydot = [(y[1]-y[0])/(t[1]-t[0])]

        # Solve for each time point
        for i in range(1, len(t)-1):
            T += [t[i]]
            ydot += [(y[i]-y[i-1])/(t[i]-t[i-1])]


        # Create mass dot Function
        self.LiquidmassDot = Function(
            np.concatenate(([T], [ydot])).transpose(),
            "Time (s)",
            "Propellant Total Mass Rate (kg s-1)",
            self.interpolate,
            "constant",
        )
        self.LiquidmassDot.setOutputs("Mass Dot (kg/s)")
        self.LiquidmassDot.setExtrapolation("zero")

        # Return Function
        return self.LiquidmassDot
    
    @property
    def exhaustVelocity(self):
        """Calculates and returns  effective exhaust velocity by assuming it is not
        constant. The formula used is thrust/ derivative of mass. The value is also stored in
        self.exhaustVelocity.

        Parameters
        ----------
        None

        Returns
        -------
        self.exhaustVelocity : float
            Constant gas exhaust velocity of the motor.
        """
        self.exhaustVelocity = self.thrust / (self.massDot + self.LiquidmassDot)
        return self.exhaustVelocity

    @property
    def throatArea(self):
        return np.pi * self.throatRadius ** 2

    def evaluateGeometry(self):
        """Calculates grain inner radius and grain height as a
        function of time by assuming that every propellant mass
        burnt is exhausted. In order to do that, a system of
        differential equations is solved using scipy.integrate.
        odeint. Furthermore, the function calculates burn area,
        burn rate and Kn as a function of time using the previous
        results. All functions are stored as objects of the class
        Function in self.grainInnerRadius, self.grainHeight, self.
        burnArea, self.burnRate and self.Kn.

        Parameters
        ----------
        None

        Returns
        -------
        geometry : list of Functions
            First element is the Function representing the inner
            radius of a grain as a function of time. Second
            argument is the Function representing the height of a
            grain as a function of time.
        """
        # Define initial conditions for integration
        y0 = [self.grainInitialInnerRadius, self.grainInitialHeight]

        # Define time mesh
        t = self.massDot.source[:, 0]

        density = self.grainDensity
        rO = self.grainOuterRadius

        # Define system of differential equations
        def geometryDot(y, t):
            grainMassDot = self.massDot(t) / self.grainNumber
            rI, h = y
            rIDot = (
                -0.5 * grainMassDot / (density * np.pi * (rO ** 2 - rI ** 2 + rI * h))
            )
            hDot = 1.0 * grainMassDot / (density * np.pi * (rO ** 2 - rI ** 2 + rI * h))
            return [rIDot, hDot]

        # Solve the system of differential equations
        sol = integrate.odeint(geometryDot, y0, t)

        # Write down functions for innerRadius and height
        self.grainInnerRadius = Function(
            np.concatenate(([t], [sol[:, 0]])).transpose().tolist(),
            "Time (s)",
            "Grain Inner Radius (m)",
            self.interpolate,
            "constant",
        )
        self.grainHeight = Function(
            np.concatenate(([t], [sol[:, 1]])).transpose().tolist(),
            "Time (s)",
            "Grain Height (m)",
            self.interpolate,
            "constant",
        )

        # Create functions describing burn rate, Kn and burn area
        self.evaluateBurnArea()
        self.evaluateKn()
        self.evaluateBurnRate()

        return [self.grainInnerRadius, self.grainHeight]

    def evaluateLiquidGeometry(self):
        """Calculates grain inner radius and grain height as a
        function of time by assuming that every propellant mass
        burnt is exhausted. In order to do that, a system of
        differential equations is solved using scipy.integrate.
        odeint. Furthermore, the function calculates burn area,
        burn rate and Kn as a function of time using the previous
        results. All functions are stored as objects of the class
        Function in self.grainInnerRadius, self.grainHeight, self.
        burnArea, self.burnRate and self.Kn.

        Parameters
        ----------
        None

        Returns
        -------
        geometry : list of Functions
            First element is the Function representing the inner
            radius of a grain as a function of time. Second
            argument is the Function representing the height of a
            grain as a function of time.
        """
        # Define initial conditions for integration
        y0 = [0, self.LiquidgrainInitialHeight]

        # Define time mesh
        t = self.LiquidmassDot.source[:, 0]

        density = self.LiquidgrainDensity
        rO = self.LiquidgrainOuterRadius

        # Define system of differential equations
        def geometryDot(y, t):
            # assume both grains burn at the same rate for now
            LiquidgrainMassDot = self.LiquidmassDot(t) / self.grainNumber
            rI, h = y
            rIDot = 0 # no inner radius in Liquid Rocket
            hDot = 1.0 * LiquidgrainMassDot / (density * np.pi * (rO ** 2 - rI ** 2 + rI * h))
            return [rIDot, hDot]

        # Solve the system of differential equations
        sol = integrate.odeint(geometryDot, y0, t)

        # Write down functions for innerRadius and height
        self.grainInnerRadius = Function(
            np.concatenate(([t], [sol[:, 0]])).transpose().tolist(),
            "Time (s)",
            "Liquid Grain Inner Radius (m)",
            self.interpolate,
            "constant",
        )
        self.grainHeight = Function(
            np.concatenate(([t], [sol[:, 1]])).transpose().tolist(),
            "Time (s)",
            "Liquid Grain Height (m)",
            self.interpolate,
            "constant",
        )

        # Create functions describing burn rate, Kn and burn area
        self.evaluateLiquidBurnArea()
        self.evaluateKn()
        self.evaluateLiquidBurnRate()

        return [self.grainInnerRadius, self.grainHeight]

    def evaluateBurnArea(self):
        """Calculates the BurnArea of the grain for
        each time. Assuming that the grains are cylindrical
        BATES grains.

        Parameters
        ----------
        None

        Returns
        -------
        burnArea : Function
        Function representing the burn area progression with the time.
        """
        self.burnArea = (
            2
            * np.pi
            * (
                self.grainOuterRadius ** 2
                - self.grainInnerRadius ** 2
                + self.grainInnerRadius * self.grainHeight
            )
            * self.grainNumber
        )
        self.burnArea.setOutputs("Burn Area (m2)")
        return self.burnArea
    
    def evaluateLiquidBurnArea(self):
        """Calculates the BurnArea of the grain for
        each time. Assuming that the grains are cylindrical
        BATES grains.

        Parameters
        ----------
        None

        Returns
        -------
        burnArea : Function
        Function representing the burn area progression with the time.
        """
        self.LiquidburnArea = (
            2
            * np.pi
            * (
                self.LiquidgrainOuterRadius ** 2
            )
            * self.LiquidgrainNumber
        )
        self.LiquidburnArea.setOutputs("Liquid Burn Area (m2)")
        return self.LiquidburnArea

    def evaluateBurnRate(self):
        """Calculates the BurnRate with respect to time.
        This evaluation assumes that it was already
        calculated the massDot, burnArea timeseries.

        Parameters
        ----------
        None

        Returns
        -------
        burnRate : Function
        Rate of progression of the inner radius during the combustion.
        """
        self.burnRate = (-1) * self.massDot / (self.burnArea * self.grainDensity)*2
        self.burnRate.setOutputs("Burn Rate (m/s)")
        return self.burnRate
    
    def evaluateLiquidBurnRate(self):
        """Calculates the BurnRate with respect to time.
        This evaluation assumes that it was already
        calculated the massDot, burnArea timeseries.

        Parameters
        ----------
        None

        Returns
        -------
        burnRate : Function
        Rate of progression of the inner radius during the combustion.
        """
        self.LiquidburnRate = (-1) * self.LiquidmassDot / (self.LiquidburnArea)*2
        self.LiquidburnRate.setOutputs("Burn Rate (m/s)")
        return self.LiquidburnRate

    def evaluateKn(self):
        KnSource = (
            np.concatenate(
                (
                    [self.grainInnerRadius.source[:, 1]],
                    [self.burnArea.source[:, 1] / self.throatArea],
                )
            ).transpose()
        ).tolist()
        self.Kn = Function(
            KnSource,
            "Grain Inner Radius (m)",
            "Kn (m2/m2)",
            self.interpolate,
            "constant",
        )
        return self.Kn

    
    def evaluateInertia(self):
        """Calculates propellant inertia I, relative to directions
        perpendicular to the rocket body axis and its time derivative
        as a function of time. Also calculates propellant inertia Z,
        relative to the axial direction, and its time derivative as a
        function of time. Products of inertia are assumed null due to
        symmetry. The four functions are stored as an object of the
        Function class.

        Parameters
        ----------
        None

        Returns
        -------
        list of Functions
            The first argument is the Function representing inertia I,
            while the second argument is the Function representing
            inertia Z.
        """

        # Inertia I
        # Calculate inertia I for each grain
        grainMass = self.mass / self.grainNumber
        grainMassDot = self.massDot / self.grainNumber
        grainNumber = self.grainNumber
        grainInertiaI = grainMass * (
            (1 / 4) * (self.grainOuterRadius ** 2 + self.grainInnerRadius ** 2)
            + (1 / 12) * self.grainHeight ** 2
        )

        # Calculate each grain's distance d to propellant center of mass
        d = 0

        # Calculate inertia for all grains
        self.inertiaI = grainNumber * grainInertiaI + grainMass * np.sum(d ** 2)
        self.inertiaI.setOutputs("Propellant Inertia I (kg*m2)")

        # Inertia I Dot
        # Calculate each grain's inertia I dot
        grainInertiaIDot = (
            grainMassDot
            * (
                (1 / 4) * (self.grainOuterRadius ** 2 + self.grainInnerRadius ** 2)
                + (1 / 12) * self.grainHeight ** 2
            )
            + grainMass
            * ((1 / 2) * self.grainInnerRadius - (1 / 3) * self.grainHeight)
            * self.burnRate
        )

        # Calculate inertia I dot for all grains
        self.inertiaIDot = grainNumber * grainInertiaIDot + grainMassDot * np.sum(
            d ** 2
        )
        self.inertiaIDot.setOutputs("Propellant Inertia I Dot (kg*m2/s)")

        # Inertia Z
        self.inertiaZ = (
            (1 / 2.0)
            * self.mass
            * (self.grainOuterRadius ** 2 + self.grainInnerRadius ** 2)
        )
        self.inertiaZ.setOutputs("Propellant Inertia Z (kg*m2)")

        # Inertia Z Dot
        self.inertiaZDot = (1 / 2.0) * self.massDot * (
            self.grainOuterRadius ** 2 + self.grainInnerRadius ** 2
        ) + self.mass * self.grainInnerRadius * self.burnRate
        self.inertiaZDot.setOutputs("Propellant Inertia Z Dot (kg*m2/s)")

        return [self.inertiaI, self.inertiaZ]

    def evaluateLiquidInertia(self):
        """Calculates liquid propellant inertia I, relative to directions
        perpendicular to the rocket body axis and its time derivative
        as a function of time. Also calculates propellant inertia Z,
        relative to the axial direction, and its time derivative as a
        function of time. Products of inertia are assumed null due to
        symmetry. The four functions are stored as an object of the
        Function class.

        Parameters
        ----------
        None

        Returns
        -------
        list of Functions
            The first argument is the Function representing inertia I,
            while the second argument is the Function representing
            inertia Z.
        """

        # Inertia I
        # Calculate inertia I for each grain
        LiquidgrainMass = self.mass / (self.grainNumber + self.LiquidgrainNumber)
        LiquidgrainMassDot = self.massDot / (self.grainNumber + self.LiquidgrainNumber)
        LiquidgrainNumber = self.LiquidgrainNumber
        totalgrainNumber = self.LiquidgrainNumber + self.grainNumber
        LiquidgrainInertiaI = LiquidgrainMass * (
            (1 / 4) * (self.LiquidgrainOuterRadius ** 2)
            + (1 / 12) * self.LiquidgrainHeight ** 2
        )

        # Calculate each grain's distance d to propellant center of mass ?
        d = 0

        # Calculate inertia for all grains
        self.LiquidinertiaI = 0 # Assume to be zero for liquid rocket
        self.LiquidinertiaI.setOutputs("Propellant Inertia I (kg*m2)")

        # Inertia I Dot
        # Calculate each grain's inertia I dot
        LiquidgrainInertiaIDot = (
            LiquidgrainMassDot
            * (
                (1 / 4) * (self.LiquidgrainOuterRadius ** 2)
                + (1 / 12) * self.LiquidgrainHeight ** 2
            )
            + LiquidgrainMass
            * (-(1 / 3) * self.LiquidgrainHeight)
            * self.burnRate
        )

        # Calculate inertia I dot for all grains
        self.LiquidinertiaIDot = LiquidgrainNumber * LiquidgrainInertiaI + LiquidgrainMass * np.sum(d ** 2)
        self.LiquidinertiaIDot.setOutputs("Propellant Inertia I Dot (kg*m2/s)")

        # Inertia Z
        self.LiquidinertiaZ = 0 # Assume zero for liquid rocket
        self.LiquidinertiaZ.setOutputs("Propellant Inertia Z (kg*m2)")

        # Inertia Z Dot
        self.LiquidinertiaZDot = (1 / 2.0) * self.LiquidmassDot * (
            self.LiquidgrainOuterRadius ** 2 + self.LiquidgrainInnerRadius ** 2
        )
        self.LiquidinertiaZDot.setOutputs("Propellant Inertia Z Dot (kg*m2/s)")

        return [self.LiquidinertiaI, self.LiquidinertiaZ]

    def evaluateTotalInertia(self):
        """
        Summing up the individual components of the inertia to evaluate the total inertia of the hybrid motor

        Parameters
        ----------
        None

        Returns
        -------
        list of Functions
            The first argument is the Function representing total inertia I,
            while the second argument is the Function representing total
            inertia Z.
        """
        self.totalinertiaI = self.inertiaI + self.LiquidinertiaI + self.tanksinertiaI
        self.totalinertiaZ = self.inertiaZ + self.LiquidinertiaZ + self.tanksinertiaZ
        return [self.totalinertiaI, self.totalinertiaZ]


    def importEng(self, fileName):
        """Read content from .eng file and process it, in order to
        return the comments, description and data points.

        Parameters
        ----------
        fileName : string
            Name of the .eng file. E.g. 'test.eng'.
            Note that the .eng file must not contain the 0 0 point.

        Returns
        -------
        comments : list
            All comments in the .eng file, separated by line in a list. Each
            line is an entry of the list.
        description: list
            Description of the motor. All attributes are returned separated in
            a list. E.g. "F32 24 124 5-10-15 .0377 .0695 RV\n" is return as
            ['F32', '24', '124', '5-10-15', '.0377', '.0695', 'RV\n']
        dataPoints: list
            List of all data points in file. Each data point is an entry in
            the returned list and written as a list of two entries.
        """
        # Intiailize arrays
        comments = []
        description = []
        dataPoints = [[0, 0]]

        # Open and read .eng file
        with open(fileName) as file:
            for line in file:
                if line[0] == ";":
                    # Extract comment
                    comments.append(line)
                else:
                    if description == []:
                        # Extract description
                        description = line.strip().split(" ")
                    else:
                        # Extract thrust curve data points
                        time, thrust = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)
                        dataPoints.append([float(time), float(thrust)])

        # Return all extract content
        return comments, description, dataPoints

    def exportEng(self, fileName, motorName):
        """Exports thrust curve data points and motor description to
        .eng file format. A description of the format can be found
        here: http://www.thrustcurve.org/raspformat.shtml

        Parameters
        ----------
        fileName : string
            Name of the .eng file to be exported. E.g. 'test.eng'
        motorName : string
            Name given to motor. Will appear in the description of the
            .eng file. E.g. 'Mandioca'

        Returns
        -------
        None
        """
        # Open file
        file = open(fileName, "w")

        # Write first line
        file.write(
            motorName
            + " {:3.1f} {:3.1f} 0 {:2.3} {:2.3} RocketPy\n".format(
                2000 * self.grainOuterRadius,
                1000
                * self.grainNumber
                * (self.grainInitialHeight + self.grainSeparation),
                self.propellantInitialMass,
                self.propellantInitialMass,
            )
        )

        # Write thrust curve data points
        for time, thrust in self.thrust.source[1:-1, :]:
            # time, thrust = item
            file.write("{:.4f} {:.3f}\n".format(time, thrust))

        # Write last line
        file.write("{:.4f} {:.3f}\n".format(self.thrust.source[-1, 0], 0))

        # Close file
        file.close()

        return None

    def info(self):
        """Prints out a summary of the data and graphs available about
        the Motor.

        Parameters
        ----------
        None

        Return
        ------
        None
        """
        # Print motor details
        print("\nMotor Details")
        print("Total Burning Time: " + str(self.burnOutTime) + " s")
        print(
            "Total Propellant Mass: "
            + "{:.3f}".format(self.propellantInitialMass)
            + " kg"
        )
        print(
            "Propellant Maximum Exhaust Velocity: "
            + "{:.3f}".format(self.maxExhaustVelocity)
            + " m/s"
        )
        print("Average Thrust: " + "{:.3f}".format(self.averageThrust) + " N")
        print(
            "Maximum Thrust: "
            + str(self.maxThrust)
            + " N at "
            + str(self.maxThrustTime)
            + " s after ignition."
        )
        print("Total Impulse: " + "{:.3f}".format(self.totalImpulse) + " Ns")

        # Show plots
        print("\nPlots")
        self.thrust()

        return None

    def allInfo(self):
        """Prints out all data and graphs available about the Motor.

        Parameters
        ----------
        None

        Return
        ------
        None
        """
        # Print nozzle details
        print("Nozzle Details")
        print("Nozzle Radius: " + str(self.nozzleRadius) + " m")
        print("Nozzle Throat Radius: " + str(self.throatRadius) + " m")

        # Print grain details
        print("\nGrain Details")
        print("Number of Grains: " + str(self.grainNumber))
        print("Grain Spacing: " + str(self.grainSeparation) + " m")
        print("Grain Density: " + str(self.grainDensity) + " kg/m3")
        print("Grain Outer Radius: " + str(self.grainOuterRadius) + " m")
        print("Grain Inner Radius: " + str(self.grainInitialInnerRadius) + " m")
        print("Grain Height: " + str(self.grainInitialHeight) + " m")
        print("Grain Volume: " + "{:.3f}".format(self.grainInitialVolume) + " m3")
        print("Grain Mass: " + "{:.3f}".format(self.grainInitialMass) + " kg")

        # Print motor details
        print("\nMotor Details")
        print("Total Burning Time: " + str(self.burnOutTime) + " s")
        print(
            "Total Propellant Mass: "
            + "{:.3f}".format(self.propellantInitialMass)
            + " kg"
        )
        ## Need to sort out a way to print average exhaust velocity?
        print(
            "Propellant Exhaust Velocity: "
            + "{:.3f}".format(self.exhaustVelocity)
            + " m/s"
        )
        print("Average Thrust: " + "{:.3f}".format(self.averageThrust) + " N")
        print(
            "Maximum Thrust: "
            + str(self.maxThrust)
            + " N at "
            + str(self.maxThrustTime)
            + " s after ignition."
        )
        print("Total Impulse: " + "{:.3f}".format(self.totalImpulse) + " Ns")

        # Show plots
        print("\nPlots")
        self.thrust()
        self.mass()
        self.massDot()
        self.exhaustVelocity()
        self.grainInnerRadius()
        self.grainHeight()
        self.burnRate()
        self.burnArea()
        self.Kn()
        self.inertiaI()
        self.inertiaIDot()
        self.inertiaZ()
        self.inertiaZDot()

        return None
