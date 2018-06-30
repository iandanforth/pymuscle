# Rajagopal and Yong 2013
#
# See https:#simtk-confluence.stanford.edu:8443/display/OpenSim/Design+of+a+Fatigable+Muscle


def forceVelocity(velocity):
    """Given velocity return a weight to apply to force"""
    weight = 1.0
    return weight


def forceForMUZero(slow_twitch_percent: float) -> float:
    force = 1.0
    return force


def activationFactor(
        slow_twitch_percent: float,
        activation: list) -> list:
    # Allowable range is 0 to 1
    assert 0.0 <= activation <= 1.0

    activation_factor = 1.0
    return activation_factor


def fatigueFactor(fatigue_state: list) -> list:
    fatigue_factor = 1.0
    if fatigue_state == 'fatigued':
        fatigue_factor = 0.0
    elif fatigue_state == 'rested':
        fatigue_factor = 1.0

    return fatigue_factor


def lengthFactor(length: float) -> float:
    """
    The possible force output varies with the length of the
    muscle.
    """
    return 1.0


def velocityFactor(velocity: float) -> float:
    """
    The possible force output varies with the velocity of
    contraction of the muscle
    """
    return 1.0


def passiveForce(length: float) -> float:
    """
    Material properties of the muscle will impart additional
    force directly dependent on the length (stretch) of the
    muscle
    """
    return 1.0


def muscleForce(
        activation,
        fatigue_state,
        length,
        velocity,
        slow_twitch_percent=5.0):

    force = forceForMUZero(slow_twitch_percent)
    # Add in activation factor
    force *= activationFactor(
        slow_twitch_percent,
        activation
    )
    # Add in the fatigue factor
    force *= fatigueFactor(fatigue_state)

    # Add in length and velocity factors
    force *= lengthFactor(length)
    force *= velocityFactor(velocity)

    # Add in passive contribution from muscle stretch
    force += passiveForce(length)

    return force


def updateFatigue(fatigue_state):

    # mu_a - active motor units
    # mu_f - fatigued motor units
    # mu_r - resting motor units

    # F - fatigue rate - 0.1778
    # R - recovery rate - 0.0041
    # tau - recruitment time constant - 0.3812
    # u - excitation
    # a - activation

    # change in active motor units with regards to time
    # is equal to
    # resting motor units times
    # contraction at time t divided by
    # the recruitment time constant
    # minus the fatigue rate
    # times active motor units
    pass


class FatigableMuscle(object):

    def __init__(self):

        self.excitation = 0.0
        self.targetActivation = 0.0
        self.activeMotorUnits = 0.0
        self.restingMotorUnits = 1.0

    def setTargetActivation(self, activation):
        self.targetActivation = activation

    def setExcitation(self, excitation):
        self.excitation = excitation

    def calculateRecruitmentOfResting(self) -> float:
        """
        Compute the function c(t) which represents the recruitment of resting 
        motor units to active motor units.
        """
        recruitmentOfResting = 0.0  # float
        excitation = self.excitation  # float
        activation = self.targetActivation  # float
        activeMotorUnits = self.activeMotorUnits  # float
        restingMotorUnits = self.restingMotorUnits  # float
        # Need more power, have more than adequate  reserves
        if ((excitation >= activation) and ((excitation - activeMotorUnits) < restingMotorUnits)):
            recruitmentOfResting = excitation - activeMotorUnits
        # Need more power, insufficient reserves
        elif ((excitation >= activation) and ((excitation - activeMotorUnits) >= restingMotorUnits)):
            recruitmentOfResting = restingMotorUnits
        # Need less power
        else:
            recruitmentOfResting = excitation - activeMotorUnits
        return recruitmentOfResting

    def computeStateVariableDerivatives(s: SimTK.State) -> list:
        # vector of the derivatives to be returned
        derivs(getNumStateVariables(), SimTK.NaN)  # list
        nd = derivs.size()  # int

        # assert nd == 6, "FatigableMuscle: Expected 5 state variables"
        #               " but encountered  %f.", nd

        # ----- MODIFIED SIR MODEL WITH TIME CONSTANT IN RECRUITMENT CURVE C(T)
        # compute the rates at which motor units are converted to/from active
        # and fatigued states based on an SIR model
        restingMotorUnits = getRestingMotorUnits(s)  # float
        recruitmentOfResting = calculateRecruitmentOfResting(s)  # float
        recruitmentFromRestingTimeConstant = getRecruitmentFromRestingTimeConstant()  # float (tau)
        fatigueFactor = getFatigueFactor()  # float
        activeMotorUnits = getActiveMotorUnits(s)  # float
        fatiguedMotorUnits = getFatiguedMotorUnits(s)  # float
        recoveryFactor = getRecoveryFactor()  # float
        activeMotorUnitsDeriv = restingMotorUnits * (
            (recruitmentOfResting / recruitmentFromRestingTimeConstant)
            - (fatigueFactor * activeMotorUnits)
        )
        fatigueMotorUnitsDeriv = (
            fatigueFactor
            * activeMotorUnits
            * restingMotorUnits
            - (recoveryFactor * fatiguedMotorUnits)
        )
        restingMotorUnitsDeriv = (
            (recoveryFactor * fatiguedMotorUnits)
            - (restingMotorUnits
                * (recruitmentOfResting / recruitmentFromRestingTimeConstant))
        )

        # Compute the target activation rate based on the given activation model
        MuscleFirstOrderActivationDynamicModel \
        = get_MuscleFirstOrderActivationDynamicModel()
        excitation = getExcitation(s)  # float
        # use the activation dynamics model to calculate the target activation
        targetActivation = actMdl.clampActivation(getTargetActivation(s))  # float
        targetActivationRate = actMdl.calcDerivative(targetActivation, excitation)  # float

        # specify the activation derivative based on the amount of active motor
        # units and the rate at which motor units are becoming active.
        # we assume that the actual activation = Ma*a       then,
        # activationRate = dMa/dt*a + Ma*da/dt  where a is the target activation
        activationRate = (
            calculateActivationFactorDeriv(s) * targetActivationRate * (1.0 - getFatiguedMotorUnits(s))
            - calculateActivationFactor(s)*fatigueMotorUnitsDeriv
        )

        # COMPLIANT TENDON
        # set the actual activation rate of the muscle to the fatigued one
        derivs[0] = activationRate
        # fiber length derivative
        derivs[1] = getFiberVelocity(s)
        # the target activation derivative
        derivs[2] = targetActivationRate
        derivs[3] = activeMotorUnitsDeriv
        derivs[4] = fatigueMotorUnitsDeriv
        derivs[5] = restingMotorUnitsDeriv

        # RIGID TENDON (fiber length is no longer a state)
        # set the actual activation rate of the muscle to the fatigued one
        derivs[0] = activationRate
        # the target activation derivative
        derivs[1] = targetActivationRate
        derivs[2] = activeMotorUnitsDeriv
        derivs[3] = fatigueMotorUnitsDeriv
        derivs[4] = restingMotorUnitsDeriv
        # cache the results for fast access by reporting, etc...
        setTargetActivationDeriv(s, targetActivationRate)
        setActiveMotorUnitsDeriv(s, activeMotorUnitsDeriv)
        setFatiguedMotorUnitsDeriv(s, fatigueMotorUnitsDeriv)
        setRestingMotorUnitsDeriv(s, restingMotorUnitsDeriv)
        return derivs
