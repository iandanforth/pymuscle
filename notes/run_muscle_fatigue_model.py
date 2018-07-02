from __future__ import print_function
from __future__ import division
import os
import sys
import argparse
import muscle_fatigue
from yaml import load
from pprint import pprint
from copy import copy
import numpy as np

class Params(object):
    def __init__(self, paramsDict):
        self.__dict__.update(paramsDict)

def main(arguments):
    config = None
    if arguments.config:
        config = load(arguments.config)
    else:
        print("No configuration file provided, using default config ...")
        with open('default_config.yaml') as fh:
            config = load(fh)

    if arguments.verbose:
        print("Parameters for run '%s':" % config["run_name"])
        pprint(config["parameters"])

    # Create isotonic data
    p = Params(config["parameters"])
    fthsamp = p.fthtime * p.samprate
    fth = np.zeros(fthsamp) + p.fthscale

    # Create Ramp Plateau Data
    # NOTE: SKIPPED - TODO come back to this

    # Calculations from the Fuglevand, Winter & Patla (1993) Model

    # NOTE: Ommitted lines 86/87 from direct port as they appear to do nothing

    # Recruitment Threshold Excitation (thr) for all neurons TODO: rename
    # This varies across the population
    thr = np.arange(p.nu, dtype='float64')
    n = np.arange(p.nu, dtype='float64')
    # this was modified from Fuglevand et al (1993) RTE(i) equation (1)
    # as that did not create the exact range of RTEs (ie. 'r') entered
    # Scaling factor of recruitment thresholds across neuron population
    b = np.log(p.r + (1 - p.mthr)) / (p.nu - 1)

    # Log thresholds
    thr *= b
    # Base thresholds
    thr = np.exp(thr)
    # Apply gain parameter to thresholds
    thr *= p.a
    # Shift all thresholds by minimum threshold
    thr -= 1 - p.mthr

    # Peak Firing Rate (frp) TODO: rename
    # This curve starts high with the first recruited motor unit and then trends down
    # modified from Fuglevand et al (1993) PFR equation (5) to remove thr(1)
    # before ratio
    firing_rate_diff = p.pfr1 - p.pfrL
    frp = p.pfr1 - ((thr - thr[0]) / (p.r - thr[0]) * firing_rate_diff)
    maxex = thr[p.nu -1] + (p.pfrL - p.minfr) / p.mir # maximum excitation
    maxact = int(round(maxex * p.res)) # max excitation * resolution
    ulr = (thr[p.nu -1] * 100) / maxex # recruitment threshold (%) of last motor unit

    # Calculation of the rested motor unit twitch properties (these will
    # change with fatigue)

    # Firing Rates for each MU with increased excitation (act)
    # Pre-calculate firing rates for each motor unit for each step of activation level
    # Step size is inversely proportional to the resolution parameter

    mufr = np.zeros((p.nu, maxact))
    # TODO: Convert to masked vector operations
    # act_range = np.arange(1, maxact + 1, dtype='float64')
    # # activation increment
    # print(maxex)
    # act_increment = 1 / p.res
    # print(act_increment)
    # act_range *= act_increment
    # threshold = 22.0
    # act_range -= 22
    # act_range *= p.mir
    # act_range += p.minfr
    for mu in range(p.nu):
        for act in range(maxact):
            acti = (act + 1) / p.res
            if acti >= thr[mu]:
                mufr[mu, act] = p.mir * (acti - thr[mu]) + p.minfr
                if mufr[mu, act] > frp[mu]:
                    mufr[mu, act] = frp[mu]
            else:
                mufr[mu, act] = 0.0

    k = np.arange(maxact, dtype='float64') # range of excitation levels

    # Twitch peak force (peak_twitch_forces)
    # this was modified from Fuglevand et al (1993) P(i) equation (13)
    # as that didn't create the exact range of twitch tensions (ie. 'rp') entered
    b = np.log(p.rp) / (p.nu - 1)
    peak_twitch_forces = np.exp((b * n))

    # Twitch contraction times
    c = np.log(p.rp) / np.log(p.rt)
    contraction_times = np.zeros(p.nu)
    # assigns contraction times to each motor unit
    for mu in range(p.nu):
        # TODO: More vectorizing
        # one = 1 / peak_twitch_forces[mu]
        # print(one)
        # two = one ** (1 / c)
        # print(two)
        # three = np.dot(p.tL, two)
        # print(three)
        contraction_times[mu] = p.tL * ((1 / peak_twitch_forces[mu]) ** (1 / c))

    # Normalized motor unit firing rates (nmufr) with increased excitation (act)
    nmufr = np.zeros((p.nu, maxact))
    for mu in range(p.nu):
        for act in range(maxact):
            nmufr[mu, act] = contraction_times[mu] * (mufr[mu, act] / 1000)

    # Motor unit force, relative to full fusion with increasing excitation
    # (Pr in other versions)
    # based on Figure 2 of Fuglevand et al (1993)
    sPr = 1 - np.exp((-2 * (0.4 ** 3)))
    relative_mu_forces = np.zeros((p.nu, maxact))
    for mu in range(p.nu):
        for act in range(maxact):
            # linear portion of curve
            # relative_mu_forces = MU force relative to rest 100% max excitation of 67
            if nmufr[mu, act] <= 0.4:
                relative_mu_forces[mu, act] = sPr * (nmufr[mu, act] / 0.4)
            else:
                relative_mu_forces[mu, act] = 1 - np.exp((-2 * (nmufr[mu, act] ** 3)))

    # Motor unit force with increased excitation
    mu_forces = np.zeros((p.nu, maxact))
    for mu in range(p.nu):
        for act in range(maxact):
            mu_forces[mu, act] = relative_mu_forces[mu, act] * peak_twitch_forces[mu]

    total_forces = np.sum(mu_forces, 0) # sum of forces across MUs for each excitation (dim 0)
    max_force = total_forces[-1]

    # Total force across all motor units when rested
    forces_now = np.zeros((p.nu, fthsamp))
    forces_now[:, 0] = peak_twitch_forces

    # Calculation of Fatigue Parameters (recovery currently set to zero in
    # this version)

    # note, if rp = 100 & fat = 180, there will be a 100 x 180 = 1800-fold difference in
    # the absolute fatigue of the highest threshold vs the lowest threshold.
    # The highest threshold MU will only achieve ~57# of its maximum (at 25 Hz), so the actual range of fatigue
    # rates is 1800 x 0.57 = 1026

    # fatigue rate for each motor unit
    fatigue_scale = np.log(p.fat) / (p.nu - 1)
    mu_fatigue_rates = np.exp(n * fatigue_scale)

    fatigues = np.zeros(p.nu)
    for mu in range(p.nu):
        fatigues[mu] = peak_twitch_forces[mu] * (mu_fatigue_rates[mu] * (p.FatFac / p.fat))

    # the only variable is the relative force: relative_mu_forces(mu, act), so this part is
    # calculated once here

    # Establishing the rested excitation required for each target load level
    # from 1 - 100 percent of maximum possible force
    # (if 0.1# resolution, then 0.1# to 100#)
    start_act = np.zeros(100)
    # TODO: LIKELY EXTREMELY SLOW TO LOOP 100 times over maxact
    for i, force in enumerate(range(1, 101)):
        # excitation will never be lower than that needed at rest for a given force
        # so it speeds the search up by starting at this excitation
        start_act[i] = 0
        for act in range(maxact):
            percent_of_max_force = (total_forces[act] / max_force) * 100
            if percent_of_max_force < force:
                start_act[i] = act
            else:
                break # Stop searching once value is found

    force_change_curves = np.zeros((p.nu, maxact))
    for act in range(maxact):
        for mu in range(p.nu):
            # Just used for graphical display
            fatigued_force = fatigues[mu] * relative_mu_forces[mu, act]
            force_change_curves[mu, act] = fatigued_force * peak_twitch_forces[mu]

    print('Start of fatigue analysis ...')

    # Moving through force time-history and determing the excitation required
    # to meet the target force at each time

    TmuPinstant = np.zeros((p.nu, fthsamp))
    m = np.zeros(fthsamp)
    mufrFAT = np.zeros((p.nu, fthsamp))
    ctFAT = np.zeros((p.nu, fthsamp))
    ctREL = np.zeros((p.nu, fthsamp))
    nmufrFAT = np.zeros((p.nu, fthsamp))
    PrFAT = np.zeros((p.nu, fthsamp))
    muPt = np.zeros((p.nu, fthsamp))
    TPt = np.zeros(fthsamp)
    Ptarget = np.zeros(fthsamp)
    Tact = np.zeros(fthsamp)
    Pchange = np.zeros((p.nu, fthsamp))
    TPtMAX = np.zeros(fthsamp)
    muPtMAX = np.zeros((p.nu, fthsamp))
    mu_on = np.zeros(p.nu)
    adaptFR = np.zeros((p.nu, fthsamp))
    recruitment_duration = np.zeros(p.nu)
    act_temp = np.zeros((fthsamp, maxact))
    muPna = np.zeros((p.nu, fthsamp))
    muForceCapacityRel = np.zeros((p.nu, fthsamp))
    timer = 0

    for i in range(fthsamp):
        # show a timer value every 15 seconds
        if i % (15 * p.samprate) == 0:
            current = i / p.samprate
            print(current)

        # used to start at the minimum possible excitation (lowest it can be is 1)
        # so start with excitation needed for fth(i) when rested (won't be lower than this)
        force = int(round(fth[i] * 100)) + 1
        if force > 100:
            force = 100
        s = start_act[force] - (5 * p.res) # starts a little below the minimum
        if s < 1:
            s = 1

        act_hop = round(maxact / p.hop) # resets 'act_hop' to larger value for new sample
        # TODO: This is funky and led to an off-by-one error, rethink how to get this index
        print("S", s)
        act = int(s) - 1 # start at lowest value then start jumping by 'act_hop'
        # this starts at the mimimum (s) then searches for excitation required to meet the target
        for a in range(maxact):
            act_temp[i, a] = act
            for mu in range(p.nu):
                # MU firing rate adaptation modified from Revill & Fuglevand (2011)
                # this was modified to directly calculate the firing rate adaption, as 1 unit change in excitation causes 1 unit change in firing rate
                # scaled to the mu threshold (highest adaptation for hightest
                # threshold mu)
                if mu_on[mu] > 0:
                    recruitment_duration[mu] = (i - mu_on[mu] + 1) / p.samprate # duration since mu was recruited at mu_on

                if recruitment_duration[mu] < 0:
                    recruitment_duration[mu] = 0 # ??? how do we get here?

                # Firing rate adaptation
                # TODO: Give temp variables real names
                temp_one = ( (thr[mu] - 1) / (thr[-1] - 1) ) * p.adaptSF
                temp_two = temp_one * (mufr[mu, act] - p.minfr + 2)
                temp_three = temp_two * (1 - np.exp(-1 * (recruitment_duration[mu]) / p.tau))
                adaptFR[mu, i] = temp_three
                if adaptFR[mu, i] < 0:
                    adaptFR[mu, i] = 0

                mufrFAT[mu, i] = mufr[mu, act] - adaptFR[mu, i] # adapted motor unit firing rate based on time since recruitment
                mufr_max = mufr[mu, -1] - adaptFR[mu, i] # adapted firing rate at max excitation
                ctFAT[mu, i] = contraction_times[mu] * (1 + (p.ctSF * (1 - forces_now[mu, i] / peak_twitch_forces[mu])))

                ctREL[mu, i] = ctFAT[mu, i] / contraction_times[mu]
                nmufrFAT[mu, i] = ctFAT[mu, i] * (mufrFAT[mu, i] / 1000)  # adapted normalized Stimulus Rate (CT * FR)
                nmufrMAX = ctFAT[mu, i] * (mufr_max / 1000) # normalized firing rate at max excitation

                if nmufrFAT[mu, i] <= 0.4: # fusion level at adapted firing rate
                    PrFAT[mu, i] = (nmufrFAT[mu, i] / 0.4) * sPr # linear portion of curve
                if nmufrFAT[mu, i] > 0.4:
                    PrFAT[mu, i] = 1 - np.exp( -2 * (nmufrFAT[mu, i] ** 3))

                muPt[mu, i] = PrFAT[mu, i] * forces_now[mu, i]
                if nmufrMAX <= 0.4:
                    PrMAX = (nmufrMAX / 0.4) * sPr
                else:
                    PrMAX = 1 - np.exp(-2 * (nmufrMAX ** 3))
                muPtMAX[mu, i] = PrMAX * forces_now[mu, i]

            # total sum of MU forces at the current time (TPt)
            TPt[i] = np.sum(muPt[:, i])
            TPtMAX[i] = np.sum(muPtMAX[:, i]) / max_force
            # used to speed up the search for the right excitation to meet the current target
            if (TPt[i] < fth[i]) and (act == maxact):
                break
            if TPt[i] < fth[i]:
                act = act + act_hop
                if act > maxact:
                    act = copy(maxact)
            if TPt[i] >= fth[i] and act_hop == 1:
                break # stop searching as the correct excitation is found
            if TPt[i] >= fth[i] and act_hop > 1:
                # if the last large jump was too much, it goes back and starts increasing by 1
                act = act - (act_hop - 1)
                if act < 1:
                    act = 1
                act_hop = 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-c', '--config', help="Configuration file", type=argparse.FileType('r'))
    parser.add_argument('-v', '--verbose', help="Verbose output", action='store_true')
    args = parser.parse_args()
    sys.exit(main(args))
