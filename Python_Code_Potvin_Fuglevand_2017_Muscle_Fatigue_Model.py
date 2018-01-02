#! /usr/bin/env python
from __future__ import division
from __future__ import print_function
import numpy as np
from smop.core import *
"""
Motor Unit Based Muscle Fatigue Model by Jim Potvin & Andrew Fuglevand
front end (rested size-principle) based on Fuglevand, Winter & Patla (1993)
last updated 2017-05-28 by Jim Potvin

Python port by Ian Danforth
"""

# Model input parameters

nu = 3            # number of neurons (ie. motor units) in the modeled pool ("n")
samprate = 2       # sample rate (10 Hz is suggested)
res = 2           # resolution of activations (set = 10 for 0.1 activation resolution, 100 for 0.01)
# allows for hopping through activations to more rapidly find that which
# meets the threshold (10 means every 1/10th of maxact)
hop = 20
r = 50              # range of recruitment thresholds (30 or 50)
fat = 180           # range of fatigue rates across the motor units (300 best)
FatFac = 0.0225     # fatigue factor (FF/S) percent of peak force of MU per second
tau = 22            # 22 based on Revill & Fuglevand (2011)
adaptSF = 0.67      # 0.67 from Revill & Fuglevand (2011)
ctSF = 0.379        # 0.379 based on Shields et al (1997)
mthr = 1            # minimum recruitment threshold
a = 1               # recruitment gain paramter (1)
minfr = 8           # minimum firing rate (8)
pfr1 = 35           # peak firing rate of first recruited motor unit (35)
pfrL = 25           # peak firing rate of last recruited motor unit (25)
mir = 1             # slope of the firing rate increase vs excitation (1)
rp = 100            # range of twitch tensions (100)
rt = 3              # range of contraction times (3 fold)
tL = 90             # longest contraction time (90)

# Various methods to create, or read in, force (#MVC)time-histories

# Create isotonic data -----------------------------------

fthscale = 0.5      # sets %MVC level for the trial duration (100% MVC is 1.00)
con = '0.50'        # for output file names
fthtime = 4       # duration to run trial (seconds)

fthsamp = dot(fthtime, samprate)
fth = zeros(1, fthsamp)
for z in arange(1, fthsamp).reshape(-1):
    fth[z] = fthscale

# Create Ramp Plateau data -----------------------------------

# TODO: Port to Python

#         con = 'Plateaus'
#         yMAXforce = 35
#         ondur = 32;
#         mag = 0.20
#         frame = 0;
#         cyc = ondur * samprate          # duration of each plateau
#         transition = 5 * samprate       # duration of transition between plateaus
#         for n = 1:cyc
#             frame = frame + 1;
#             fth(frame) = mag * 1;
#         end
#         for n = 1:transition
#             frame = frame + 1;
#             fth(frame) = (mag * 1) + (mag * n / transition);
#         end
#         for n = 1:cyc
#             frame = frame + 1;
#             fth(frame) = mag * 2;
#         end
#         for n = 1:transition
#             frame = frame + 1;
#             fth(frame) = (mag * 2) + (mag * n / transition);
#         end
#         for n = 1:cyc
#             frame = frame + 1;
#             fth(frame) = mag * 3;
#         end
#         fthsamp = frame

# Calculations from the Fuglevand, Winter & Patla (1993) Model

ns = arange(1, fthsamp, 1) # array of samples for fth
fth = fth[ns]

# Recruitment Threshold Excitation (thr)
thr = zeros(1, nu)
n = arange(1, nu, 1)  # TODO: strip the last 1 here
# this was modified from Fuglevand et al (1993) RTE(i) equation (1)
# as that did not create the exact range of RTEs (ie. 'r') entered
b = log(r + (1 - mthr)) / (nu - 1)
for i in arange(1, nu).reshape(-1):
    thr[i] = dot(a, exp(dot((i - 1), b))) - (1 - mthr)

# Peak Firing Rate (frp)
# modified from Fuglevand et al (1993) PFR equation (5) to remove thr(1)
# before ratio
frdiff = pfr1 - pfrL
frp = pfr1 - dot(frdiff, ((thr[n] - thr[1]) / (r - thr[1])))
maxex = thr[nu] + (pfrL - minfr) / mir  # maximum excitation
maxact = int(round(dot(maxex, res)))    # max excitation x resolution
ulr = dot(100, thr[nu]) / maxex         # recruitment threshold (%) of last motor unit

# Calculation of the rested motor unit twitch properties (these will
# change with fatigue)

# Firing Rates for each MU with increased excitation (act)
mufr = zeros(nu, maxact)
for mu in arange(1, nu).reshape(-1):
    for act in arange(1, maxact).reshape(-1):
        acti = act / res
        if acti >= thr[mu]:
            mufr[mu, act] = dot(mir, (acti - thr[mu])) + minfr
            if mufr[mu, act] > frp[mu]:
                mufr[mu, act] = frp[mu]
        else:
            if acti < thr[mu]:
                mufr[mu, act] = 0

k = arange(1, maxact, 1) # range of excitation levels

# Twitch peak force (P)
# this was modified from Fuglevand et al (1993) P(i) equation (13)
# as that didn't create the exact range of twitch tensions (ie. 'rp') entered
b = log(rp) / (nu - 1)
P = exp(dot(b, (n - 1)))

# Twitch contraction time (ct)
c = log(rp) / log(rt) # scale factor
ct = zeros(1, nu)
# assigns contraction times to each motor unit (moved into loop)
for mu in arange(1, nu).reshape(-1):
    ct[mu] = dot(tL, (1 / P[mu]) ** (1 / c))

# Normalized motor unit firing rates (nmufr) with increased excitation (act)
nmufr = zeros(nu, maxact)
for mu in arange(1, nu).reshape(-1):
    for act in arange(1, maxact).reshape(-1):
        nmufr[mu, act] = dot(ct[mu], (mufr[mu, act] / 1000)) # same as CT / ISI

# Motor unit force,  relative to full fusion (Pr) with increasing excitation
# based on Figure 2 of Fuglevand et al (1993)
sPr = 1 - exp(dot(- 2, (0.4 ** 3)))
Pr = zeros(nu, maxact)
for mu in arange(1, nu).reshape(-1):
    for act in arange(1, maxact).reshape(-1):
        # linear portion of curve
        # Pr = MU force relative to rest 100% max excitation of 67
        if nmufr[mu, act] <= 0.4:
            Pr[mu, act] = dot(nmufr[mu, act] / 0.4, sPr)

        if nmufr[mu, act] > 0.4:
            Pr[mu, act] = 1 - exp(dot(- 2, (nmufr[mu, act] ** 3)))

# Motor unit force (muP) with increased excitation
muP = zeros(nu, maxact)
for mu in arange(1, nu).reshape(-1):
    for act in arange(1, maxact).reshape(-1):
        muP[mu, act] = dot(Pr[mu, act], P[mu])

totalP = sum(muP, 1) # sum of forces across MUs for each excitation (dim 1)
maxP = totalP[maxact]

# Total Force across all motor units when rested
Pnow = zeros(nu, fthsamp)
Pnow[:, 1] = ravel(P)[:, 1]

# Calculation of Fatigue Parameters (recovery currently set to zero in
# this version)

# note, if rp = 100 & fat = 180, there will be a 100 x 180 = 1800-fold difference in
# the absolute fatigue of the highest threshold vs the lowest threshold.
# The highest threshold MU will only achieve ~57# of its maximum (at 25 Hz), so the actual range of fatigue
# rates is 1800 x 0.57 = 1026

# fatigue rate for each motor unit  (note: "log" means "ln" in Matlab)
b2 = log(fat) / (nu - 1)
mufatrate = exp(dot(b2, (n - 1)))

fatigue = zeros(1, nu)
for mu in arange(1, nu).reshape(-1):
    fatigue[mu] = dot(dot(mufatrate[mu], (FatFac / fat)), P[mu])


# the only variable is the relative force: Pr(mu, act), so this part is
# calculated once here

# Establishing the rested excitation required for each target load level
# (if 0.1# resolution, then 0.1# to 100#)
startact = zeros(1, 100)

for force in arange(1, 100).reshape(-1):
    # excitation will never be lower than that needed at rest for a given force
    # so it speeds the search up by starting at this excitation
    startact[force] = 0
    for act in arange(1, maxact).reshape(-1):
        one = (dot(totalP[act] / maxP, 100))
        if (dot(totalP[act] / maxP, 100)) < force:
            startact[force] = act - 1
        # TODO - Add this in
        # else:
        #     break # Stop search once value is found

Pchangecurves = zeros(nu, maxact)

for act in arange(1, maxact).reshape(-1):
    for mu in arange(1, nu).reshape(-1):
        # just used for graphical display
        Pchangecurves[mu, act] = dot(dot(fatigue[mu], Pr[mu, act]), P[mu])

print('start of fatigue analysis')

# Moving through force time-history and determing the excitation required
# to meet the target force at each time

TmuPinstant = zeros(nu, fthsamp)
m = zeros(1, fthsamp)
mufrFAT = zeros(nu, fthsamp)
ctFAT = zeros(nu, fthsamp)
ctREL = zeros(nu, fthsamp)
nmufrFAT = zeros(nu, fthsamp)
PrFAT = zeros(nu, fthsamp)
muPt = zeros(nu, fthsamp)
TPt = zeros(nu, fthsamp) # TODO: I think this should be 1 dimensional
Ptarget = zeros(1, fthsamp)
Tact = zeros(1, fthsamp)
Pchange = zeros(nu, fthsamp)
TPtMAX = zeros(1, fthsamp)
muPtMAX = zeros(nu, fthsamp)
muON = zeros(nu)
adaptFR = zeros(nu, fthsamp)
Rdur = zeros(1, nu)
acttemp = zeros(fthsamp, maxact)
muPna = zeros(nu, fthsamp)
muForceCapacityRel = zeros(nu, fthsamp)
timer = 0

for i in arange(1, fthsamp).reshape(-1):
    # shows a timer value every 15 seconds
    if i == dot(dot((timer + 1), samprate), 60):
        timer = timer + 1
        current = i / samprate
        print(current)

    # used to start at the minimum possible excitation (lowest it can be is 1)
    # so start with excitation needed for fth(i) when rested (won't be lower than this)
    force = round(dot(fth[i], 100)) + 1
    if force > 100:
        force = 100
    s = startact[force] - (dot(5, res)) # starts a little below the minimum
    if s < 1:
        s = 1
    acthop = round(maxact / hop) # resets 'acthop' to larger value for new sample
    act = copy(s) # start at lowest value then start jumping by 'acthop'
    # this starts at the mimimum (s) then searches for excitation required to meet the target
    for a in arange(1, maxact).reshape(-1):
        acttemp[i, a] = act
        for mu in arange(1, nu).reshape(-1):
            # MU firing rate adaptation modified from Revill & Fuglevand (2011)
            # this was modified to directly calculate the firing rate adaption, as 1 unit change in excitation causes 1 unit change in firing rate
            # scaled to the mu threshold (highest adaptation for hightest
            # threshold mu)
            if muON[mu] > 0:
                Rdur[mu] = (i - muON[mu] + 1) / samprate        # duration since mu was recruited at muON

            if Rdur[mu] < 0:
                Rdur[mu] = 0
            adaptFR[mu, i] = dot(dot(dot((thr[mu] - 1) / (thr[nu] - 1), adaptSF),
                                     (mufr[mu, act] - minfr + 2)), (1 - exp(dot(- 1, Rdur[mu]) / tau)))

            if adaptFR[mu, i] < 0:                              # firing rate adaptaion
                adaptFR[mu, i] = 0

            mufrFAT[mu, i] = mufr[mu, act] - adaptFR[mu, i]     # adapted motor unit firing rate: based on time since recruitment
            mufrMAX = mufr[mu, maxact] - adaptFR[mu, i]         # adapted FR at max excitation
            ctFAT[mu, i] = dot(ct[mu], (1 + dot(ctSF, (1 - Pnow[mu, i] / P[mu]))))  # corrected contraction time: based on MU fatigue

            ctREL[mu, i] = ctFAT[mu, i] / ct[mu]
            nmufrFAT[mu, i] = dot(ctFAT[mu, i], (mufrFAT[mu, i] / 1000))    # adapted normalized Stimulus Rate (CT * FR)
            nmufrMAX = dot(ctFAT[mu, i], (mufrMAX / 1000))                  # normalized FR at max excitation

            if nmufrFAT[mu, i] <= 0.4:                                      # fusion level at adapted firing rate
                PrFAT[mu, i] = dot(nmufrFAT[mu, i] / 0.4, sPr)              # linear portion of curve
            if nmufrFAT[mu, i] > 0.4:
                PrFAT[mu, i] = 1 - exp(dot(- 2, (nmufrFAT[mu, i] ** 3)))

            muPt[mu, i] = dot(PrFAT[mu, i], Pnow[mu, i])                    # MU force at the current time (muPt): based on adapted position on fusion curve
            if nmufrMAX <= 0.4:                                             # fusion force at 100% maximum excitation
                PrMAX = dot(nmufrMAX / 0.4, sPr)
            if nmufrMAX > 0.4:
                PrMAX = 1 - exp(dot(- 2, (nmufrMAX ** 3)))
            muPtMAX[mu, i] = dot(PrMAX, Pnow[mu, i])

        # print("Before")
        # print(TPt)
        # print(TPt[i])
        TPt[i] = sum(muPt[:, i]) / maxP                                  # total sum of MU forces at the current time (TPt)
        # print("After")
        print(TPt)
        print(TPt[i])
        TPtMAX[i] = sum(muPtMAX[:, i]) / maxP
        # used to speed up the search for the right excitation to meet the current target
        # quit()
        if TPt[i] < fth[i] and act == maxact:
            break
        if TPt[i] < fth[i]:
            act = act + acthop
            if act > maxact:
                act = copy(maxact)
        if TPt[i] >= fth[i] and acthop == 1:
            break # stop searching as the correct excitation is found
        if TPt[i] >= fth[i] and acthop > 1:
            act = act - (acthop - 1)    # if the last large jump was too much, it goes back and starts increasing by 1
            if act < 1:
                act = 1
            acthop = 1

    # for mu in arange(1, nu).reshape(-1):
    #     # can be modified to reset if the MU turns off
    #     if muON[mu] == 0 and (act / res) >= thr[mu]:
    #         muON[mu] = i # time of onset of mu recruitment (s)

    # Ptarget[i] = TPt[i] # modeled force level ?? do I need to do this, or can I just use TPt(i)
    # Tact[i] = act # descending (not adapted) excitation required to meet the target force at the current time

    # # Calculating the fatigue (force loss) for each motor unit
    # for mu in arange(1, nu).reshape(-1):
    #     if mufrFAT[mu, i] >= 0:
    #         Pchange[mu, i] = dot(
    #             dot(- 1, (fatigue[mu] / samprate)), PrFAT[mu, i])
    #     else:
    #         if mufrFAT[mu, i] < recminfr:
    #             Pchange[mu, i] = recovery(mu) / samprate
    #     if i < 2:
    #         Pnow[mu, i + 1] = P[mu]
    #     else:
    #         if i >= 2:
    #             Pnow[mu, i + 1] = Pnow[mu, i] + Pchange[mu, i]  # instantaneous strength of MU, right now without adaptation
    #     if Pnow[mu, i + 1] >= P[mu]:
    #         Pnow[mu, i + 1] = P[mu] # does not let it increase past rested strength
    #     if Pnow[mu, i + 1] < 0:
    #         Pnow[mu, i + 1] = 0 # does not let it fatigue below zero

Tstrength = zeros(1, fthsamp)
for i in arange(1, fthsamp).reshape(-1):
    for mu in arange(1, nu).reshape(-1):
        muPna[mu, i] = dot(Pnow[mu, i], muP[mu, maxact]) / P[mu]    # non-adapted MU max force at 100% excitation (muPna)
    Tstrength[i] = sum(muPna[:, i]) / maxP  # Current total strength without adaptation relative to max rested capacity

for i in arange(1, fthsamp).reshape(-1):
    endurtime = i / samprate
    if TPtMAX[i] < fth[i]:
        break

# Output
EndStrength = (dot(TPtMAX[fthsamp], 100))

print(endurtime)

for mu in arange(0, nu).reshape(-1):
    if mu == 0:
        mu = 1
    muForceCapacityRel[mu, ns] = dot(Pnow[mu, ns], 100) / P[mu] # for outputs below

# TODO: Replace graphing code
combo = matlabarray(cat(ravel(ns) / samprate, ravel(fth), dot(ravel(Tact) / res / maxex, 100),
                        dot(ravel(Tstrength), 100), dot(ravel(Ptarget), 100), dot(ravel(TPtMAX), 100)))
combo = combo.reshape(-1, 6)

fileA = 'python - ' + con + \
    ' A - Target - Act - Strength (no adapt) - Force - Strength (w adapt).csv'
fileB = 'python - ' + con + ' B - Firing Rate.csv'
fileC = 'python - ' + con + ' C - Individual MU Force Time-History.csv'
fileD = 'python - ' + con + ' D - MU Capacity - relative.csv'

np.savetxt(fileA, combo, delimiter=',', fmt='%1.16g', newline='\n')
np.savetxt(fileB, mufrFAT[:, :].T, delimiter=',', fmt='%1.16g', newline='\n')
np.savetxt(fileC, muPt[:, :].T, delimiter=',', fmt='%1.16g', newline='\n')
np.savetxt(fileD, muForceCapacityRel[:, :].T,
           delimiter=',', fmt='%1.16g', newline='\n')
