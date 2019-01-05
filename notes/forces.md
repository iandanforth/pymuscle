## I want to lift a 20kg weight repeatedly until failure, wait until recovery,
then do it again.


How do I relate a weight to the force needed to lift it?

F = ma

F = 20kg * 9.8 m/s^2

F = 196 kg m/s^2

F = 196 N

https://opentextbc.ca/physicstestbook2/chapter/forces-and-torques-in-muscles-and-joints/


### How do I determine the maximum force a muscle will output?

The peak force of each unit is calculated as an exponentially decaying curve from a max of 100 arbitrary units.

See PotvinFuglevand2017MuscleFibers.\_calc_peak_twitch_forces() or Fig 1b in Fuglevand 93.

The maximum force a muscle can output is the sum of those forces. 

In the default/example muscle of 120 motor units the sum of these values is 2609.03.

However this is only a theoretical maximum. Voluntary contractions cannot tap the full
reserve of potentially generated force, especially in the larger, fast twitch motor units.

The maximum voluntary force is described by the product of the normalized_forces array and the peak_twitch_forces array

The sum of this product is 2119.14 force units.

If this were interpreted as newtons this would be nonsensical for two reasons.

1. A muscle with 120 motor units is small. It's about the size of the Abductor Pollicis Brevis, the abductor muscle for the thumb.

2. The Abductor Pollicis Brevis can produce a maximum isometric force of ~ 60N of force in an adult human 

Using this ratio of 60 N/ 2119.14 force units results in ~ 0.028 N / force unit.

## In reality how large of a muscle (how many motor units) could realistically lift this weight?

The brachialis could likely lift this weight. Estimates of motor unit count in this muscle have a very large range (211-1816).

If we scaled up from the above estimate of 60 N / 120 MU then we would need ~ 3.3x more motor units or 396 motor units.

In our model a muscle with 396 motor units has a peak force output of 8542.139 arbitrary units or ~ 196.469 N.

The 0.028 N/ force unit ratio might not hold for each muscle however. If the smallest motor units of a large muscle produce more force
then this would effectively increase this ratio and decrease the number of required motor units for the same amount of force.

In addition the composition of each muscle will be different with some having more slow twitch fibers and
others having more fast twitch fibers.

In addition the mechanics of the arm mean that if the brachialis were actually lifting this weight it would be
doing so at a mechanical disadvantage proprotional to the length of the forearm. The force required would then be significantly increased
and thus the proposed motor unit count might increase as well.

In the arm all the biceps muscles combined exert a force on the forearm ~ 7.38x that of the force held in the hand.

The brachialis provides > 50% of the power in flexing the elbow, so might produce 4-5x the held force.

The force produced by each motor unit can also change over time with exercise.


## How do I measure the fatigue of the muscle?


## How do I measure the fatigue of individual motor units?


## How do I change the composition, fast vs slow, of the muscle?



Why do we get stronger as we get older?

"For instance, when muscle mass increases due to physical development during childhood, this may be only due to an increase in length of the muscle fibers, with no change in fiber thickness (PCSA) or fiber type." - https://en.wikipedia.org/wiki/Physiological_cross-sectional_area


