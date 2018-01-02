# Test Artifacts

Output reference files for a **limited** run of the script. Both Matlab and Python versions were run with the below parameters.

### Generation tag

8927bbfb0207c67ee75f49936f08f5e7bafc22e6

### Generation parameters

Default parameters were altered to reduce computation.

```matlab
%% Model input parameters
    nu = 12;           % number of neurons (ie. motor units) in the modeled pool ("n")

    samprate = 3;      % sample rate (10 Hz is suggested)
    res = 10;          % resolution of activations (set = 10 for 0.1 activation resolution, 100 for 0.01)
    hop = 20;           % allows for hopping through activations to more rapidly find that which meets the threshold (10 means every 1/10th of maxact)
    r = 50;             % range of recruitment thresholds (30 or 50)

    fat = 180;          % range of fatigue rates across the motor units (300 best)
    FatFac = 0.0225;    % fatigue factor (FF/S) percent of peak force of MU per second

    tau = 22;           % 22 based on Revill & Fuglevand (2011)
    adaptSF = 0.67;     % 0.67 from Revill & Fuglevand (2011)
    ctSF = 0.379;       % 0.379 based on Shields et al (1997)

    mthr = 1;           % minimum recruitment threshold
    a = 1;              % recruitment gain paramter (1)
    minfr = 8;          % minimum firing rate (8)
    pfr1 = 35;          % peak firing rate of first recruited motor unit (35)
    pfrL = 25;          % peak firing rate of last recruited motor unit (25)
    mir = 1;            % slope of the firing rate increase vs excitation (1)
    rp = 100;           % range of twitch tensions (100)
    rt = 3;             % range of contraction times (3 fold)
    tL = 90;            % longest contraction time (90)


%% Various methods to create, or read in, force (%MVC)time-histories

%     % Create isotonic data -----------------------------------

        fthscale = 0.5             % sets %MVC level for the trial duration (100% MVC is 1.00)
        con = '0.50';               % for output file names
        fthtime = 2;              % duration to run trial (seconds)
```
