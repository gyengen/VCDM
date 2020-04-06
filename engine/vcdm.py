# -*- coding: utf-8 -*-

"""
Ideas for future development:

1. User should be able to
 - define boundary conditions
 - run simulation and capture both variations over the simulation period
   as well as the state of the system at the end of the simulation
 - this state information can then be used as boundary information
   when undertaking a subsequent simulation over a second time period.
That's how this should work.  Need to think about capturing state info
for cohesive layer condition _as_well_as_ state of pipe segments.

2. Decouple mass transport engine from VCDM as could in theory be used with
   any cohesive layer discolouration model e.g. PODDS

"""
from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt
from collections import deque

try:
    xrange
except NameError:
    xrange = range


class Seg(object):
    """ Represents a plug/segment of water in a Lagrangian time-driven water
    quality advection model.

    A segment is defined in terms of its volume (:attr:`vol`) and
    concentration (:attr:`conc`). Its position along a pipe is
    determined by the :class:`Seg` object's position in a data
    structure of objects of the same class.

    """
    def __init__(self, vol, conc):
        """ Initialise a segment given a volume (:param:`vol`) and concentration
        (:param:`conc`) """
        self.vol = vol
        self.conc = conc

    def __repr__(self):
        return "Seg(vol={},conc={})".format(self.vol, self.conc)
# END class Seg


class VCDM(object):
    """ Variable Condition Discolouration Model simulation."""

    def __init__(self, times_v, strength_v, init_cond_v, applied_shear_v,
                 removal_rate, regen_rate, alpha, n=1.0):
        """Initialise but not run a VCDM simulation.

        Keyword arguments:
        times_v         -- An ndarray of times over which to simulate [s].
        strength_v      -- An ndarray of the _boundaries_ of discrete layer
                           strength bands for which the condition and material
                           release will be tracked over time [Pa].
        init_cond_v     -- An ndarray of the condition of all layers at t=0. [-]
                           Each element must be in [0,1], with 1 indicating
                           maximal material present (worst condition).
                           Must be of length length(strength_v) - 1.
        applied_shear_v -- An ndarray of values of boundary shear stress per
                           timestep [Pa].  Must be the same cardinality as
                           times_v.
        removal_rate    -- Coeff that describes ease with which layer condition
                           decreases upon erosion. Aka beta_e [Pa^-1 s^-1]
        regen_rate      -- Coeff that describes ease with which layer condition
                           increases due to material regeneration. Aka beta_r
                           [s^-1].  Can be constant (a scalar) or time-varying
                           (an array that is the same length as times_v).
        alpha           -- Used to convert the relative amount of material
                           mobilsed to a 'material release' [TPMU Pa^-1 m^-2]

        """
        # Check all time-series arrays are of the same shape
        arrs = [times_v, applied_shear_v]
        if isinstance(regen_rate, np.ndarray):
            arrs.append(regen_rate)
        if not all(arrs[0].shape == a.shape for a in arrs[1:]):
            raise ValueError("times_v, applied_shear_v (and regen_rate, if it" +
                             " is an array and not a scalar) should be 1D" +
                             " arrays of the same length.")

        # Check that times_v and strength_v are strictly increasing
        for arr, arr_name in zip((times_v, strength_v),
                                 ('times_v', 'strength_v')):
            if not _is_monotonically_incr(strength_v):
                raise ValueError("Each element of {}".format(arr_name) +
                                 "should be greater than the previous.")
        self.times_v = times_v
        self.strength_v = strength_v
        self._strength_v_midpoints = (self.strength_v[1:] +
                                      self.strength_v[:-1]) / 2.0
        self.dstrength = self.strength_v[1] - self.strength_v[0]

        self.cond_m = np.empty((self.times_v.size,
                                self._strength_v_midpoints.size))
        if not init_cond_v.size == self.strength_v.size - 1:
            raise ValueError('the init_cond_v numpy.ndarray argument must ' +
                             'be one element shorter than the strength_v ' +
                             'argument.')
        self.cond_m[0, :] = init_cond_v

        self.applied_shear_v = applied_shear_v

        self.mat_release_rate_v = np.zeros_like(self.times_v)
        self.mat_regen_v = np.zeros_like(self.times_v)

        self.removal_rate = removal_rate
        self.regen_rate = regen_rate
        self.alpha = alpha
        self.n = n

    def sim(self, cython=True):
        """VCDM model where the erosion rate is
         - constant wrt t
         - variable wrt shear strength

        Arguments
        cython -- boolean for whether to use fast Cython simulation function

        Improvements over 4.0
        v4.1 - scaled output by dt to ensure output is material release per
               m^2 per unit time rather than per timestep.
        v4.2 - ?
        v4.3 - allow upstream turbidity profile as input to advection process
               + cython maths
        v4.4 - changed signs in certain equations to make more logical (beta_r
               must now be positive)
        v4.5 - fixed bug in advection code where wall to bulk transfer at ends
               of pipe was over estimated by a factor of 2
        v4.6 - Erosion/regen now occurs depending on relation of shear stress
               to strength band _mid-point_.  Fixed bug where material release
               at extremities of pipe was not properly calculated.
        v4.7 - Regeneration rate can be a scalar or a time-series array.

        """

        """
        if cython:
            vcdm_cython.sim_cython(self.times_v, self._strength_v_midpoints,
                                   self.applied_shear_v, self.cond_m,
                                   self.mat_release_rate_v, self.mat_regen_v,
                                   self.removal_rate,
                                   np.atleast_1d(self.regen_rate),
                                   self.alpha, self.dstrength, self.n)
            return self.mat_release_rate_v

        """
        dcond_dt_v = np.zeros_like(self._strength_v_midpoints)

        for t_idx, _ in enumerate(self.times_v):
            if t_idx == 0:
                continue

            dt = self.times_v[t_idx] - self.times_v[t_idx - 1]

            # Create boolean array to identify eroding layers
            eroding_layers_v = self.applied_shear_v[t_idx] >= \
                self._strength_v_midpoints

            # For erosion the rate of change in relative condition is
            dcond_dt_v[eroding_layers_v] = - self.removal_rate * \
                        ((self.applied_shear_v[t_idx] -
                          self._strength_v_midpoints[eroding_layers_v])**self.n)
            # For regeneration the rate of change in relative condition is simply
            dcond_dt_v[~ eroding_layers_v] = self.regen_rate if \
                np.isscalar(self.regen_rate) else self.regen_rate[t_idx]

            # The new condition per layer is the old + (the rate of change * dt)
            self.cond_m[t_idx, :] = self.cond_m[t_idx - 1, :] + dcond_dt_v * dt
            # Layer condition must be limited to the range [0,1]
            self.cond_m[t_idx, :].clip(0., 1., out=self.cond_m[t_idx, :])

            # Material flux _from_ the pipe wall (TPMU/m^2/s)
            self.mat_release_rate_v[t_idx] = - self.alpha * \
                                    np.sum(self.cond_m[t_idx, eroding_layers_v] -
                                           self.cond_m[t_idx - 1, eroding_layers_v]) \
                                    * self.dstrength / dt
            # Material flux _to_ the pipe wall (TPMU/m^2/s)
            self.mat_regen_v[t_idx] = self.alpha * \
                                  np.sum(self.cond_m[t_idx, ~eroding_layers_v] -
                                         self.cond_m[t_idx - 1, ~eroding_layers_v]) \
                                  * self.dstrength / dt

        return self.mat_release_rate_v

    def get_quant_series(self, series, val, rel=False):
        """Get a time or shear strength series of the quantity of material at
        the wall given a particular shear strength or time.

        Keyword arguments:
         series -- 'time' or 'shear_strength'
         val -- a 'series' series will be returned for the time or shear
                strength nearest to this value
         rel -- the selected 'val' is a relative value (i.e. in [0,1])
                rather than is an absolute shear strength or time

        """
        if series not in ('time', 'shear_strength'):
            raise ValueError("series must be one of 'time' or 'shear_strength.")
        arr = self._strength_v_midpoints if series == 'shear_strength' else \
              self.times_v

        if not rel and val < 0:
            raise ValueError("Absolute {} cannot be negative.".format(series))
        if rel:
            if val < 0 or val > 1:
                raise ValueError("Relative {} cannot lie outside [0,1]".format(series))
            val = arr.ptp() * val + arr.min()
        idx = (np.absolute(arr - val)).argmin()
        return self.cond_m[:, idx] if series == 'shear_strength' else \
            self.cond_m[idx, :]

    def advect(self, Q_v, D, L, init_num_segs, upstream_conc_v=None,
               conc_tol=0.05, anim=False, cython=True):
        """ After running a VCDM simulation to generate the time-series response
        to a shear stress profile (in pseudo-mass per m^2 pipe wall area per s)
        this method uses a Lagrangian mass transport method to advect any
        mobilised discolouration material to the end of the pipe.

        Keyword arguments:
         Q_v -- pipe flow (m^3/s). Should really correspond to the shear profile
                used in VCDM sim.
         D -- pipe diameter (m)
         L -- pipe length (m)
         init_num_segs -- at the start of the advection process the pipe volume
                          is divided into a number of equal-sized segments of
                          zero concentration.
         upstream_conc_v -- time-series concentration values at node at upstream
                            end of pipe
         conc_tol -- tolerance used to determine whether a new segment should
                     be added to the upstream end of the pipe at a particular
                     timestep (some units of conc or pseudo-conc e.g. NTU or
                     kg/m^3)
         anim -- view animation of pipe segment positions, volumes and
                 concentrations (darker segment shading is low conc, lighter is
                 high conc).  Only valid if cython is False.
         cython -- boolean stating whether to use fast cython Lagrangian
                   transport function

        See section on the Lagrangian time-driven mass transport method in:

        Rossman, L., Boulos, P., 1996. Numerical Methods for Modeling Water
        Quality in Distribution Systems: A Comparison. Journal of Water
        Resources Planning and Management 122, pp.137-146.

        """

        self.Q_v = Q_v

        """
        if cython:
            if not (self.times_v.shape == self.Q_v.shape ==
                    self.mat_release_rate_v.shape):
                raise Exception("Time, flow and material release vectors " +
                                "must all be of the same size")
            if upstream_conc_v is None:
                upstream_conc_v = np.zeros_like(self.times_v)
            elif not upstream_conc_v.shape == self.times_v.shape:
                raise Exception("Time and upstream turbidity profiles must " +
                                "all be of the same size")
            if D <= 0 or L <= 0 or conc_tol <= 0:
                raise Exception("Diameter, length and concentration " +
                                "tolerance must all be positive real numbers")
            if init_num_segs <= 0:
                raise Exception("The initial number of segments must be " +
                                "positive")
            node_conc_v = _advect(self.times_v, self.Q_v,
                                  self.mat_release_rate_v, float(D), float(L),
                                  int(init_num_segs), upstream_conc_v,
                                  float(conc_tol))
            return node_conc_v

        """
        # Vector for storing conc per timestep at d/s node
        node_conc_v = np.zeros(self.times_v.size)

        # Total vol of single pipe
        total_vol = L * np.pi * ((D/2.)**2)

        # Maintain concentrations and volumes of Lagrangian segs
        # Lowest index of deque is most d/s seg
        v = total_vol / float(init_num_segs)
        segs = deque((Seg(conc=0, vol=v) for s in xrange(init_num_segs)))

        #if anim:
            #pipe_anim = PipeAnim(total_vol, L)

        for t_idx, _ in enumerate(self.times_v):
            if t_idx == 0 or Q_v[t_idx] == 0.0:
                continue

            dt = self.times_v[t_idx] - self.times_v[t_idx - 1]
            dvol = self.Q_v[t_idx] * dt

            # React all segments
            #
            # Originally used area_sweep_rate as measure of
            # amount of wall affected per segment per timestep
            #
            # # area_sweep_rate = velocity * circumference
            # area_sweep_rate = 4 * self.Q_v[t_idx] / float(D)
            # seg.conc += self.mat_release_rate_v[t_idx] * area_sweep_rate * \
            #    dt / seg.vol
            #
            # However, this is not correct as it means the concentration update
            # for two contiguous but different sized segments would be
            # different.
            #
            # To ensure that the concentration increments would be the same as a
            # result of the effect of differing volumes being cancelled out by
            # differing lengths the segment surface area must be used instead of
            # the velocity-driven area sweep rate.  This must be calculated per
            # parcel (within loop below).
            #
            # To see why this is correct consider that the concentration update
            # of a stationary parcel and a moving one.  The concentration update
            # for a constant mat_release_rate_v[t_idx] (mass per m^2 wall per s)
            # should be the same for both parcels, regardless of their velocity.
            # Plus, when using segment wall area rather than area_sweep_rate the
            # units of the concentration update are correct w.r.t time.
            #
            # NB this error in the orignal mass transport was detected as a
            # result of their being a anomalous spike in turbidity one turnover
            # time after a step increase in shear stress.
            for i in xrange(len(segs)):
                # Needed as we are iterating over a deque
                # using 'rotate'
                seg = segs[0]
                # Conc incr is mass release in dt given pipe wall area,
                # all divided by dilution volume
                seg_wall_area = np.pi * D * L * seg.vol / total_vol
                prev_mass = seg.conc * seg.vol
                additional_mass = self.mat_release_rate_v[t_idx] * dt * \
                    seg_wall_area
                seg.conc = (prev_mass + additional_mass) / seg.vol
                segs.rotate(-1)

            # Advect all segments:
            # Add new parcel at u/s node if conc tollerance exceeded
            new_seg_wall_area = np.pi * D * L * dvol / total_vol
            # Need to factor in the 'background' concentration at the
            # upstream node
            upstream_conc = upstream_conc_v[t_idx] \
                if isinstance(upstream_conc_v, np.ndarray) else 0
            new_seg_conc = (0.5 * self.mat_release_rate_v[t_idx] *
                            new_seg_wall_area * dt / dvol) + upstream_conc
            if abs(new_seg_conc - segs[-1].conc) > conc_tol:
                segs.append(Seg(conc=new_seg_conc, vol=dvol))
            # Otherwise enlarge the existing most upstream segment:
            else:
                segs[-1].conc = ((segs[-1].conc * segs[-1].vol) + (new_seg_conc * dvol)) / \
                                (segs[-1].vol + dvol)  # vol-weighted mean
                segs[-1].vol += dvol  # sum

            # Remove mass and vol from d/s end
            vol_need_to_remove = dvol  # 'decumulator'
            node_mass_in = 0
            node_vol_in = 0
            while vol_need_to_remove > 0:
                vol_removed = min(vol_need_to_remove, segs[0].vol)
                node_vol_in += vol_removed
                # The mass transfer from the pipe wall to the bulk water as
                # previously calculated is an overestimate by a factor of 2 for
                # the most downstream vol_need_to_remove volume in the pipe as
                # only half the wall area of this volume of water is reactive
                # (on a time-averaged basis).  This over-estimation is
                # accounted for here using the expression to the right of the
                # minus sign.
                node_mass_in += vol_removed * segs[0].conc - \
                    ((self.mat_release_rate_v[t_idx] * dt) *
                     (0.5 * vol_removed * np.pi / D))
                if vol_removed == segs[0].vol:
                    segs.popleft()  # remove element with lowest index
                else:
                    segs[0].vol -= vol_removed
                vol_need_to_remove -= vol_removed
            node_conc_v[t_idx] = node_mass_in / node_vol_in

            if anim:
                pipe_anim.draw(segs, self.times_v[t_idx] * dt)
        if anim:
            pipe_anim.close()
        return node_conc_v

    def mat_release_rate_plot(self, specific_strengths=None,
                              turb_legal_limit=4):
        """NEED DOCSTRING"""
        if specific_strengths:
            fig, (ax_strengths, ax_cond, ax_mass_flux, ax_turb) = \
                plt.subplots(nrows=4, ncols=1, sharex=True)
        else:
            fig, (ax_strengths, ax_mass_flux, ax_turb) = \
                plt.subplots(nrows=3, ncols=1, sharex=True)

        ax_strengths.plot(self.times_v, self.applied_shear_v)
        ax_strengths.set_ylabel(r"$\tau_a [Pa]$")

        if specific_strengths:
            for tau in specific_strengths:
                shear_idx = (np.abs(self._strength_v_midpoints - tau)).argmin()
                ax_cond.plot(self.times_v, self.cond_m[:, shear_idx],
                             label=r"$\tau=" + "{}Pa$".format(tau))
            ax_cond.set_ylabel(r"$\varphi(\tau,t) [-]$")
            ax_cond.set_ylim(ymax=1.1)

        ax_mass_flux.plot(self.times_v, self.mat_release_rate_v)
        ax_mass_flux.set_ylabel(r"$N(t) \frac{[TPMU}{m^{2}s^{1}}]$")

        turb_v = self.advect(Q_v=self.Q_v, D=self.D, L=self.L,
                             init_num_segs=1000, conc_tol=0.001, anim=False)
        ax_turb.plot(self.times_v, turb_v)
        ax_turb.axhline(turb_legal_limit)
        ax_turb.set_ylabel("$T(t) [NTU]$")
        ax_turb.set_xlabel("Time (s)")

        styles = ('-', '--', ':', '-.', '_')
        for ax in fig.get_axes():
            for i, l in enumerate(ax.get_lines()):
                l.set_linestyle(styles[i])
                l.set_color('k')
            ax.grid(True)
        if specific_strengths:
            ax_strengths.legend()
        plt.tight_layout()
        return fig

    def heatmap(self, width=16, height=9, ax=None):
        """Generate a plot of the temporal change in layer condition

        Keyword arguments:
        width -- width in inches of created figure
        height -- height in inches of created figure
        ax -- optional Matplotlib Axes object

        Returns:
         Matplotlib Figure instance

        """
        if ax is None:
            fig = plt.figure(figsize=(width, height))
            ax = fig.add_subplot(1, 1, 1)
        ax_contour = ax.contourf(self.times_v, self._strength_v_midpoints,
                                 self.cond_m.T, 40,
                                 cmap=plt.cm.gray_r)  # 4th arg is # levels
        ax_colorbar = ax.figure.colorbar(ax_contour, use_gridspec=True)
        ax_colorbar.set_label("Relative material quantity")

        ax.set_xlabel('Time')
        ax.set_xlim(0, self.times_v[-1])

        ax.set_ylabel('Shear strength [Pa]')
        ax.set_ylim(0, self._strength_v_midpoints[-1])

        if ax is None:
            fig.tight_layout()
            return fig

    def surf_plot(self, use_mpl=False):
        """Surface plot of layer condition for each layer strength at each timestep.

         Keyword arguments:
         use_mpl -- Set to True to use Matplotlib's 'mplot3d' toolkit rather
                    than Mayavi2's 'mlab' API for drawing the plot
                    (default False)

        """
        if not use_mpl:
            from mayavi import mlab
            fig = mlab.figure(size=(1200, 800), fgcolor=(1, 1, 1),
                              bgcolor=(0.5, 0.5, 0.5))
            surf_plot = mlab.surf(self.times_v, self._strength_v_midpoints,
                                  self.cond_m, warp_scale="auto")
            mlab.axes(surf_plot,
                      nb_labels=5,
                      ranges=[self.times_v.min(), self.times_v.max(),
                              self._strength_v_midpoints.min(),
                              self._strength_v_midpoints.max(),
                              self.cond_m.min(), self.cond_m.max()],
                      xlabel="Time",
                      ylabel="Layer strength",
                      zlabel="Condition")
            mlab.show()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            xim, yim = np.meshgrid(self.times_v, self._strength_v_midpoints)
            ax.plot_surface(xim, yim, self.cond_m)

            ax.set_xlabel("Time")
            ax.set_ylabel("Strength")
            ax.set_zlabel("Condition")
            plt.show()

    def advect_recirc_w_tank(self, Q_v, D, L_to_tank, L_from_tank, tank_vol,
                             init_num_segs=100, init_conc=0.0, conc_tol=0.05):
        """Advect turbid material around a loop containing a well-mixed tank."""
        self.Q_v = Q_v

        if not (self.times_v.shape == self.Q_v.shape ==
                self.mat_release_rate_v.shape):
            raise Exception("Time, flow and material release vectors must " +
                            "all be of the same size")
        if init_conc < 0:
            raise Exception("Initial concentration must be a single " +
                            "non-negative float")
        if D <= 0 or L_to_tank <= 0 or L_from_tank <= 0 or conc_tol <= 0:
            raise Exception("Diameter, lengths and concentration tolerance " +
                            "must all be positive real numbers")
        if init_num_segs <= 0:
            raise Exception("The initial number of segments must be positive")
        node_conc_v = _advect_recirc_w_tank(self.times_v, self.Q_v,
                                            self.mat_release_rate_v, float(D),
                                            float(L_to_tank),
                                            float(L_from_tank),
                                            float(tank_vol),
                                            int(init_num_segs),
                                            float(init_conc), float(conc_tol))
        return node_conc_v


def calc_init_cond_v_from_two_cusps(strength_v, s_low_cusp, s_high_cusp,
                                    init_cond_min=0.0, init_cond_max=1.0):
    """Calculate an estimation of the VCDM material condition per layer
    strength at t=0 using two cusps.

    Arguments:
    strength_v -- vector of boundaries of layer strength bands at which
                  material to be tracked
    s_low_cusp -- strength below which relative amount of material is
                  init_cond_min
    s_high_cusp -- strength above which relative amount of material is
                   init_cond_max

    Returns:
    vector (of length of length(strength_v) - 1) of relative amount of
    material per layer strength at t=0

    NB the relative amount of material at strengths between s_low_cusp and
    s_high_cusp is calculated using linear interpolation

    """
    strength_v_midpoints = (strength_v[1:] + strength_v[:-1]) / 2.0
    if s_low_cusp < strength_v_midpoints.min():
        raise Exception("s_low_cusp must be greater than mean(strength_v[:2])")
    if s_high_cusp > strength_v.max():
        raise Exception("s_high_cusp must be less than mean(strength_v[-2:])")

    init_cond_v = np.empty_like(strength_v_midpoints)
    if s_low_cusp != s_high_cusp:
        gradient = (init_cond_max - init_cond_min) / (s_high_cusp - s_low_cusp)
        init_cond_v = gradient * (strength_v_midpoints - s_low_cusp) + \
            init_cond_min
        init_cond_v.clip(init_cond_min, init_cond_max, out=init_cond_v)
    else:  # avoid div by zero error when calculating gradient
        init_cond_v[strength_v_midpoints < s_low_cusp] = init_cond_min
        init_cond_v[strength_v_midpoints >= s_low_cusp] = init_cond_max

    return init_cond_v


def _is_monotonically_incr(arr):
    # Are all values of the enumerable arr strictly greater than the previous?
    return np.ediff1d(arr).min() > 0.0
