from collections.abc import Callable
import gvar as _gvar
import numpy as _numpy
import time as _time
import gc as _gc
import warnings
from itertools import chain, combinations
from functools import partial as _partial
from tqdm import tqdm as _tqdm
import decimal as _decimal

from setup import SetupBetaFunction, BetaFunctionException

from swissfit import fit as fitter
from swissfit.optimizers import scipy_least_squares
from swissfit.model_averaging import model_averaging as _vrg
from swissfit.empirical_bayes import single_parameter_surrogate as sps

class BetaFunction(SetupBetaFunction): 
    """ Continuous beta-function

    Description:
        Analysis of the continuous beta-function based on arXiv:2109.09720, 
        PRD108(1)014502, and PRD109(11)114507.
    
    """
    def __init__(self,
                 nc: float | int = 3., 
                 nf: float | int = None, 
                 gauge_action: str = None,
                 logfn: str = None
                 ): 
        super().__init__(
            nc = nc, 
            nf = nf, 
            gauge_action = gauge_action,
            logfn = logfn
        )
        self.iv_fits = None
        self.ntrp_fits = None
        self.cnt_fits = None

        self.optimizer = scipy_least_squares.SciPyLeastSquares()

        if logfn is not None: warnings.showwarning = self.log

    def _new_empty_iv_fits_entry(self, fs, xtrp):
        return {x: {f: {o: {} for o in self.os} for f in fs} for x in xtrp}
    
    def _gather_iv_info(self, c, xtrp, mnt, mxt):
        vs = self.avg_data[c].keys()
        fs = list(set(
            f for v in vs 
            for f in self.avg_data[c][v].keys()
        ))
        ts = list(set(
            t for v in vs 
            for f in fs 
            for x in xtrp
            for o in self.os
            for t in self.avg_data[c][v][f]['_'.join([x,o])].keys()
            if mnt <= float(t) <= mxt
        ))
        ts.sort(key = (lambda x: float(x)))
        xfot = [
            (x,f,o,t) for x in xtrp 
            for f in fs 
            for o in self.os 
            for t in ts[::self._iv_thin]
        ]
        return (vs, fs, xfot)
    
    def get_fv_data(self, c, f, x, o, t, vs):
        xo = '_'.join([x,o])
        data = {
            'x': [1./self._vol(v) for v in vs if v not in self._iv_exclude[c]],
            'y': [
                    self.avg_data[c][v][f][xo][t] for v in vs
                    if v not in self._iv_exclude[c]
                ]    
        }
        return data

    def _nprm(self, prior, p0): 
        result = 0
        if prior is not None:
            for _,kv in prior.items(): result += len(kv)
        for k,kv in p0.items():
            if prior is not None:
                if k not in prior.keys(): result += len(kv)
            else: result += len(kv)
        return result

    def _powerset(self, x, nprm):
        l,h = len(x)-nprm-1, len(x)
        pwrst = chain.from_iterable(combinations(x,n+1) for n in range(l,h))
        result = []
        for v in list(pwrst): result.append(list(v))
        return result

    def _vol(self, vol):
        dims = vol.replace('t','l').split('l')[1:]
        return _numpy.prod([*map(float, dims)])

    def iv_xtrp(self, 
                mnt: float = None, 
                mxt: float = None, 
                exclude: dict[str,list[str]] = None,
                fcn: Callable[[any,dict[str,any]],any] = None, 
                prior: dict[str,any] = None,
                p0: dict[str,any] = None,
                xtrp: list[str] = ['g2', 'beta'],
                model_average: bool = False,
                v: int = 1,
                thin: int = 1,
        ):
        # Setup
        if not hasattr(self, 'avg_data'): 
            BetaFunctionException('Must grab fv data before iv extrapolation')
        if mnt is None: mnt = self._min_fv_flt
        if mxt is None: mxt = self._max_fv_flt
        if fcn is None: self.iv_fcn = self._iv_xtrp_fcn
        if prior is None: pass
        else: prior = {
            key: [val] if not hasattr(val,'__len__') else val 
            for key,val in prior.items()
            }
        if p0 is None: p0 = {'k1(t;beta)': [0.], 'k2(t;beta)': [0.]}
        else: p0 = {
            key: [val] if not hasattr(val,'__len__') else val 
            for key,val in p0.items()
            }
        nprm = self._nprm(prior, p0)

        if self.iv_fits is not None:
            del self.iv_fits
            _gc.collect()
        self.iv_fits = {}
        self.iv_qof = {}
        
        if model_average: d = {}
        self._iv_thin = thin
        if exclude is None:
            self._iv_exclude = {c: [] for c in self.avg_data.keys()}
        else: self._iv_exclude = exclude
        cs = list(self.data.keys())

        # Infinite volume extrapolation
        if v >= 1: 
            msg = 'Infinite volume extrapolation'
            if model_average: msg += '... this may take a while...'
            msg += '\n' + 50 * '~' 
            print(msg)
        for c in cs:
            if v >= 1: self._start_timer()
            (vs, fs, xfot) = self._gather_iv_info(c,xtrp,mnt,mxt)
            self.iv_fits[c] = self._new_empty_iv_fits_entry(fs,xtrp)
            self.iv_qof[c] = self._new_empty_iv_fits_entry(fs,xtrp)
            if model_average: d[c] = self._new_empty_iv_fits_entry(fs,xtrp)
            for x,f,o,t in _tqdm(xfot):
                try:
                    data = self.get_fv_data(c, f, x, o, t, vs)
                    if v >= 2: print(c, f, x, o, t)
                    if model_average: 
                        d[c][x][f][o][t] = data['y']
                        datax = self._powerset(data['x'], nprm)
                        datay = self._powerset(data['y'], nprm)
                        fits = []
                        self.iv_qof[c][x][f][o][t] = []
                        for n,(dtx,dty) in enumerate(zip(datax, datay)):
                            fits.append(
                                fitter.SwissFit(
                                    data = {'x': dtx, 'y': dty},
                                    prior = prior,
                                    p0 = p0,
                                    fit_fcn = self.iv_fcn
                                )(self.optimizer)
                            )
                            self.iv_qof[c][x][f][o][t].append(
                                {'chi2': fits[n].chi2, 'dof': fits[n].dof, 
                                 'p-value': fits[n].Q, 'logml': fits[n].logml}
                            )
                            if v >= 2: print(fits[n])
                        self.iv_fits[c][x][f][o][t] = (
                            _vrg.BayesianModelAveraging(
                                models = fits, ydata = data['y']
                            ).p
                        )
                    else: 
                        fit = fitter.SwissFit(
                            data = data,
                            prior = prior,
                            p0 = p0,
                            fit_fcn = self.iv_fcn
                        )(self.optimizer)
                        self.iv_fits[c][x][f][o][t] = fit.p
                        self.iv_qof[c][x][f][o][t] = {
                            'chi2': fit.chi2, 'dof': fit.dof, 
                            'p-value': fit.Q, 'logml': fit.logml
                        }
                        if v >= 2: print(self.iv_fits[c][x][f][o][t])
                except KeyError as err:
                    s1,s2 = 'ERROR: ' + ' '.join([c,x,f,o,t]), repr(err)
                    self.log.write(s1,s2)
                    pass
            if v >= 1:
                msg = 'Finished beta_b = ' + c.replace('p','.') 
                msg += ' in ' + str(self._stop_timer()) + ' secs'
                print(msg)
        if v >= 1: print('\n')
        _gc.collect()

    def _flatten(self, dct, pk = tuple(), stop = 5):
        x = []
        for k,kv in dct.items():
            nk = pk + (k,) if pk else (k,)
            if len(nk) < stop:
                x.extend(self._flatten(kv, pk = nk, stop = stop).items())
            else: x.append((nk,kv))
        return dict(x)

    def iv_ntrp(
            self,
            fcn: Callable[[any,dict[str,any]],any] = None, 
            prior: dict[str,any] = None,
            p0: dict[str,any] = None,
            v: int = 1,
            xerrors: bool = False, 
            emp_bayes_fcn: Callable[[float,any,any],float] = None,
            eblb: float = None,
            ebub: float = None,
            ebnpt: int = None,
            ebprms: list[str] = None,
            ebalg: str = 'steffen' 
        ):
        if fcn is None:
            msg = 'Must provide interpolating function'
            msg += ' as callable fcn(x,p) or fcn(p)'
            raise BetaFunctionException(msg)
        else: self.ntrp_fcn = fcn
        self.emp_bayes_fcn = emp_bayes_fcn
        if (emp_bayes_fcn is not None) and any(x is None for x in [eblb,ebub,ebnpt]):
            msg = 'Must provide lower bound (eblb), upper bound (ebub), '
            msg += 'and number of points (ebnpt) for empirical Bayes. '
            msg += 'Must also provide parameter name (ebprm) to be '
            msg += 'returned by empirical Bayes'
            raise BetaFunctionException(msg)
        if prior is None: 
            if xerrors: prior = {} 
        else: prior = {
            key: [val] if not hasattr(val,'__len__') else val 
            for key,val in prior.items()
            }
        if p0 is None: 
            msg = 'Must provide starting values for parameters'
            raise BetaFunctionException(msg)
        else: p0 = {
            key: [val] if not hasattr(val,'__len__') else val 
            for key,val in p0.items()
            }

        try:
            del self.ntrp_fits, self.ntrp_qof
            _gc.collect()
        except AttributeError: pass

        iv_data = {}
        for (_,x,f,o,t),p in self._flatten(self.iv_fits, stop = 5).items():
            if f not in iv_data.keys(): iv_data[f] = {}
            if o not in iv_data[f].keys(): iv_data[f][o] = {}
            if t not in iv_data[f][o].keys(): iv_data[f][o][t] = {}
            if x not in iv_data[f][o][t].keys(): iv_data[f][o][t][x] = [] 
            iv_data[f][o][t][x].append(self.iv_fcn(0.,p))

        if v >= 1: print('Intermediate interpolation\n' + 50 * '~')
        self.ntrp_fits,self.ntrp_qof,self.ntrp_nf = {},{},{}
        if v == 0: fotd = self._flatten(iv_data, stop = 3).items()
        else: fotd = _tqdm(self._flatten(iv_data, stop = 3).items())
        for (f,o,t),d in fotd:
            self.ntrp_flt = float(t)
            if o in self.os:
                if f not in self.ntrp_fits.keys(): 
                    self.ntrp_fits[f],self.ntrp_qof[f] = {},{}
                    self.ntrp_nf[f] = {}
                if o not in self.ntrp_fits[f].keys(): 
                    self.ntrp_fits[f][o],self.ntrp_qof[f][o] = {},{}
                    self.ntrp_nf[f][o] = {}
                    if v >= 1: print('Working on flow,discr. =', ','.join([f,o]))
                
                if xerrors: prior['x'],data = d['g2'],{'y': d['beta']}
                else: data = {'x': _gvar.mean(d['g2']), 'y': d['beta']}
                if self.emp_bayes_fcn is not None:
                    ebdata = {'x': d['g2'], 'y': d['beta']}
                    fcn = _partial(self.emp_bayes_fcn,data=ebdata,p0=p0)
                    ebfit = sps.SingleParameterSurrogate(
                        fcn = fcn,
                        lb = eblb, ub = ebub,
                        n_points = ebnpt,
                        spline_algorithm = ebalg
                    )
                    ebp = ebfit().x[0]
                    for ebprm in ebprms: prior[ebprm] = [_gvar.gvar(0.,ebp)]
                fit = fitter.SwissFit(
                    data = data,
                    prior = prior,
                    p0 = p0,
                    fit_fcn = self.ntrp_fcn
                )(self.optimizer)
                if v >= 2: print(fit)
                
                self.ntrp_fits[f][o][t] = fit.p
                self.ntrp_qof[f][o][t] = {
                    'chi2': fit.chi2, 'dof': fit.dof, 
                    'p-value': fit.Q, 'logml': fit.logml
                }
                self.ntrp_nf[f][o][t] = [
                    min(_gvar.mean(d['g2'])),max(_gvar.mean(d['g2']))
                ]

    def _cnt_fcn(self,x,p):
        return p['beta'][0] + p['slope'][0]*x

    def cnt_xtrp(
            self,
            mnt: float,
            mxt: float,
            mng2: float,
            mxg2: float,
            dg2: float = 0.1,    
            fcn: Callable[[any,dict[str,any]],any] = None, 
            ntrp_fcn: Callable[[dict[str,any]],any] = None, 
            prior: dict[str,any] = None,
            p0: dict[str,any] = None,
            v: int = 1,
            diagonal: bool = False,
            g2_round_precis: int = None
        ):
        if fcn is None: self.cnt_fcn = self._cnt_fcn
        else: self.cnt_fcn = fcn
        if p0 is None: p0 = {'beta': [0.], 'slope': [0.]}
        if g2_round_precis is None:
            d = _decimal.Decimal(dg2)
            g2_round_precis = -d.as_tuple().exponent
        g2s = _numpy.round(_numpy.arange(mng2, mxg2 + dg2, dg2), g2_round_precis)
        fo = [
            (f,o) for f in self.ntrp_fits.keys()
            for o in self.ntrp_fits[f].keys()
        ]
        self.mnt,self.mxt = mnt,mxt
        if ntrp_fcn is None: ntrp_fcn = self.ntrp_fcn

        self.g2s,self.betas,self.cnt_fits = {},{},{}
        for f,o in fo:
            if f not in self.g2s.keys():
                self.g2s[f],self.betas[f],self.cnt_fits[f] = {},{},{}
            if o not in self.g2s[f].keys():
                self.g2s[f][o],self.betas[f][o],self.cnt_fits[f][o] = [],[],[]
            if v >= 1: print('Working on flow,discr. =', ','.join([f,o]))
            for g2 in _tqdm(g2s):
                beta = []
                ts = [
                    t for t in self.ntrp_fits[f][o].keys()
                    if (mnt <= float(t) <= mxt) and
                    (self.ntrp_nf[f][o][t][0] <= g2 <= self.ntrp_nf[f][o][t][-1])
                ]
                for t in ts: 
                    p = self.ntrp_fits[f][o][t]
                    beta.append(ntrp_fcn(g2,p))
                if ts:
                    self.g2s[f][o].append(g2)
                    x = 1./_numpy.array([*map(float,ts)])
                    if diagonal: 
                        beta = _gvar.gvar(_gvar.mean(beta),_gvar.sdev(beta))
                        ps = []
                        for sig in [-1.,0.,1.]:
                            fit = fitter.SwissFit(
                                data = {
                                    'x': x, 
                                    'y': beta + sig*_gvar.sdev(beta)
                                },
                                prior = prior,
                                p0 = p0,
                                fit_fcn = self.cnt_fcn
                            )(self.optimizer)
                            ps.append(fit.p)
                        self.cnt_fits[f][o].append(ps)
                        mean = _gvar.mean(self.cnt_fcn(0.,ps[1]))
                        err = _gvar.mean(self.cnt_fcn(0.,ps[0]))
                        err -= _gvar.mean(self.cnt_fcn(0.,ps[-1]))
                        err = 0.5*abs(err)
                        self.betas[f][o].append(_gvar.gvar(mean,err))
                        if v >= 2: print(fit)
                    else:
                        fit = fitter.SwissFit(
                            data = {'x': x, 'y': beta},
                            prior = prior,
                            p0 = p0,
                            fit_fcn = self.cnt_fcn
                        )(self.optimizer)
                        p = fit.p
                        self.cnt_fits[f][o].append(p)
                        self.betas[f][o].append(self.cnt_fcn(0.,p))
                        if v >= 2: print(fit)



