#!/usr/bin/env python
"""Write the harness's toric geometry inputs in benchmarks/data/.

The toric cases need cymetric's ``toric_data`` dict: the anticanonical
hypersurface (``exp_aK``, ``coeff_aK``), the sections of the Kaehler-cone
generators (``exps_sections``), the GLSM charges, the patch masks of the
maximal cones, and the volume normalisation.  Upstream cymetric produces that
dict inside SageMath with
``cymetric.sage.sagelib.prepare_toric_cy_data(ToricVariety(fan), fname)``.

This box has no SageMath, so the committed files in ``benchmarks/data`` were
produced with CYTools (the McAllister group's toolchain) from the
Kreuzer-Skarke polytopes whose vertices each file records, and converted here
into a plain JSON that the harness can read with numpy alone.  Every file
carries its own provenance block, and the format is exactly the keys cymetric
itself consumes, so a third party can regenerate the same files either way:

    # from a sage-produced pickle
    python benchmarks/make_toric_data.py --from-pickle toric_data.pickle \
        --h11 5 --out benchmarks/data/toric_h11_5.json

    # from a dict that also carries 'kmoduli' / 'int_nums' (as ours do)
    python benchmarks/make_toric_data.py --from-pickle staged.pkl --h11 5 \
        --key toric_data --out benchmarks/data/toric_h11_5.json

Run with ``--check`` to validate the committed files instead: it loads each
one, builds cymetric's toric point generator from it and prints the geometry
sizes.
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for _p in (REPO_ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

REQUIRED = ['dim_cy', 'vol_j_norm', 'coeff_aK', 'exp_aK', 'exps_sections',
            'patch_masks', 'glsm_charges']


def to_json_safe(td):
    """Plain-JSON form of a toric_data dict (complex -> [re, im] pairs)."""
    out = {}
    for key, val in td.items():
        if key == 'coeff_aK':
            out[key] = [[float(np.real(c)), float(np.imag(c))] for c in val]
        elif isinstance(val, np.ndarray):
            out[key] = val.tolist()
        elif isinstance(val, (np.integer, np.floating)):
            out[key] = val.item()
        elif isinstance(val, list):
            out[key] = [v.tolist() if isinstance(v, np.ndarray) else v
                        for v in val]
        else:
            out[key] = val
    return out


def write_case(td, kmoduli, int_nums, h11, out_path, source):
    missing = [k for k in REQUIRED if k not in td]
    if missing:
        raise SystemExit('toric_data is missing %s' % missing)
    blob = {
        'h11': int(h11),
        'toric_data': to_json_safe(td),
        'kmoduli': [float(x) for x in np.asarray(kmoduli, dtype=float)],
        'int_nums': np.asarray(int_nums, dtype=float).tolist(),
        'provenance': source,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as fh:
        json.dump(blob, fh, indent=1, sort_keys=True)
    print('wrote %s (%d coords, %d monomials, %d patches, %d generators)'
          % (out_path, len(td['glsm_charges'][0]), len(td['exp_aK']),
             len(td['patch_masks']), len(td['exps_sections'])))


def check():
    import cases
    data_dir = os.path.join(HERE, 'data')
    for name in sorted(os.listdir(data_dir)):
        if not name.endswith('.json'):
            continue
        h11 = int(name.split('_')[-1].split('.')[0])
        blob = cases.load_toric_data(h11)
        pg, _ = cases.toric_pointgen(h11)
        td = blob['toric_data']
        print('%-22s h11=%-3d ncoords=%-3d monomials=%-5d patches=%-4d '
              'generators=%-3d sections=%d'
              % (name, h11, pg.ncoords, len(td['exp_aK']),
                 len(td['patch_masks']), len(td['exps_sections']),
                 sum(len(s) for s in td['exps_sections'])))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--from-pickle')
    p.add_argument('--key', default=None,
                   help="key holding the toric_data dict inside the pickle")
    p.add_argument('--h11', type=int)
    p.add_argument('--out')
    p.add_argument('--source', default='unspecified',
                   help='free-text provenance recorded in the output file')
    p.add_argument('--check', action='store_true',
                   help='validate the committed data files and exit')
    args = p.parse_args()

    if args.check:
        return check()
    if not (args.from_pickle and args.h11 and args.out):
        p.error('--from-pickle, --h11 and --out are required')
    with open(args.from_pickle, 'rb') as fh:
        blob = pickle.load(fh)
    td = blob[args.key] if args.key else blob
    kmoduli = blob.get('generator_coefficients',
                       np.ones(len(td['exps_sections'])))
    int_nums = blob.get('kappa_gen', td.get('int_nums'))
    write_case(td, kmoduli, int_nums, args.h11, args.out,
               {'note': args.source, 'from_pickle': args.from_pickle})


if __name__ == '__main__':
    main()
