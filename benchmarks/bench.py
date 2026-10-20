#!/usr/bin/env python
"""Reproducible performance benchmark harness for the cymetric package.

Design goals
------------
1. Compile time is reported separately from steady-state time.  In JAX the
   first call to a jitted function pays XLA compilation, which on these models
   is 10-100x the steady-state step.  Mixing the two hides exactly the
   regressions this harness is meant to catch, so every timed quantity is
   reported as ``first_call_s`` (compile + execute) plus the median and spread
   of ``repeats`` further calls.
2. Median over repeats with the spread (p25/p75, min/max), never a single
   sample and never a mean.
3. One record per case in machine-readable JSON, with full provenance
   (library commit, device, library versions, case parameters).
4. Each case runs in a fresh subprocess, so peak device memory and compile
   time are per-case and a crash or an OOM in one case cannot take the run
   down.  Cases that cannot run are recorded with status ``blocked`` and the
   exact error text, and stay in the harness.
5. Degrades to CPU when no GPU is visible.

Usage
-----
    python benchmarks/bench.py --help
    python benchmarks/bench.py --list
    python benchmarks/bench.py --quick                  # smoke run, few repeats
    python benchmarks/bench.py --cases 'train_jax.*'    # regex filter
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
        python benchmarks/bench.py --out benchmarks/baseline.json

The harness always imports cymetric from the repository that contains this
file (it prepends the repo root to sys.path), and records the resolved
``cymetric.__file__`` in the provenance block, so a stale ``pip install -e``
elsewhere cannot silently be the thing you measured.
"""

import argparse
import json
import os
import platform
import re
import subprocess
import statistics
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
# Import cymetric from this repository, and the case list from this directory,
# whatever the current working directory or an editable install elsewhere says.
for _p in (REPO_ROOT, BENCH_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
RESULT_MARKER = '###BENCH_RESULT###'


# ---------------------------------------------------------------------------
# timing core
# ---------------------------------------------------------------------------

def percentile(sorted_vals, q):
    """Linear-interpolation percentile of an already sorted list."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def time_calls(call, repeats, block=None, keep_samples=True):
    """Time ``call`` once (compile + execute), then ``repeats`` more times.

    Args:
        call: zero-argument callable.  Its return value is passed to ``block``.
        repeats (int): number of steady-state samples.
        block (callable, optional): forces completion of an asynchronous
            result (``lambda r: r.block_until_ready()`` for JAX).  The cost of
            blocking is inside the measured window, which is what we want: an
            unblocked JAX timing measures dispatch, not compute.
        keep_samples (bool): store all steady-state samples in the record.

    Returns:
        dict: timing block.  ``first_call_s`` includes compilation.
    """
    if block is None:
        def block(_r):
            return None

    t0 = time.perf_counter()
    res = call()
    block(res)
    first = time.perf_counter() - t0

    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        res = call()
        block(res)
        samples.append(time.perf_counter() - t0)

    out = {'first_call_s': first, 'repeats': len(samples)}
    if samples:
        srt = sorted(samples)
        out.update({
            'median_s': statistics.median(srt),
            'min_s': srt[0],
            'max_s': srt[-1],
            'p25_s': percentile(srt, 0.25),
            'p75_s': percentile(srt, 0.75),
            'compile_over_steady': (first / statistics.median(srt)
                                    if statistics.median(srt) > 0 else None),
        })
        if keep_samples:
            out['samples_s'] = samples
    return out


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------

def git_provenance():
    def git(*args):
        try:
            return subprocess.check_output(['git', '-C', REPO_ROOT] + list(args),
                                           stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None
    dirty = [l.strip() for l in (git('status', '--porcelain') or '').splitlines()
             if l.strip()]
    # Only changes under cymetric/ affect what was measured; untracked files in
    # benchmarks/ are the harness itself.
    lib_dirty = [l for l in dirty if ' cymetric/' in ' ' + l]
    return {
        'repo_root': REPO_ROOT,
        'commit': git('rev-parse', 'HEAD'),
        'commit_subject': git('log', '-1', '--format=%s'),
        'branch': git('rev-parse', '--abbrev-ref', 'HEAD'),
        'dirty_files': len(dirty),
        'library_dirty_files': lib_dirty,
        'library_is_pristine': not lib_dirty,
    }


def nvidia_smi_gpus():
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,name,memory.total,driver_version',
             '--format=csv,noheader'], stderr=subprocess.DEVNULL).decode()
        return [l.strip() for l in out.strip().splitlines()]
    except Exception:
        return None


def library_versions():
    vers = {}
    for mod in ['jax', 'jaxlib', 'equinox', 'optax', 'numpy', 'scipy', 'sympy',
                'torch', 'joblib']:
        try:
            vers[mod] = __import__(mod).__version__
        except Exception:
            vers[mod] = None
    try:
        import cymetric
        vers['cymetric'] = getattr(cymetric, '__version__', None)
        vers['cymetric_file'] = cymetric.__file__
    except Exception as exc:
        vers['cymetric'] = 'import failed: %s' % exc
    return vers


def device_provenance():
    """Device description without hard-failing when jax has no GPU."""
    info = {'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'xla_python_client_preallocate':
                os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE')}
    try:
        import jax
        dev = jax.local_devices()[0]
        info.update({'jax_platform': dev.platform,
                     'jax_device_kind': dev.device_kind,
                     'jax_device_count': len(jax.local_devices())})
    except Exception as exc:
        info['jax_device_error'] = str(exc)
    return info


def provenance(extra=None):
    prov = {
        'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'host': platform.node(),
        'python': platform.python_version(),
        'platform': platform.platform(),
        'cpu_count': os.cpu_count(),
        'git': git_provenance(),
        'versions': library_versions(),
        'device': device_provenance(),
        'gpus_visible_to_nvidia_smi': nvidia_smi_gpus(),
    }
    if extra:
        prov.update(extra)
    return prov


# ---------------------------------------------------------------------------
# device memory
# ---------------------------------------------------------------------------

def device_memory_stats():
    """Peak device memory for the current process, or None on CPU.

    ``memory_stats`` is only implemented for the GPU/TPU backends; on CPU it
    returns None and we record null rather than failing the case.
    """
    try:
        import jax
        dev = jax.local_devices()[0]
        if dev.platform == 'cpu':
            # the CPU backend reports zeros, which would read as "0 bytes used"
            return None
        stats = dev.memory_stats()
        if not stats:
            return None
        return {
            'peak_bytes_in_use': stats.get('peak_bytes_in_use'),
            'bytes_in_use': stats.get('bytes_in_use'),
            'bytes_limit': stats.get('bytes_limit'),
            'peak_mb': (round(stats['peak_bytes_in_use'] / 2 ** 20, 1)
                        if stats.get('peak_bytes_in_use') is not None else None),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# worker: run a single case in this process
# ---------------------------------------------------------------------------

def run_one_case(case, repeats_scale=1.0, quick=False):
    """Run one case definition and return its record."""
    import cases as case_mod  # local import: needs sys.path set up first

    fn = getattr(case_mod, case['fn'])
    params = dict(case.get('params', {}))
    repeats = case.get('repeats', 5)
    if quick:
        repeats = max(2, int(repeats * 0.4))
        params = case_mod.shrink_params(params)
    repeats = max(1, int(round(repeats * repeats_scale)))

    record = {
        'name': case['name'],
        'group': case['group'],
        'what': case['what'],
        'params': params,
        'repeats_requested': repeats,
        'status': 'ok',
        'error': None,
    }
    t0 = time.perf_counter()
    try:
        out = fn(repeats=repeats, **params)
        record.update(out)
    except case_mod.KnownBlocked as exc:
        record['status'] = 'blocked'
        record['error'] = str(exc)
        record['error_type'] = type(exc.__cause__).__name__ if exc.__cause__ else None
    except Exception as exc:
        import traceback
        record['status'] = 'error'
        record['error'] = '%s: %s' % (type(exc).__name__, exc)
        record['traceback'] = traceback.format_exc()[-3000:]
    record['wall_s'] = time.perf_counter() - t0
    record['memory'] = device_memory_stats()
    try:
        import resource
        record['host_peak_rss_mb'] = round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    except Exception:
        record['host_peak_rss_mb'] = None
    record['device'] = device_provenance()
    return record


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run_case_subprocess(case, args):
    """Run a case in a fresh interpreter and collect its JSON record."""
    cmd = [sys.executable, os.path.abspath(__file__), '--_worker',
           json.dumps(case), '--repeats-scale', str(args.repeats_scale)]
    if args.quick:
        cmd.append('--quick')
    env = dict(os.environ)
    env.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    if args.cpu:
        env['JAX_PLATFORMS'] = 'cpu'
        env['CUDA_VISIBLE_DEVICES'] = ''
    try:
        proc = subprocess.run(cmd, env=env, cwd=BENCH_DIR, timeout=args.timeout,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.TimeoutExpired:
        return {'name': case['name'], 'group': case['group'],
                'what': case['what'], 'params': case.get('params', {}),
                'status': 'timeout',
                'error': 'case exceeded --timeout of %s s' % args.timeout}
    out = proc.stdout.decode('utf-8', 'replace')
    err = proc.stderr.decode('utf-8', 'replace')
    for line in out.splitlines():
        if line.startswith(RESULT_MARKER):
            rec = json.loads(line[len(RESULT_MARKER):])
            if args.verbose and err.strip():
                rec['stderr_tail'] = err.strip()[-800:]
            return rec
    return {'name': case['name'], 'group': case['group'], 'what': case['what'],
            'params': case.get('params', {}), 'status': 'error',
            'error': 'worker produced no result (exit %d)' % proc.returncode,
            'stdout_tail': out[-1500:], 'stderr_tail': err[-2500:]}


def fmt_time(s):
    if s is None:
        return '-'
    if s < 1e-3:
        return '%.0f us' % (s * 1e6)
    if s < 1.0:
        return '%.2f ms' % (s * 1e3)
    return '%.2f s' % s


def table(records):
    """Human-readable table of the headline metric of each record."""
    rows = [('case', 'status', 'first call', 'median', 'spread p25-p75',
             'headline', 'peak dev mem')]
    for r in records:
        if r.get('status') != 'ok':
            rows.append((r['name'], r.get('status', '?'), '-', '-', '-',
                         (r.get('error') or '')[:46], '-'))
            continue
        t = r.get('timing', {})
        head = r.get('headline', '')
        mem = r.get('memory') or {}
        rows.append((
            r['name'], 'ok', fmt_time(t.get('first_call_s')),
            fmt_time(t.get('median_s')),
            '%s - %s' % (fmt_time(t.get('p25_s')), fmt_time(t.get('p75_s'))),
            head,
            ('%.0f MB' % mem['peak_mb']) if mem.get('peak_mb') else '-'))
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    lines = []
    for n, row in enumerate(rows):
        lines.append('  '.join(str(c).ljust(widths[i]) for i, c in enumerate(row)).rstrip())
        if n == 0:
            lines.append('  '.join('-' * w for w in widths))
    return '\n'.join(lines)


def main(argv=None):
    import cases as case_mod

    p = argparse.ArgumentParser(
        description='cymetric performance benchmark harness',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('Usage')[-1])
    p.add_argument('--list', action='store_true', help='list cases and exit')
    p.add_argument('--cases', default='.*',
                   help='regex matched against case names (default: all)')
    p.add_argument('--groups', default=None,
                   help='comma-separated group filter, e.g. pointgen,train_jax')
    p.add_argument('--out', default=None,
                   help='write the JSON result file here')
    p.add_argument('--repeats-scale', type=float, default=1.0,
                   help='multiply every case repeat count (default 1.0)')
    p.add_argument('--quick', action='store_true',
                   help='smoke mode: fewer repeats and smaller problem sizes; '
                        'the numbers are not a baseline')
    p.add_argument('--cpu', action='store_true',
                   help='force the CPU backend (also the automatic fallback '
                        'when no GPU is visible)')
    p.add_argument('--timeout', type=float, default=1800,
                   help='per-case wall-clock limit in seconds (default 1800)')
    p.add_argument('--verbose', action='store_true',
                   help='keep worker stderr tails in the records')
    p.add_argument('--table-from', default=None, metavar='JSON',
                   help='print the table of an existing result file and exit')
    p.add_argument('--_worker', default=None, help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    if args.table_from:
        with open(args.table_from) as fh:
            payload = json.load(fh)
        prov = payload['provenance']
        print('%s  commit %s  %s  jax %s  python %s'
              % (prov.get('timestamp_utc'), (prov['git'] or {}).get('commit'),
                 prov['device'].get('jax_device_kind'),
                 prov['versions'].get('jax'), prov.get('python')))
        print()
        print(table(payload['records']))
        return 0

    if args._worker:
        case = json.loads(args._worker)
        rec = run_one_case(case, repeats_scale=args.repeats_scale, quick=args.quick)
        sys.stdout.write(RESULT_MARKER + json.dumps(rec) + '\n')
        return 0

    selected = [c for c in case_mod.CASES if re.search(args.cases, c['name'])]
    if args.groups:
        groups = set(g.strip() for g in args.groups.split(','))
        selected = [c for c in selected if c['group'] in groups]

    if args.list:
        for c in case_mod.CASES:
            print('%-34s %-12s %s' % (c['name'], c['group'], c['what']))
            print('%-34s %-12s params: %s' % ('', '', c.get('params', {})))
        return 0

    if not selected:
        print('no cases matched %r' % args.cases)
        return 1

    prov = provenance({'argv': sys.argv, 'quick': args.quick,
                       'forced_cpu': args.cpu})
    print('cymetric benchmark harness')
    print('  commit   : %s (%s)' % (prov['git']['commit'],
                                    prov['git']['branch']))
    print('  library  : %s' % ('pristine at this commit'
                               if prov['git']['library_is_pristine'] else
                               'MODIFIED: %s' % prov['git']['library_dirty_files']))
    print('  cymetric : %s' % prov['versions'].get('cymetric_file'))
    print('  device   : %s / %s' % (prov['device'].get('jax_platform'),
                                    prov['device'].get('jax_device_kind')))
    print('  jax      : %s   python %s' % (prov['versions'].get('jax'),
                                           prov['python']))
    print('  cases    : %d' % len(selected))
    if prov['device'].get('jax_platform') == 'cpu' and not args.cpu:
        print('  NOTE: no GPU visible to jax; running the CPU fallback. '
              'Timings are not comparable to GPU baselines.')
    print()

    records = []
    for i, case in enumerate(selected, 1):
        print('[%d/%d] %s ...' % (i, len(selected), case['name']), flush=True)
        rec = run_case_subprocess(case, args)
        records.append(rec)
        status = rec.get('status')
        if status == 'ok':
            print('        %s | first call %s | median %s'
                  % (rec.get('headline', ''),
                     fmt_time(rec.get('timing', {}).get('first_call_s')),
                     fmt_time(rec.get('timing', {}).get('median_s'))), flush=True)
        else:
            print('        %s: %s' % (status, (rec.get('error') or '')[:300]),
                  flush=True)

    print()
    print(table(records))

    payload = {'provenance': prov, 'records': records}
    if args.out:
        out = args.out if os.path.isabs(args.out) else os.path.join(REPO_ROOT, args.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w') as fh:
            json.dump(payload, fh, indent=1, sort_keys=True)
        print('\nwrote %s' % out)
    n_ok = sum(1 for r in records if r.get('status') == 'ok')
    print('%d/%d cases ok' % (n_ok, len(records)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
