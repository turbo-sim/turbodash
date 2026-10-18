"""Compare the current profile refinement with the preceding reviewed audit."""
import ast
import csv
import gzip
import io
import json
import zipfile
from common import BASE, REPO, numeric_digest, sha256, write_json
import numpy as np


def compare():
    path=BASE/'data/history/before_wider_compressibility_blends.zip'
    saved=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    assert sha256(path)==saved['archive_sha256']
    studies=[]
    with zipfile.ZipFile(path) as archive:
        old_sources=saved['source_hashes']
        protected={n:d for n,d in old_sources.items() if n!='turbodash/loss_models_new/loss_model_kacker_okapuu.py'}
        assert all(sha256(REPO/n)==d for n,d in protected.items())
        changes=[n for n,d in old_sources.items() if n.startswith('turbodash/loss_models_new/') and sha256(REPO/n)!=d]
        assert changes==['turbodash/loss_models_new/loss_model_kacker_okapuu.py']
        def executable_functions(source):
            functions = {}
            for node in ast.parse(source).body:
                if isinstance(node, ast.FunctionDef):
                    if (node.body and isinstance(node.body[0], ast.Expr)
                            and isinstance(node.body[0].value, ast.Constant)
                            and isinstance(node.body[0].value.value, str)):
                        node.body = node.body[1:]
                    functions[node.name] = ast.dump(node, include_attributes=False)
            return functions
        prior_functions = executable_functions(archive.read('source_before/loss_model_kacker_okapuu.py').decode('utf-8'))
        current_functions = executable_functions((REPO/changes[0]).read_text(encoding='utf-8'))
        changed_functions = sorted(n for n in prior_functions.keys() | current_functions.keys()
                                   if prior_functions.get(n) != current_functions.get(n))
        assert changed_functions == ['get_compressible_correction_factors'], changed_functions
        for study in json.loads((BASE/'data/sweep_manifest.json').read_text(encoding='utf-8')):
            name=study['data_path']
            old=list(csv.DictReader(io.StringIO(gzip.decompress(archive.read(name)).decode('utf-8'))))
            with gzip.open(BASE/name,'rt',newline='',encoding='utf-8') as stream:new=list(csv.DictReader(stream))
            old_map={(r['version'],float(r['x'])):r for r in old}
            keys=[k for k in old[0] if k not in ('version','x','base_grid') and not k.startswith('d_')]
            common=[(r,old_map[(r['version'],float(r['x']))]) for r in new if (r['version'],float(r['x'])) in old_map]
            count=0; candidate_changed=0
            for now,prior in common:
                if now['version']=='baseline':
                    assert all(now[k]==prior[k] for k in keys),(name,now['x'])
                    count+=len(keys)
                else:candidate_changed+=any(now[k]!=prior[k] for k in keys)
            assert sum(r['base_grid'].lower()=='true' for r,_ in common if r['version']=='baseline')==200
            studies.append(dict(study=study['id'],unchanged_baseline_fields=count,candidate_changed_common_rows=candidate_changed))
        refs=json.loads(archive.read('data/reference_digests.json'))
        retired=set()
        changed=[n for n,d in refs['files'].items() if n not in retired and numeric_digest(BASE/n)!=d]
    result=dict(passed=True,archive=path.relative_to(BASE).as_posix(),archive_sha256=sha256(path),protected_source_files=len(protected),changed_candidate_sources=changes,
        changed_executable_functions=changed_functions,
        standard_sweeps=len(studies),all_baseline_values_and_statuses_unchanged=True,studies=studies,changed_reference_tables=changed,
        note='Baseline comparisons include all 200 original base points per sweep. K1 uses a wider low-Mach join from 0.15 to 0.25 and a rational tail starting at 0.9; K2 saturation starts at ratio 0.8. Join probes follow the new boundaries. The Kp rational continuation, reference fits, reaction/impulse interpolation and thickness treatment are unchanged. Profile and KO-secondary corrections inherit the updated component factors.')
    write_json(BASE/'data/validation/profile_revision.json',result)
    return result

if __name__=='__main__':
    r=compare()
    print(json.dumps({k:v for k,v in r.items() if k!='studies'},indent=2))
