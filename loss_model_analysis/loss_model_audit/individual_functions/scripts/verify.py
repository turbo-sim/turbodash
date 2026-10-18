"""Verify current-source diagnostics, saved grids, hashes and local navigation."""
import csv
import gzip
import importlib
import json
from pathlib import Path
import re
from urllib.parse import unquote
import numpy as np
from common import BASE,sha256,write_json


def verify(require_figures=True):
    source_manifest=json.loads((BASE/'data/provenance/source_manifest.json').read_text(encoding='utf-8'))
    from common import REPO
    for name,digest in source_manifest['files'].items():
        if name.startswith('turbodash/'):
            assert sha256(REPO/name)==digest, 'Production source changed; recalculate the audit'
    for name in ('profile_terms','profile_interpolation_review','profile_final_review',
                 'secondary_review','clearance_review','trailing_review','ko_safeguards_review','reynolds_review'):
        importlib.import_module(name).verify_saved()
    sweeps=json.loads((BASE/'data/sweep_manifest.json').read_text(encoding='utf-8'))
    assert len(sweeps)==78
    digests={};statuses={}
    for study in sweeps:
        path=BASE/study['data_path']
        with gzip.open(path,'rt',newline='',encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
        assert {r['version'] for r in rows}=={'candidate'}
        assert study['base_points']==200
        assert sum(r['base_grid'].lower()=='true' for r in rows)==200
        x=np.array([float(r['x']) for r in rows]);assert np.all(np.diff(x)>0)
        for r in rows:statuses[r['status']]=statuses.get(r['status'],0)+1
        digests[study['data_path']]=sha256(path)
    broken=[];checked=0
    for document in BASE.rglob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)',document.read_text(encoding='utf-8')):
            target=unquote(target.strip('<>').split('#',1)[0])
            if not target or '://' in target:continue
            path=(document.parent/target).resolve()
            if path in ((BASE/'data/validation/reproduction.json').resolve(),(BASE/'data/provenance/render_manifest.json').resolve()):continue
            if not require_figures and path.suffix == '.pdf':continue
            if not path.exists():broken.append(f'{document.name}: {target}')
            checked+=1
    result=dict(passed=not broken,sweeps=len(sweeps),base_points=200,implementation='current production source',
        statuses=statuses,data_sha256=digests,checked_markdown_links=checked,broken_links=broken,figures_checked=require_figures)
    write_json(BASE/'data/validation/reproduction.json',result)
    assert not broken,broken
    return result


if __name__=='__main__':print(json.dumps(verify(),indent=2))
